"""Expert-queue plan Q1: event/tick simulator of multi-request MoE decode with offloaded experts.

N decode streams run concurrently (closed loop: each stream replays a trace request from a random offset and
wraps around). Experts live in a slow tier and are cached in fast memory (C experts, LRU over all layers). One
loader moves one expert at a time; the GPU runs one piece of work at a time.

Modes
  sync         synchronous continuous batching: all N tokens go through layer l together -- attention + routing
               for the batch, the missing experts are loaded (demand), then one grouped expert computation; then
               layer l+1. Optional one-layer-ahead prefetch (--la1: the recorded GATE h=1 prediction).
  queue        every token advances on its own; a global heartbeat (period T) batches the ready work of each tick:
                 1. attention + routing for every layer that has ready tokens (one batch per layer);
                    resident experts are computed this tick, missing ones go to the (layer, expert) queue;
                 2. queues with >= n tokens, or whose oldest token waited >= W, are sent to the loader, longest
                    queue first;
                 3. grouped expert computation, per layer, for every token whose expert is resident;
                 4. a token whose k experts are all done moves to the next layer (ready next tick).
               n = 1, W = 0 is "async without accumulation" (load as soon as a token needs it).
Costs (per call, so batching small work pays the fixed parts once): attention(B) from a measured table, grouped
experts(m distinct) = g0 + g1 * (m - 1), a kernel-launch / routing-sync overhead per tick, load = expert bytes / BW.

Ablations inside the synchronous frame: "sync:sub4" (missing experts needed by < 4 tokens of the step are skipped:
lossy, the dropped share of expert uses / routing weight is reported); "sync:pool16:prev:k2" (pool of N streams,
16 advance per step, chosen by route affinity with the previous token's route as the prediction).

  python experiments/moe_edge/tsim_async.py --trace outputs/moe/traces_v3/GLM-4.5-Air_cwq_M0_cot_chat_cap8 \
      --hw h200_pcie --N 16 32 64 --cache 0.5 --modes sync,sync:la1,queue:n1:W0:T0,queue:n8:W2000:T200 --out out.json
"""
import argparse
import glob
import heapq
import json
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np

# Times in microseconds. h200_pcie from experiments/moe/queue_microbench.py (H200, GLM-4.5-Air bf16 shapes):
# attention+projections+router per layer vs batch (S = 2048 context); grouped GEMM 50 us + 12.5 us per extra
# expert, independent of tokens up to 64/expert; one 34.6 MB expert over PCIe ~1.0 ms (35 GB/s).
# mac_ssd: GLM-4.5-Air 4bit on the M5 Max (tsim.py glm4bit): non-expert layer 0.39 ms, 43 us per expert read from
# memory, 1.15 ms to read one 10.4 MB expert from the internal SSD.
HW = {
    "h200_pcie": dict(attn=[(1, 138), (4, 146), (8, 155), (16, 172), (32, 246), (64, 363), (128, 581), (256, 1052)],
                      g0=50.0, g1=12.5, load=1000.0, tick=32.0),
    "mac_ssd": dict(attn=[(1, 390), (8, 420), (32, 520), (64, 650), (128, 900), (256, 1400)],
                    g0=43.0, g1=43.0, load=1150.0, tick=60.0),
}


def attn_cost(hw, B):
    xs, ys = zip(*hw["attn"])
    return float(np.interp(B, xs, ys))


def gmm_cost(hw, m):
    return 0.0 if m == 0 else hw["g0"] + hw["g1"] * (m - 1)


class Cache:
    """C resident experts; evicts the unpinned one with the fewest uses so far (LFU, MoE-Infinity-like). LRU is
    a poor fit here: a synchronous step scans all layers in order, so LRU evicts what the next layers need."""

    def __init__(self, C):
        self.C = C
        self.res = set()
        self.cnt = defaultdict(int)
        self.pin = defaultdict(int)
        self.heap = []                           # (count, key) lazy min-heap over resident keys

    def has(self, key):
        return key in self.res

    def touch(self, key):
        self.cnt[key] += 1
        if key in self.res:
            heapq.heappush(self.heap, (self.cnt[key], key))

    def insert(self, key):
        if key in self.res:
            return
        skipped = []
        while len(self.res) >= self.C and self.heap:
            c, v = heapq.heappop(self.heap)
            if v not in self.res or c != self.cnt[v]:
                continue                         # stale entry
            if self.pin[v]:
                skipped.append((c, v))
                continue
            self.res.discard(v)
        for x in skipped:
            heapq.heappush(self.heap, x)
        self.res.add(key)
        heapq.heappush(self.heap, (self.cnt[key], key))


def warm(cache, streams, E):
    """Start from a full cache: the most used (layer, expert) pairs of these streams (steady state, no cold start)."""
    L = streams[0][0].shape[1]
    f = np.zeros(L * E)
    for R, *_ in streams:
        for l in range(L):
            f[l * E:(l + 1) * E] += np.bincount(R[:, l].ravel(), minlength=E)
    for key in np.argsort(f)[-cache.C:]:
        cache.cnt[int(key)] = int(f[key])
        cache.insert(int(key))


def load_trace(trace, N, seed=0):
    files = sorted(glob.glob(trace + "/q*.npz"))
    streams = []
    rng = np.random.default_rng(seed)
    for f in files:
        z = np.load(f)
        d = z["dec_idx"]
        if d.shape[1] < 64:
            continue
        g = z["dec_gate_h"][0] if "dec_gate_h" in z.files else z["la_dec_idx"]
        R = d.transpose(1, 0, 2).astype(np.int64)                          # [D, L, k]
        if "dec_score" in z.files:                                       # GLM: weight = score / sum of top-k scores
            sc = np.take_along_axis(z["dec_score"].transpose(1, 0, 2).astype(np.float32), R, -1)
            Wt = sc / sc.sum(-1, keepdims=True)
        else:
            Wt = np.full(R.shape, 1.0 / R.shape[-1], np.float32)
        streams.append((R, g.transpose(1, 0, 2).astype(np.int64), Wt))
        if len(streams) >= N:
            break
    offs = [int(rng.integers(0, len(s[0]))) for s in streams]
    return streams, offs


def run_sync(streams, offs, hw, C, E, horizon, la1, B=None, select="fcfs", klim=10 ** 9, sub_n=0):
    """Synchronous decode. B < N: a pool -- every step only B streams advance, chosen by `select`
    (fcfs = longest-waiting; prev / oracle = greedy fewest new (layer, expert) pairs using the previous token's /
    the true route, with streams forced in once they waited ceil(N/B)-1+klim steps). sub_n > 0: a missing expert
    needed by fewer than sub_n tokens of the step is not loaded; those tokens skip it (lossy, counted)."""
    L, k = streams[0][0].shape[1], streams[0][0].shape[2]
    N = len(streams)
    B = B or N
    cache = Cache(C)
    warm(cache, streams, E)
    ready = {}                                                # key -> time its load completes
    t, t_loader = 0.0, 0.0
    pos = list(offs)
    wait = np.zeros(N, int)
    emit = [[] for _ in range(N)]
    gpu_busy = 0.0
    n_loads = 0
    drop_uses = drop_w = all_uses = 0.0

    def route(i, d):                                          # [L, k] route of stream i at token offset d
        R = streams[i][0]
        return R[(pos[i] + d) % len(R)]

    while t < horizon:
        if B >= N:
            act = list(range(N))
        elif select == "fcfs":
            act = list(np.lexsort((np.arange(N), -wait))[:B])
        else:
            lim = int(np.ceil(N / B)) - 1 + klim
            forced = [int(i) for i in np.argsort(-wait) if wait[i] >= lim][:B]
            act = list(forced)
            U = np.zeros((L, E), bool)
            for i in act:
                U[np.arange(L)[:, None], route(i, 0 if select == "oracle" else -1)] = True
            cand = [i for i in range(N) if i not in set(act)]
            while len(act) < B and cand:
                best, bi = None, None
                for i in cand:
                    m = np.zeros((L, E), bool)
                    m[np.arange(L)[:, None], route(i, 0 if select == "oracle" else -1)] = True
                    inc = (m & ~U).sum() - 1e-6 * wait[i]
                    if best is None or inc < best:
                        best, bi, bm = inc, i, m
                act.append(bi)
                U |= bm
                cand.remove(bi)
        for l in range(L):
            a = attn_cost(hw, len(act)) + hw["tick"]
            t += a
            gpu_busy += a
            users = defaultdict(list)
            for i in act:
                for e, w in zip(route(i, 0)[l], streams[i][2][pos[i] % len(streams[i][0]), l]):
                    users[int(e)].append(w)
            all_uses += sum(len(v) for v in users.values())
            if sub_n:
                for e in list(users):
                    if not cache.has(l * E + e) and len(users[e]) < sub_n:
                        drop_uses += len(users[e])
                        drop_w += float(sum(users[e]))
                        del users[e]
            keys = [l * E + e for e in users]
            for key in keys:
                cache.pin[key] += 1
            miss = [key for key in keys if not cache.has(key)]
            for key in miss:                                  # demand loads, after whatever is in flight
                t_loader = max(t_loader, t) + hw["load"]
                ready[key] = t_loader
                cache.insert(key)
                n_loads += 1
            t = max([t] + [ready.get(key, 0.0) for key in keys])   # wait for demand loads and late prefetches
            if la1 and l + 1 < L:                             # prefetch next layer's predicted experts
                pred = set()
                for i in act:
                    R, G = streams[i][0], streams[i][1]
                    pred.update(int(e) for e in G[pos[i] % len(R), l])
                for e in pred:
                    key = (l + 1) * E + e
                    if not cache.has(key):
                        t_loader = max(t_loader, t) + hw["load"]
                        ready[key] = t_loader
                        cache.insert(key)
                        n_loads += 1
            g = gmm_cost(hw, len(users))
            t += g
            gpu_busy += g
            for key in keys:
                cache.touch(key)
                cache.pin[key] -= 1
        sel = np.zeros(N, bool)
        sel[act] = True
        wait[~sel] += 1
        wait[sel] = 0
        for i in act:
            emit[i].append(t)
            pos[i] += 1
    r = summarize(emit, t, n_loads, gpu_busy, hw, horizon)
    r.update(dropped_use_frac=drop_uses / max(all_uses, 1), dropped_weight_per_token_layer=drop_w / max(all_uses / k, 1))
    return r


def run_queue(streams, offs, hw, C, E, horizon, n, W, T):
    L, k = streams[0][0].shape[1], streams[0][0].shape[2]
    N = len(streams)
    cache = Cache(C)
    warm(cache, streams, E)
    pos = list(offs)
    layer = [0] * N                     # current layer of each stream's token
    state = ["attn"] * N                # attn: ready for attention at `layer`; wait: waiting for experts
    pending = [set() for _ in range(N)]  # experts (keys) not yet computed at the current layer
    queues = defaultdict(deque)         # key -> deque of (arrival time, stream)
    loading = set()
    eligible = set()                    # experts allowed to be loaded (threshold or wait limit reached)
    t_loader = 0.0
    stats = dict(n_loads=0)

    def start_load(t0):
        """Start the eligible load with the longest current queue at time t0; returns the loader's free time."""
        best = max(eligible, key=lambda x: (len(queues[x]), -queues[x][0][0] if queues[x] else 0))
        eligible.discard(best)
        if not queues[best] or cache.has(best):
            return t0
        loading.add(best)
        heapq.heappush(inflight, (t0 + hw["load"], best))
        stats["n_loads"] += 1
        return t0 + hw["load"]
    inflight = []                       # heap of (finish time, key)
    emit = [[] for _ in range(N)]
    t, gpu_busy = 0.0, 0.0
    while t < horizon:
        # loads finished by now become resident
        while inflight and inflight[0][0] <= t:
            _, key = heapq.heappop(inflight)
            cache.insert(key)
            loading.discard(key)
        work = hw["tick"]
        # 1. attention + routing, one batch per layer
        by_layer = defaultdict(list)
        for i in range(N):
            if state[i] == "attn":
                by_layer[layer[i]].append(i)
        for l, members in by_layer.items():
            work += attn_cost(hw, len(members))
            for i in members:
                R = streams[i][0]
                pending[i] = {l * E + int(e) for e in R[pos[i] % len(R), l]}
                state[i] = "wait"
                for key in pending[i]:
                    cache.pin[key] += 1
                    if not cache.has(key):
                        queues[key].append((t, i))
        # 2. trigger loads: queues past the threshold (or the wait limit) become eligible; whenever the loader
        #    is free it takes the eligible expert with the LONGEST queue right now (most tokens per byte read)
        for key, q in queues.items():
            if q and key not in loading and not cache.has(key) and (len(q) >= n or t - q[0][0] >= W):
                eligible.add(key)
        while eligible and t_loader <= t + 1e-9:
            t_loader = start_load(max(t_loader, t))
        # 3. grouped expert computation per layer for resident experts
        comp = defaultdict(set)
        for i in range(N):
            if state[i] == "wait":
                done = {key for key in pending[i] if cache.has(key)}
                if done:
                    comp[layer[i]].update(done)
                    for key in done:
                        cache.touch(key)
                        cache.pin[key] -= 1
                        if queues.get(key):
                            queues[key] = deque(x for x in queues[key] if x[1] != i)
                    pending[i] -= done
        for l, keys in comp.items():
            work += gmm_cost(hw, len(keys))
        # 4. advance tokens whose experts are all done
        for i in range(N):
            if state[i] == "wait" and not pending[i]:
                if layer[i] == L - 1:
                    emit[i].append(t + work)
                    pos[i] += 1
                    layer[i] = 0
                else:
                    layer[i] += 1
                state[i] = "attn"
        gpu_busy += work
        dt = max(T, work)
        # if nothing could run, jump to the next load completion
        if work <= hw["tick"] + 1e-9 and inflight:
            dt = max(dt, inflight[0][0] - t)
        elif work <= hw["tick"] + 1e-9 and eligible:
            dt = max(dt, t_loader - t)
        t += dt
        # loader keeps working between ticks
        while eligible and t_loader < t:
            t_loader = start_load(t_loader)
    return summarize(emit, t, stats["n_loads"], gpu_busy, hw, horizon)


def summarize(emit, t_end, n_loads, gpu_busy, hw, horizon):
    toks = sum(len(e) for e in emit)
    gaps = np.concatenate([np.diff(e) for e in emit if len(e) > 1]) if toks > len(emit) else np.zeros(1)
    return dict(tokens=toks, throughput_tok_s=toks / (t_end / 1e6), tpot_p50_ms=float(np.percentile(gaps, 50)) / 1e3,
                tpot_p99_ms=float(np.percentile(gaps, 99)) / 1e3, loads_per_token=n_loads / max(toks, 1),
                gpu_busy=gpu_busy / t_end, loader_busy=min(1.0, n_loads * hw["load"] / t_end))


def job(args):
    trace, hwname, N, frac, mode, E, horizon = args
    hw = HW[hwname]
    streams, offs = load_trace(trace, N)
    L = streams[0][0].shape[1]
    C = int(frac * L * E)
    if mode.startswith("sync"):
        # sync[:la1][:sub<n>][:pool<B>:<fcfs|prev|oracle>[:k<klim>]]
        parts = mode.split(":")[1:]
        kw = dict(la1="la1" in parts)
        for x in parts:
            if x.startswith("sub"):
                kw["sub_n"] = int(x[3:])
            elif x.startswith("pool"):
                kw["B"] = int(x[4:])
            elif x in ("fcfs", "prev", "oracle"):
                kw["select"] = x
            elif x.startswith("k") and x[1:].isdigit():
                kw["klim"] = int(x[1:])
        r = run_sync(streams, offs, hw, C, E, horizon, **kw)
    else:
        p = dict(x.split("=") if "=" in x else (x[0], x[1:]) for x in mode.split(":")[1:])
        r = run_queue(streams, offs, hw, C, E, horizon, n=int(p["n"]), W=float(p["W"]), T=float(p["T"]))
    r.update(hw=hwname, N=N, cache=frac, mode=mode)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--hw", nargs="+", default=["h200_pcie"])
    ap.add_argument("--N", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--cache", type=float, nargs="+", default=[0.5])
    ap.add_argument("--modes", default="sync,sync:la1,queue:n1:W0:T0,queue:n4:W2000:T200,queue:n8:W4000:T200")
    ap.add_argument("--E", type=int, default=128)
    ap.add_argument("--horizon_ms", type=float, default=20000)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jobs = [(a.trace, h, N, c, m, a.E, a.horizon_ms * 1e3) for h in a.hw for N in a.N for c in a.cache
            for m in a.modes.split(",")]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(job, jobs))
    json.dump(rows, open(a.out, "w"), indent=1)
    for r in rows:
        print(f"{r['hw']:9} N={r['N']:3} cache {r['cache']:.0%} {r['mode']:24} {r['throughput_tok_s']:7.1f} tok/s  "
              f"TPOT p50 {r['tpot_p50_ms']:7.1f} p99 {r['tpot_p99_ms']:7.1f} ms  loads/tok {r['loads_per_token']:6.2f}  "
              f"GPU {r['gpu_busy']:.0%} loader {r['loader_busy']:.0%}  ({r['tokens']} tok)", flush=True)


if __name__ == "__main__":
    main()
