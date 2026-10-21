"""Trace-driven simulator for E2 (affinity admission), E3 (queue-lookahead expert cache) and
E4 (which experts to keep after an offloaded prefill). Single GPU, routed experts cached in HBM
(capacity C experts), the rest in pinned host memory (PCIe ~54 GB/s).

Real routing comes from the E0 traces; request-level predictions come from
`e1_analysis.py --export` (<model>_<workload>_pred.npz).

mode=cache  (E3/E4) requests are served one at a time in arrival order (batch 1): prefill touches
            every expert its prompt routes to, layer by layer, then decode steps touch the actual
            top-k per layer. Eviction policies:
              LRU, LFU, MOEINF (uses within the current request so far),
              PRED:<p>:<Q> (predicted decode distribution of the current request + the next Q queued
                            requests, discounted 0.7^j), BELADY (true future accesses),
              HYB:<p>:<Q>:<lam> (MOEINF's within-request use count + lam x the expected decode uses
                            k*D*pred of the current and next Q requests: a prior that MOEINF lacks).
              SW:<p>:<lam> (E4: GLOBAL scores during the prefill sweep, which keeps the globally hot experts
                            resident instead of thrashing, then at the first decode step re-score with
                            HYB:<p>:0:<lam> -- i.e. decide what to keep for decode once the prefill is done).
            MIX<a> predictors = a*ORACLE + (1-a)*GLOBAL, to map predictor quality -> bytes saved.
            Reports bytes moved (prefill / decode), decode hit rate, decode misses in the first 8 steps.
mode=batch  (E2) Poisson arrivals, continuous batching up to Bmax requests. Each step decodes one
            token per active request; per layer the step needs the union of the actual experts.
            Admission when a slot frees: FCFS, AFF:<p> (waiting request whose predicted distribution
            is most similar to the active batch's), forced FCFS once the oldest has waited tau steps.
            Cache policy LRU. Reports throughput, bytes per token, latency, waiting time.

Cost model (per step and layer): load = missing experts x expert_bytes / PCIe;
compute = sum over needed experts of max(expert_bytes / HBM, 2 x n x expert_params / peak) + launch.
Non-expert weights are read once per step from HBM.

  python experiments/moe/sim.py --model GLM-4.5-Air --workload W1-cwq --mode cache --cap 0.3
"""

import argparse
import glob
import heapq
import json
import os
import random
from collections import Counter, OrderedDict, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PCIE, HBM, PEAK, LAUNCH = 54e9, 4.8e12, 989e12, 5e-6
# routed-expert params (3 x hidden x moe_inter) and non-expert bytes (bf16), from configs / checkpoint sizes
MODELS = {
    "GLM-4.5-Air": dict(exp_params=3 * 4096 * 1408, nonexp_bytes=221e9 - 199.3e9),
    "Mixtral-8x22B-Instruct-v0.1": dict(exp_params=3 * 6144 * 16384, nonexp_bytes=281e9 - 270.6e9),
    "Qwen3-235B-A22B": dict(exp_params=3 * 4096 * 1536, nonexp_bytes=470e9 - 454.2e9),
    "DeepSeek-V2.5": dict(exp_params=3 * 5120 * 1536, nonexp_bytes=471e9 - 445.4e9),
    "Qwen3-30B-A3B": dict(exp_params=3 * 2048 * 768, nonexp_bytes=61e9 - 3 * 2048 * 768 * 2 * 48 * 128),
}
WL = {"W1-cwq": "cwq_M0_cap8", "W1-webqsp": "webqsp_M0_cap8", "Wsyn-cwq": "cwq_M0_syn_cap8",
      "Wsynr-cwq": "cwq_M0_synr_cap8", "W2-cwq": "cwq_M0_cot_cap8"}


def load(model, workload, traces, e1dir):
    reqs = []
    for f in sorted(glob.glob(os.path.join(traces, f"{model}_{WL[workload]}", "q*.npz"))):
        z = np.load(f, allow_pickle=False)
        reqs.append(dict(pre=z["pre_idx"].astype(np.int32), dec=z["dec_idx"].astype(np.int32), q=str(z["question"])))
    pred = np.load(os.path.join(e1dir, f"{model}_{workload}_pred.npz"))
    assert [r["q"] for r in reqs] == list(pred["questions"]), "trace / prediction order mismatch"
    preds = {n: pred[n].astype(np.float32) for n in pred.files if n != "questions"}
    for a in (5, 10, 25, 50, 75, 90):
        m = a / 100 * preds["ORACLE"] + (1 - a / 100) * preds["GLOBAL"]
        preds[f"MIX{a}"] = m / m.sum(-1, keepdims=True)
    return reqs, preds


def pred_recall(reqs, P, B):
    """Token-level recall@B of a request-level prediction P [N, L, E] against the actual decode experts."""
    hit = tot = 0
    for r, p in zip(reqs, P):
        top = np.argpartition(-p, B - 1, axis=-1)[:, :B]                 # [L, B]
        for s in range(r["dec"].shape[1]):
            a = r["dec"][:, s]                                            # [L, k]
            hit += int((top[:, :, None] == a[:, None, :]).any(1).sum())
            tot += a.size
    return hit / max(1, tot)


def expert_time(n, eb, ep):
    return max(eb / HBM, 2.0 * n * ep / PEAK) + LAUNCH


# ----------------------------------------------------------------------------- E3 / E4: cache
def access_stream(reqs):
    """Per request: list of (phase, step, layer, expert) in execution order."""
    out = []
    for r in reqs:
        acc = []
        L = r["pre"].shape[0]
        for l in range(L):
            for e in np.unique(r["pre"][l]):
                acc.append(("p", -1, l, int(e)))
        for s in range(r["dec"].shape[1]):
            for l in range(L):
                for e in r["dec"][l, s]:
                    acc.append(("d", s, l, int(e)))
        out.append(acc)
    return out


def run_cache(reqs, preds, policy, C, E, eb):
    streams = access_stream(reqs)
    flat = [(ri, a) for ri, acc in enumerate(streams) for a in acc]
    nxt_use = None
    if policy == "BELADY":
        nxt_use = [0] * len(flat)
        last = {}
        for i in range(len(flat) - 1, -1, -1):
            key = (flat[i][1][2], flat[i][1][3])
            nxt_use[i] = last.get(key, 1 << 60)
            last[key] = i
    pname, Q, lam = None, 0, 0.0
    if policy.startswith("PRED:"):
        _, pname, Q = policy.split(":")
        Q = int(Q)
    elif policy.startswith("HYB:") or policy.startswith("SW:"):
        if policy.startswith("SW:"):
            _, pname, lam = policy.split(":")
            Q, lam = 0, float(lam)
        else:
            _, pname, Q, lam = policy.split(":")
            Q, lam = int(Q), float(lam)
        k = reqs[0]["dec"].shape[-1]
        kD = k * float(np.mean([r["dec"].shape[1] for r in reqs]))   # expected decode uses per layer
    cache = OrderedDict()                       # key -> None, ordered by recency (LRU)
    freq, req_use = Counter(), Counter()
    heap, cur = [], {}                          # lazy min-heap of (score, tiebreak, key); cur = live score
    stats = defaultdict(float)

    in_prefill = [True]

    def score(ri, key, i):
        if policy.startswith("SW:") and in_prefill[0]:
            return float(preds["GLOBAL"][ri][key[0], key[1]]) - 1e3   # below every decode-time score
        if policy == "LFU":
            return float(freq[key])
        if policy == "MOEINF":
            return req_use[key] * 1e6 + freq[key]
        if policy == "BELADY":
            return -float(nxt_use[i])            # farthest next use = smallest score = evicted first
        l, e = key                               # PRED / HYB
        if policy.startswith(("HYB:", "SW:")):
            prior = sum((0.7 ** j) * preds[pname][ri + j][l, e] for j in range(Q + 1) if ri + j < len(reqs))
            return req_use[key] + lam * kD * prior + 1e-6 * freq[key]
        return sum((0.7 ** j) * preds[pname][ri + j][l, e] for j in range(Q + 1) if ri + j < len(reqs))

    def push(key, sc, i):
        cur[key] = sc
        heapq.heappush(heap, (sc, i, key))

    cur_req = -1
    for i, (ri, (ph, s, l, e)) in enumerate(flat):
        key = (l, e)
        if ri != cur_req:
            cur_req = ri
            req_use.clear()
            in_prefill[0] = True
            if policy in ("MOEINF",) or pname:   # per-request scores: rebuild the heap
                heap, cur = [], {}
                for k_ in cache:
                    push(k_, score(ri, k_, i), i)
        if ph == "d" and in_prefill[0]:
            in_prefill[0] = False
            if policy.startswith("SW:"):         # prefill done: re-score what is resident for the decode
                heap, cur = [], {}
                for k_ in cache:
                    push(k_, score(ri, k_, i), i)
        stats[f"{ph}_acc"] += 1
        if key in cache:
            cache.move_to_end(key)
        else:
            stats[f"{ph}_miss"] += 1
            if ph == "d" and s < 8:
                stats["d_miss_first8"] += 1
            if len(cache) >= C:
                if policy == "LRU":
                    victim = next(iter(cache))
                else:
                    while True:
                        sc, _, kk = heapq.heappop(heap)
                        if kk in cache and cur.get(kk) == sc:
                            victim = kk
                            break
                del cache[victim]
                cur.pop(victim, None)
            cache[key] = None
        freq[key] += 1
        req_use[key] += 1
        if policy != "LRU":
            push(key, score(ri, key, i), i)
    n = len(reqs)
    return dict(policy=policy, C=C, pred_recall_2k=(pred_recall(reqs, preds[pname], 2 * reqs[0]["dec"].shape[-1])
                                                    if pname else None),
                prefill_GB_per_req=stats["p_miss"] * eb / 1e9 / n, decode_GB_per_req=stats["d_miss"] * eb / 1e9 / n,
                decode_hit=1 - stats["d_miss"] / max(1, stats["d_acc"]),
                decode_miss_first8_per_req=stats["d_miss_first8"] / n,
                decode_stall_ms_per_req=stats["d_miss"] * eb / PCIE * 1e3 / n)


# ----------------------------------------------------------------------------- E2: batching
def run_batch(reqs, preds, policy, Bmax, rate_per_step, tau, C, E, eb, ep, nonexp, seed=0, group=1):
    rnd = random.Random(seed)
    N = len(reqs)
    order = list(range(N))
    rnd.shuffle(order)
    arrivals, t = [], 0.0
    for _ in order:
        t += rnd.expovariate(rate_per_step)
        arrivals.append(t)
    pname = policy.split(":")[1] if policy.startswith("AFF:") else None
    cache = OrderedDict()
    waiting, active, done = [], [], {}
    nxt_arr, step, clock = 0, 0, 0.0
    tokens, loaded, stall_steps = 0, 0, 0
    dec_loaded, dec_tokens, dec_time = 0, 0, 0.0             # steps with no prefill in them
    arrive_clock, arrive_step = {}, {}
    while len(done) < N:
        while nxt_arr < N and arrivals[nxt_arr] <= step:
            w = order[nxt_arr]
            waiting.append(w); arrive_clock[w] = clock; arrive_step[w] = step; nxt_arr += 1
        # admission (group > 1: wait until `group` slots are free so that prefills share one expert sweep,
        # MoE-Gen-style; never hold back when the engine is idle or the oldest request waited > 32 steps)
        hold = bool(group > 1 and active and waiting and Bmax - len(active) < group
                    and step - arrive_step[waiting[0]] <= 32)
        while len(active) < Bmax and waiting and not hold:
            pick = 0
            if pname and active and len(waiting) > 1:
                agg = sum(preds[pname][a["r"]] for a in active).ravel()
                agg = agg / (np.linalg.norm(agg) + 1e-9)
                sims = [float(preds[pname][w].ravel() @ agg / (np.linalg.norm(preds[pname][w]) + 1e-9)) for w in waiting]
                pick = int(np.argmax(sims))
                w0 = waiting[0]
                if step - arrive_step[w0] > tau:
                    pick = 0                     # starvation guard: oldest first
            r = waiting.pop(pick)
            active.append(dict(r=r, s=0, prefilled=False, admit_clock=clock))
        if not active:
            step += 1
            continue
        # one engine step: prefill newly admitted requests, decode one token for the others
        L = reqs[0]["pre"].shape[0]
        step_time = nonexp / HBM
        has_prefill = any(not a["prefilled"] for a in active)
        step_loaded = 0
        for l in range(L):
            need = Counter()
            for a in active:
                rq = reqs[a["r"]]
                if not a["prefilled"]:
                    for e, c in zip(*np.unique(rq["pre"][l], return_counts=True)):
                        need[int(e)] += int(c)
                elif a["s"] < rq["dec"].shape[1]:
                    for e in rq["dec"][l, a["s"]]:
                        need[int(e)] += 1
            miss = 0
            for e in need:
                key = (l, e)
                if key in cache:
                    cache.move_to_end(key)
                else:
                    miss += 1
                    if len(cache) >= C:
                        cache.popitem(last=False)
                    cache[key] = None
            loaded += miss
            step_loaded += miss
            step_time += miss * eb / PCIE + sum(expert_time(n, eb, ep) for n in need.values())
        clock += step_time
        if not has_prefill:
            dec_loaded += step_loaded
            dec_tokens += len(active)
            dec_time += step_time
        nxt_active = []
        for a in active:
            rq = reqs[a["r"]]
            if not a["prefilled"]:
                a["prefilled"] = True
            else:
                a["s"] += 1
                tokens += 1
            if a["prefilled"] and a["s"] >= rq["dec"].shape[1]:
                done[a["r"]] = dict(latency=clock - arrive_clock[a["r"]], wait=a["admit_clock"] - arrive_clock[a["r"]])
            else:
                nxt_active.append(a)
        active = nxt_active
        step += 1
    lat = [d["latency"] for d in done.values()]
    wt = [d["wait"] for d in done.values()]
    return dict(policy=policy, Bmax=Bmax, tau=tau, rate=rate_per_step, C=C, group=group,
                throughput_tok_s=tokens / clock, GB_per_token=loaded * eb / 1e9 / max(1, tokens),
                decode_GB_per_token=dec_loaded * eb / 1e9 / max(1, dec_tokens),
                decode_tok_s=dec_tokens / max(dec_time, 1e-9),
                lat_mean_s=float(np.mean(lat)), lat_p99_s=float(np.percentile(lat, 99)),
                wait_mean_s=float(np.mean(wt)), sim_seconds=clock)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--workload", default="W1-cwq")
    ap.add_argument("--mode", choices=["cache", "batch"], required=True)
    ap.add_argument("--traces", default=os.path.join(ROOT, "outputs/moe/traces"))
    ap.add_argument("--e1dir", default=os.path.join(ROOT, "outputs/moe/e1"))
    ap.add_argument("--caps", default="0.1,0.3,0.5,0.7", help="cache capacity as fraction of all routed experts")
    ap.add_argument("--policies", default=None)
    ap.add_argument("--bmax", default="4,8,16,32")
    ap.add_argument("--taus", default="0,8,32")
    ap.add_argument("--rate", type=float, default=None, help="arrivals per engine step (default: saturating)")
    ap.add_argument("--group", type=int, default=1, help="batch mode: admit only when this many slots are free")
    ap.add_argument("--n", type=int, default=None, help="use the first n requests")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    reqs, preds = load(args.model, args.workload, args.traces, args.e1dir)
    if args.n:
        reqs, preds = reqs[:args.n], {k: v[:args.n] for k, v in preds.items()}
    L, E = reqs[0]["pre"].shape[0], preds["GLOBAL"].shape[-1]
    ep = MODELS[args.model]["exp_params"]
    eb, nonexp = 2 * ep, MODELS[args.model]["nonexp_bytes"]
    total = L * E
    rows = []
    if args.mode == "cache":
        pols = (args.policies.split(",") if args.policies else
                ["LRU", "LFU", "MOEINF", "PRED:GLOBAL:0", "PRED:RPP:0", "PRED:OURS_ADMIT:0", "PRED:PREFILL:0",
                 "PRED:BASE_ADMIT:4", "PRED:OURS_ADMIT:4", "PRED:ORACLE:0", "PRED:ORACLE:4", "BELADY"])
        for cf in [float(x) for x in args.caps.split(",")]:
            C = max(E, int(cf * total))
            for p in pols:
                r = run_cache(reqs, preds, p, C, E, eb)
                r["cap_frac"] = cf
                rows.append(r)
                print(json.dumps(r), flush=True)
    else:
        pols = args.policies.split(",") if args.policies else ["FCFS", "AFF:BASE_ADMIT", "AFF:OURS_ADMIT", "AFF:ORACLE"]
        mean_dec = np.mean([r["dec"].shape[1] for r in reqs])
        for cf in [float(x) for x in args.caps.split(",")]:
            C = max(E, int(cf * total))
            for Bm in [int(x) for x in args.bmax.split(",")]:
                rate = args.rate or 1.5 * Bm / (mean_dec + 1)          # keep a queue so admission has a choice
                for tau in [int(x) for x in args.taus.split(",")]:
                    for p in pols:
                        r = run_batch(reqs, preds, p, Bm, rate, tau, C, E, eb, ep, nonexp, group=args.group)
                        r["cap_frac"] = cf
                        rows.append(r)
                        print(json.dumps(r), flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(rows, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
