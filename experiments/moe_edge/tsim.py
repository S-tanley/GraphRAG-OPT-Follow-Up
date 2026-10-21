"""Time-stepped simulator for single-user MoE offloading (scenario S1 of the edge plan: E3 / E4).

Hardware model (Mac, see plan §1): routed experts live in a slow tier (internal SSD, ~9 GB/s) and are cached
in unified memory (capacity C experts); everything else is resident. One loader moves one expert at a time
(bandwidth BW); the compute engine runs layer by layer. Times are calibrated with MLX measurements:
  decode layer time  = (non-expert bytes per layer + used experts x expert bytes) / MEM_BW_eff
  prefill layer time = T x active params per layer x C_PRE  (per-token, per-B-params cost measured on MLX)
A layer's MoE can start only when all its experts are resident; waiting is "stall".

Loader priorities: demand (the current layer needs it) > lookahead (predicted for the next layer) >
background (request- or segment-level prediction). A background load only evicts a resident expert whose
score is lower than the new expert's score; demand / lookahead loads evict the lowest-score expert. Experts
used by the running layer and in-flight loads are pinned.

Per request: an idle window W (retrieval / planning) precedes prefill -- the loader may work during it if a
T0 (admission-time) prior is used. Prefill streams each layer's needed experts with a P-layer prefetch
window. Decode steps go layer by layer.

Policies (--policy, comma separated, each "evict/prefetch/prior"):
  evict:    lru | lfu (global use count; scan-resistant) | moeinf (within-request use count, then global
            freq) | hyb (moeinf + lam * k*D * prior) | sw-<mode> (lfu during prefill, <mode> during decode)
  prefetch: none | la1 (one-layer-ahead gate prediction, la_dec in the trace)
  prior:    none | T0:<name> (known at arrival: background prefetch in the window + hyb scores)
            | T1:<name> (known after prefill) | T01:<t0 name>:<t1 name> (T0 at arrival, replaced by T1 after
            prefill) | seg<S> (previous-segment profile, refreshed every S steps)
            names: GLOBAL, ORACLE, CANDFIRST, FIRSTTOK, CANDDEC (T1), or any array in --preds npz (T0)

  python experiments/moe_edge/tsim.py --trace outputs/moe/traces/GLM-4.5-Air_cwq_M0_cap8 --hw glm4bit \
      --budget_gb 41 --policy lru/none/none,moeinf/la1/none,hyb/la1/T1:CANDFIRST --out out.json
"""

import argparse
import glob
import heapq
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "moe"))
from e1_analysis import counts, load_run, norm  # noqa: E402

# --- hardware / model constants (bytes, seconds); MLX-measured effective bandwidths, see plan §1
HW = {
    # GLM-4.5-Air, MLX 4bit (group 64): expert 3*4096*1408 params * 0.5625 B; non-expert ~4.3 GB over 46 layers
    "glm4bit": dict(eb=3 * 4096 * 1408 * 0.5625, nonexp_layer=4.3e9 / 46, act_params_layer=0.27e9,
                    mem_bw=242e9, c_pre=85e-6 / 1e9, ssd=9e9),
    # A tier (fit in memory; offload emulated with --cache_frac). Calibrated against MLX decode: OLMoE bf16
    # 0.40 ms/layer measured (model 0.33), Qwen3-30B-A3B 4bit 0.16 ms/layer measured (model 0.13).
    # c_pre from MLX prefill of 256 tokens: OLMoE 146 us, Qwen3 4bit 104 us per token per 1e9 active params.
    "olmoe": dict(eb=3 * 2048 * 1024 * 2, nonexp_layer=4 * 2048 * 2048 * 2, act_params_layer=67e6,
                  mem_bw=410e9, c_pre=146e-6 / 1e9, ssd=9e9),
    "qwen3_30b_4bit": dict(eb=3 * 2048 * 768 * 0.5625, nonexp_layer=19e6 * 0.5625, act_params_layer=57e6,
                           mem_bw=242e9, c_pre=104e-6 / 1e9, ssd=9e9),
    # DeepSeek-V2-Lite bf16: MLA attention ~14M params + 2 shared experts (17.3M) resident per layer
    "dslite": dict(eb=3 * 2048 * 1408 * 2, nonexp_layer=31e6 * 2, act_params_layer=83e6,
                   mem_bw=410e9, c_pre=146e-6 / 1e9, ssd=9e9),
    "glm4bit_ext": dict(eb=3 * 4096 * 1408 * 0.5625, nonexp_layer=4.3e9 / 46, act_params_layer=0.27e9,
                        mem_bw=242e9, c_pre=85e-6 / 1e9, ssd=1.1e9),
}


class Sim:
    """Expert cache + one loader with three FIFO queues (0 demand, 1 lookahead, 2 background)."""

    def __init__(self, L, E, k, C, hw, lam=1.0, kD=48.0):
        self.L, self.E, self.k, self.C, self.hw, self.lam, self.kD = L, E, k, C, hw, lam, kD
        n = L * E
        self.res = np.zeros(n, bool)              # resident
        self.queued = np.full(n, -1, np.int8)     # queue priority if waiting to be loaded, else -1
        self.ready = np.zeros(n)                  # time a resident expert became usable
        self.last = np.zeros(n)                   # last use (LRU)
        self.freq = np.zeros(n)                   # global use count
        self.ruse = np.zeros(n)                   # uses within the current request
        self.prior = np.zeros(n)                  # current request's predicted decode distribution
        self.pin = np.zeros(n, bool)
        self.used = np.zeros(n, bool)             # used since it was loaded
        self.n_res = 0
        self.t_loader = 0.0
        self.qs = [[], [], []]                    # heaps of (issue_time, seq, key)
        self.seq = 0
        self.dur = hw["eb"] / hw["ssd"]
        self.evict_mode = "lru"
        self.stats = dict(loaded=0, demand=0, la=0, bg=0, bg_dropped=0, wasted=0)

    def score(self, keys):
        if self.evict_mode == "lru":
            return self.last[keys]
        if self.evict_mode == "lfu":                # global use count: scan-resistant; experts the
            return self.freq[keys] + 1e9 * self.prior[keys]   # current prior predicts for decode are kept
        if self.evict_mode == "moeinf":
            return self.ruse[keys] * 1e6 + self.freq[keys]
        return self.ruse[keys] + self.lam * self.kD * self.prior[keys] + 1e-6 * self.freq[keys]

    def issue(self, keys, prio, t):
        for key in np.asarray(keys).ravel():
            key = int(key)
            if self.res[key]:
                continue
            q = self.queued[key]
            if q != -1 and q <= prio:
                continue                          # already queued at the same or a higher priority
            # (a lower-priority queued copy stays in its heap and is skipped when popped: queued[] changed)
            self.queued[key] = prio
            self.seq += 1
            heapq.heappush(self.qs[prio], (t, self.seq, key))

    def clear_background(self):
        for _t, _s, key in self.qs[2]:
            if self.queued[key] == 2:
                self.queued[key] = -1
        self.qs[2] = []

    def _pop(self):
        """Next item the loader can start at t_loader (highest priority among those already issued);
        if nothing has been issued yet, the earliest future one. Skips stale entries."""
        while True:
            best = None
            for p in range(3):
                h = self.qs[p]
                while h and (self.queued[h[0][2]] != p or self.res[h[0][2]]):   # stale entries
                    heapq.heappop(h)
                if h and h[0][0] <= self.t_loader:
                    best = p
                    break
            if best is None:
                heads = [(h[0][0], p) for p, h in enumerate(self.qs) if h]
                if not heads:
                    return None
                return ("wait", min(heads)[0])
            t_i, _s, key = heapq.heappop(self.qs[best])
            return ("load", best, key)

    def _load(self, prio, key):
        self.queued[key] = -1
        if self.res[key]:
            return
        if self.n_res >= self.C:
            cand = np.flatnonzero(self.res & ~self.pin)
            if len(cand) == 0:
                return
            sc = self.score(cand)
            j = int(np.argmin(sc))
            if prio == 2 and sc[j] >= self.score(np.array([key]))[0]:
                self.stats["bg_dropped"] += 1   # not worth evicting anything for this prefetch
                return
            v = int(cand[j])
            self.res[v] = False
            self.n_res -= 1
            if not self.used[v]:
                self.stats["wasted"] += 1
        self.t_loader += self.dur
        self.res[key] = True
        self.n_res += 1
        self.ready[key] = self.t_loader
        self.used[key] = False
        self.stats["loaded"] += 1
        self.stats[("demand", "la", "bg")[prio]] += 1

    def advance(self, until, stop_keys=None):
        """Run the loader until time `until`; with stop_keys, run (past `until` if necessary) until all of
        them are resident."""
        while True:
            if stop_keys is not None and self.res[stop_keys].all():
                return
            if stop_keys is None and self.t_loader >= until:
                return
            nxt = self._pop()
            if nxt is None:
                if stop_keys is not None:          # a needed expert is missing and nothing is queued
                    self.issue(stop_keys[~self.res[stop_keys]], 0, self.t_loader)
                    continue
                self.t_loader = max(self.t_loader, until)
                return
            if nxt[0] == "wait":
                if stop_keys is None and nxt[1] >= until:
                    self.t_loader = max(self.t_loader, until)
                    return
                self.t_loader = max(self.t_loader, nxt[1])
                continue
            self._load(nxt[1], nxt[2])

    def need(self, keys, t):
        """The layer computing at time t needs `keys`; returns when they are all usable (>= t)."""
        keys = np.asarray(keys).ravel()
        self.advance(t)                            # whatever the loader did up to now
        self.pin[:] = False
        self.pin[keys] = True
        self.issue(keys, 0, t)
        self.advance(t, stop_keys=keys)
        done = max(t, float(self.ready[keys].max()))
        self.last[keys] = done
        self.used[keys] = True
        return done


def run(reqs, policy, hw, C, E, W, P_pre, seg_tables, preds, lam, prefill_tokens="all"):
    evict, pref, prior = policy.split("/")
    L, k = reqs[0]["pre"].shape[0], reqs[0]["pre"].shape[-1]
    kD = k * float(np.mean([r["dec"].shape[1] for r in reqs]))
    sim = Sim(L, E, k, C, hw, lam, kD)
    pre_mode, dec_mode = ("lfu", evict[3:]) if evict.startswith("sw-") else (evict, evict)
    t = 0.0
    out = dict(ttft=[], tpot=[], stall_dec=[], stall_pre=[], first8=[], e2e=[])
    lt_dec = lambda n: (hw["nonexp_layer"] + n * hw["eb"]) / hw["mem_bw"]
    for ri, r in enumerate(reqs):
        sim.ruse[:] = 0
        sim.prior[:] = 0
        t_arr = t
        # ---- idle / retrieval window W before prefill: T0 prior can be loaded in the background
        sim.clear_background()
        sim.evict_mode = pre_mode
        if prior.startswith(("T0:", "T01:")):
            p = preds[prior.split(":")[1]][ri].reshape(-1)
            sim.prior[:] = p
            order = np.argsort(-p)[:C]
            sim.issue(order[p[order] > 0], 2, t)
        t += W
        sim.advance(t)
        # ---- prefill (E5: tokens whose KV is reused from a cache do not go through the MoE)
        sim.evict_mode = pre_mode
        cat = r["cat"]
        if prefill_tokens == "body":                 # fixed few-shot prefix cached (exact prefix caching)
            sel = [i for i, c in enumerate(cat) if c != "pre"]
        elif prefill_tokens == "question":           # prefix + retrieved paths cached (path-KV reuse)
            sel = [i for i, c in enumerate(cat) if c == "question"]
        else:
            sel = list(range(len(cat)))
        pre_idx = r["pre"][:, sel] if len(sel) else r["pre"][:, -1:]
        T = pre_idx.shape[1]
        stall_pre = 0.0
        for l in range(L):
            for ll in range(l + 1, min(L, l + 1 + P_pre)):          # prefetch window over next layers
                sim.issue(ll * E + np.unique(pre_idx[ll]), 1, t)
            keys = l * E + np.unique(pre_idx[l])
            t_ready = sim.need(keys, t)
            stall_pre += t_ready - t
            t = t_ready + T * hw["act_params_layer"] * hw["c_pre"]
            sim.ruse[keys] += 1
            sim.freq[keys] += 1
        out["ttft"].append(t - t_arr)
        out["stall_pre"].append(stall_pre)
        # ---- T1 prior (after prefill)
        if prior.startswith(("T1:", "T01:")):
            sim.clear_background()
            p = seg_tables["t1"](r, prior.split(":")[-1]).reshape(-1)
            sim.prior[:] = p
            order = np.argsort(-p)[:C]
            sim.issue(order[p[order] > 0], 2, t)
        # ---- decode
        sim.evict_mode = dec_mode
        D = r["dec"].shape[1]
        t_dec0, stall_dec, first8 = t, 0.0, 0.0
        S = int(prior[3:]) if prior.startswith("seg") else 0
        for s in range(D):
            if S and s % S == 0 and s >= S:
                sim.clear_background()
                p = norm(counts(r["dec"][:, s - S:s], E)).reshape(-1)
                sim.prior[:] = p
                order = np.argsort(-p)[:C]
                sim.issue(order[p[order] > 0], 2, t)
            for l in range(L):
                t_gate = t + hw["nonexp_layer"] / hw["mem_bw"]                  # attention + gate done
                if pref == "la1" and l + 1 < L and r["la_dec"] is not None:
                    sim.issue((l + 1) * E + r["la_dec"][l, s], 1, t_gate)
                keys = l * E + r["dec"][l, s]
                t_ready = sim.need(keys, t_gate)
                st = t_ready - t_gate
                stall_dec += st
                if s < 8:
                    first8 += st
                t = t_ready + k * hw["eb"] / hw["mem_bw"]
                sim.ruse[keys] += 1
                sim.freq[keys] += 1
        if D:
            out["tpot"].append((t - t_dec0) / D)
        out["e2e"].append(t - t_arr)
        out["stall_dec"].append(stall_dec)
        out["first8"].append(first8)
    st = sim.stats
    return dict(policy=policy, C=C, W=W,
                ttft_ms=1e3 * float(np.mean(out["ttft"])), tpot_ms=1e3 * float(np.mean(out["tpot"])),
                e2e_ms=1e3 * float(np.mean(out["e2e"])),
                stall_dec_ms_per_req=1e3 * float(np.mean(out["stall_dec"])),
                stall_pre_ms_per_req=1e3 * float(np.mean(out["stall_pre"])),
                first8_stall_ms=1e3 * float(np.mean(out["first8"])),
                GB_loaded_per_req=st["loaded"] * hw["eb"] / 1e9 / len(reqs),
                wasted_frac=st["wasted"] / max(1, st["loaded"]), loads=dict(st))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--train", default=None, help="Wtrain trace dir, for GLOBAL / T1 decode-position tables")
    ap.add_argument("--preds", default=None, help="E1 export npz with T0 predictions (OURS_ADMIT, ...)")
    ap.add_argument("--hw", default="glm4bit")
    ap.add_argument("--E", type=int, default=128)
    ap.add_argument("--budget_gb", type=float, default=41.0, help="unified memory for the expert cache")
    ap.add_argument("--cache_frac", type=float, default=None, help="cache size as a fraction of all experts "
                    "(overrides --budget_gb; used for the A tier, whose models would fit entirely)")
    ap.add_argument("--W", type=float, default=0.6, help="idle window before prefill (s), e.g. RoG planning")
    ap.add_argument("--P", type=int, default=2, help="prefill prefetch window (layers)")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--prefill_tokens", default="all", choices=["all", "body", "question"],
                    help="E5: which prompt tokens still go through the MoE in prefill (the rest reuse cached KV)")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    hw = HW[args.hw]
    reqs = load_run(args.trace)[:args.n]
    L = reqs[0]["pre"].shape[0]
    C = min(L * args.E, int(args.budget_gb * 1e9 / hw["eb"]))
    if args.cache_frac is not None:
        C = int(args.cache_frac * L * args.E)
    preds = {}
    if args.preds:
        z = np.load(args.preds)
        qi = {q: i for i, q in enumerate(z["questions"])}
        for nm in z.files:
            if nm != "questions":
                preds[nm] = np.stack([z[nm][qi[r["q"]]].astype(np.float64) for r in reqs])
    preds["ORACLE"] = np.stack([norm(counts(r["dec"], args.E)) for r in reqs])
    tables = {}
    if args.train:
        from e1_phases import DecTable, t1_predictors
        from e1_analysis import attach_paths
        tr = load_run(args.train)
        T = DecTable(tr, args.E)
        preds["GLOBAL"] = np.stack([T.glob] * len(reqs))
        src = "webqsp" if "webqsp" in args.trace else "cwq"
        attach_paths(reqs, src)
        tables["t1"] = lambda r, nm: (t1_predictors(r, T, args.E)[0][nm] if r["dec"].shape[1] else T.glob)
    else:
        tables["t1"] = lambda r, nm: norm(counts(r["dec"], args.E)) if nm == "ORACLE" else None
    print(f"{args.trace}: {len(reqs)} requests, L={L}, cache C={C} experts ({C / (L * args.E):.0%}), "
          f"expert {hw['eb'] / 1e6:.1f} MB, load {hw['eb'] / hw['ssd'] * 1e3:.2f} ms, "
          f"decode layer {(hw['nonexp_layer'] + reqs[0]['dec'].shape[-1] * hw['eb']) / hw['mem_bw'] * 1e3:.2f} ms",
          flush=True)
    rows = []
    for pol in args.policy.split(","):
        r = run(reqs, pol, hw, C, args.E, args.W, args.P, tables, preds, args.lam, args.prefill_tokens)
        r["prefill_tokens"] = args.prefill_tokens
        rows.append(r)
        print(f"  {pol:28s} TTFT {r['ttft_ms']:7.0f} ms  TPOT {r['tpot_ms']:6.2f} ms  e2e {r['e2e_ms']:7.0f} ms  "
              f"decode stall {r['stall_dec_ms_per_req']:7.1f} ms/req  first-8 stall {r['first8_stall_ms']:6.1f} ms  "
              f"loaded {r['GB_loaded_per_req']:5.1f} GB/req  wasted {r['wasted_frac']:.0%}", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(rows, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
