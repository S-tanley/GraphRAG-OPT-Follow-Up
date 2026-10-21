"""Anchor plan P3: tsim with multi-layer-ahead decode prefetch (A3: does predicting h layers ahead cut stalls?).

Extends tsim.Sim (tsim.py itself is unchanged, so earlier results stay reproducible):
  * the lookahead queue is ordered by deadline (absolute layer index the expert is needed at), so a
    prediction for the next layer overtakes one for 6 layers ahead; issue-time order inside a deadline.
    For one-layer-ahead (la1) this is the same order as tsim's.
  * a newer prediction for a target layer replaces the older one: queued experts of that target that the
    new prediction drops are cancelled.
  * an expert prefetched for a layer that has not run yet is protected from eviction; a prefetch whose only
    possible victims are such protected experts (needed no later than it) is dropped.
Decode prefetch policy "ah<H>:<pred>[:m<M>][:t<tau>]": at layer l of each step, issue the top-M predicted
experts of layers l+1..l+H. pred is one of the anchor_dump.py predictors (GATE, TOKGATE, AV, AVG, ORACLE)
or MIX = GATE for h = 1, and for h >= 2 AVG when the anchor confidence >= tau, else GATE.
":p<thr>" keeps only the ranked experts whose predictor score is >= thr (AV: vote share; AVG: vote share +
beta * [in GATE set]; TOKGATE: TOKID frequency + beta * [in GATE set]) -- confidence-thresholded prefetch.
Baselines in the same run: lru/none, moeinf/la1 (= ah1:GATE), moeinf/ah<H>:GATE (HOBBIT-style multi-layer),
moeinf/ah<H>:ORACLE (perfect H-ahead knowledge: the upper bound).

  python experiments/moe_edge/tsim_ahead.py --trace outputs/moe/traces_v3/GLM-4.5-Air_cwq_M0_cap8 \
      --preds outputs/moe/anchor_preds/GLM-4.5-Air_cwq_M0_cap8 --hw glm4bit --cache_frac 0.5 \
      --policy moeinf/la1,moeinf/ah4:GATE:m8,moeinf/ah4:MIX:m8:t0.9 --out out.json
"""
import argparse
import glob
import heapq
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "moe"))
from tsim import HW, Sim  # noqa: E402
from e1_analysis import load_run  # noqa: E402

HW = dict(HW)
# GLM-4.5-Air bf16 on one H200, experts offloaded to pinned host memory over PCIe 5 (~50 GB/s effective H2D).
# Non-expert weights ~6.4B params over 46 layers; HBM ~3.5 TB/s effective for batch-1 decode.
HW["h200_glm_bf16"] = dict(eb=3 * 4096 * 1408 * 2, nonexp_layer=6.4e9 * 2 / 46, act_params_layer=0.6e9,
                           mem_bw=3.5e12, c_pre=2e-6 / 1e9, ssd=50e9)


class SimAhead(Sim):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        n = self.L * self.E
        self.want = np.full(n, -1.0)            # deadline (absolute layer index) of a pending prefetch
        self.now = 0.0                          # absolute index of the layer now running
        self.tq = {}                            # deadline -> keys queued for it at lookahead priority
        self.stats.update(la_dropped=0, la_cancelled=0)

    # heaps hold (order, seq, key, t_issue); order = deadline for lookahead, issue time otherwise
    def issue(self, keys, prio, t, deadline=None):
        keys = np.asarray(keys).ravel()
        keys = keys[~self.res[keys]]
        for key in keys:
            key = int(key)
            q = self.queued[key]
            if q != -1 and q <= prio:
                continue
            self.queued[key] = prio
            self.seq += 1
            heapq.heappush(self.qs[prio], (deadline if prio == 1 and deadline is not None else t, self.seq, key, t))
        if prio == 1 and deadline is not None:
            self.want[keys] = np.maximum(self.want[keys], deadline)

    def lookahead(self, keys, t, deadline):
        """Issue a prediction for one target layer, cancelling what an older prediction queued for it."""
        keys = np.asarray(keys).ravel()
        old = self.tq.get(deadline)
        if old is not None:
            drop = np.setdiff1d(old, keys)
            drop = drop[self.queued[drop] == 1]
            if len(drop):
                self.queued[drop] = -1
                self.want[drop] = -1
                self.stats["la_cancelled"] += len(drop)
        self.tq[deadline] = keys
        self.issue(keys, 1, t, deadline)

    def clear_background(self):
        for item in self.qs[2]:
            if self.queued[item[2]] == 2:
                self.queued[item[2]] = -1
        self.qs[2] = []

    def _pop(self):
        while True:
            best = None
            for p in range(3):
                h = self.qs[p]
                while h and (self.queued[h[0][2]] != p or self.res[h[0][2]]):
                    heapq.heappop(h)
                if h and h[0][3] <= self.t_loader:
                    best = p
                    break
            if best is None:
                heads = [(h[0][3], p) for p, h in enumerate(self.qs) if h]
                if not heads:
                    return None
                return ("wait", min(heads)[0])
            _o, _s, key, _t = heapq.heappop(self.qs[best])
            return ("load", best, key)

    def score(self, keys):
        base = super().score(keys)
        return base + 1e12 * (self.want[keys] >= self.now)

    def _load(self, prio, key):
        if prio == 1 and not self.res[key] and self.n_res >= self.C:
            cand = np.flatnonzero(self.res & ~self.pin)
            if len(cand):
                sc = self.score(cand)
                v = int(cand[int(np.argmin(sc))])
                if self.want[v] >= self.now and self.want[v] <= self.want[key]:
                    self.queued[key] = -1
                    self.want[key] = -1
                    self.stats["la_dropped"] += 1
                    return
        super()._load(prio, key)

    def need(self, keys, t):
        done = super().need(keys, t)
        self.want[np.asarray(keys).ravel()] = -1      # used: no longer a pending prefetch
        return done


def parse_pref(pref):
    if pref in ("none", "la1"):
        return dict(H=1 if pref == "la1" else 0, pred="GATE", M=None, tau=None, thr=None)
    parts = pref.split(":")
    d = dict(H=int(parts[0][2:]), pred=parts[1], M=None, tau=None, thr=None)
    for x in parts[2:]:
        if x[0] == "m":
            d["M"] = int(x[1:])
        elif x[0] == "t":
            d["tau"] = float(x[1:])
        elif x[0] == "p":
            d["thr"] = float(x[1:])
    return d


def run(reqs, preds, policy, hw, C, E, W, P_pre, kref=None):
    evict, pref = policy.split("/")
    pp = parse_pref(pref)
    L, k = reqs[0]["pre"].shape[0], reqs[0]["pre"].shape[-1]
    M = pp["M"] or k
    sim = SimAhead(L, E, k, C, hw, 1.0, 1.0)
    sim.evict_mode = evict
    t, pos = 0.0, 0.0
    out = dict(stall_dec=[], tpot=[], ttft=[], toks=0, dec_loads=0, dec_wasted=0)
    for ri, r in enumerate(reqs):
        sim.ruse[:] = 0
        sim.clear_background()
        t_arr = t
        t += W
        sim.advance(t)
        pre_idx = r["pre"]
        T = pre_idx.shape[1]
        for l in range(L):                              # prefill as in tsim (P-layer window)
            for ll in range(l + 1, min(L, l + 1 + P_pre)):
                sim.issue(ll * E + np.unique(pre_idx[ll]), 1, t, deadline=pos + ll)
            sim.now = pos + l
            keys = l * E + np.unique(pre_idx[l])
            t = sim.need(keys, t) + T * hw["act_params_layer"] * hw["c_pre"]
            sim.ruse[keys] += 1
            sim.freq[keys] += 1
        pos += L
        sim.tq.clear()
        out["ttft"].append(t - t_arr)
        D = r["dec"].shape[1]
        P = preds[ri] if preds is not None else None
        t_dec0, stall = t, 0.0
        l0, w0 = sim.stats["loaded"], sim.stats["wasted"]
        for s in range(D):
            for l in range(L):
                sim.now = pos + l
                t_gate = t + hw["nonexp_layer"] / hw["mem_bw"]
                for h in range(1, pp["H"] + 1):
                    if l + h >= L:
                        break
                    if pp["pred"] == "GATE" and pp["H"] == 1 and P is None:
                        ids = r["la_dec"][l, s][:M]
                    else:
                        name = pp["pred"]
                        if name == "MIX":
                            name = "AVG" if h >= 2 and float(P["conf"][l, s]) >= pp["tau"] else "GATE"
                        ids = P[name][l, s, h - 1, :M]
                        if pp["thr"] is not None and name + "_s" in P:   # only the experts it is sure about
                            ids = ids[P[name + "_s"][l, s, h - 1, :M] >= pp["thr"]]
                    sim.lookahead((l + h) * E + ids.astype(np.int64), t_gate, pos + l + h)
                keys = l * E + r["dec"][l, s]
                t_ready = sim.need(keys, t_gate)
                stall += t_ready - t_gate
                t = t_ready + k * hw["eb"] / hw["mem_bw"]
                sim.ruse[keys] += 1
                sim.freq[keys] += 1
            for l in range(L):
                sim.tq.pop(pos + l, None)
            pos += L
        out["dec_loads"] += sim.stats["loaded"] - l0
        out["dec_wasted"] += sim.stats["wasted"] - w0
        if D:
            out["tpot"].append((t - t_dec0) / D)
            out["toks"] += D
        out["stall_dec"].append(stall)
    st = sim.stats
    toks = max(out["toks"], 1)
    return dict(policy=policy, C=C, ttft_ms=1e3 * float(np.mean(out["ttft"])),
                tpot_ms=1e3 * float(np.mean(out["tpot"])),
                stall_ms_per_tok=1e3 * float(np.sum(out["stall_dec"])) / toks,
                loads_per_tok=out["dec_loads"] / toks, wasted_frac=out["dec_wasted"] / max(1, out["dec_loads"]),
                loads=dict(st))


def _job(args):
    trace, pdir, n, pol, hw, frac, E, W, P = args
    reqs = load_run(trace)[:n]
    preds = None
    if pdir:
        files = sorted(glob.glob(pdir + "/q*.npz"))[:len(reqs)]
        preds = [dict(np.load(f)) for f in files]
        for r, p in zip(reqs, preds):
            assert p["GATE"].shape[1] == r["dec"].shape[1], "preds / trace mismatch"
    L = reqs[0]["pre"].shape[0]
    C = int(frac * L * E)
    res = run(reqs, preds, pol, HW[hw], C, E, W, P)
    res.update(hw=hw, cache_frac=frac, n=len(reqs))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--preds", default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--hw", nargs="+", default=["glm4bit"])
    ap.add_argument("--cache_frac", type=float, nargs="+", default=[0.5])
    ap.add_argument("--E", type=int, default=128)
    ap.add_argument("--W", type=float, default=0.0)
    ap.add_argument("--P", type=int, default=2)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jobs = [(a.trace, a.preds, a.n, pol, hw, f, a.E, a.W, a.P)
            for hw in a.hw for f in a.cache_frac for pol in a.policy.split(",")]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(_job, jobs))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=1)
    for r in rows:
        print(f"{r['hw']:14} cache {r['cache_frac']:.0%}  {r['policy']:28} stall {r['stall_ms_per_tok']:7.2f} ms/tok  "
              f"TPOT {r['tpot_ms']:7.2f} ms  decode loads/tok {r['loads_per_tok']:5.1f}  wasted {r['wasted_frac']:.0%}  "
              f"dropped {r['loads']['la_dropped']}  cancelled {r['loads']['la_cancelled']}", flush=True)


if __name__ == "__main__":
    main()
