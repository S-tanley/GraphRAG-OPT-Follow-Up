"""Expert-queue plan Q0: can tokens of a multi-request decode pool be accumulated per (layer, expert) before a
load, and how long does a token then wait for its slowest expert?

Pool of N decode streams (random start offsets, as in pool_bound.py); every step all N advance one token
(synchronous decode, the baseline's batching). Experts outside a static resident set (the C most used experts of
each layer over the whole trace, an LFU-like cache) must be loaded to be used. Per layer, independently:
  sync      every needed non-resident expert is loaded in the step it is needed; the load serves the tokens of
            that step only
  queue n w tokens wait in a per-(layer, expert) queue; the expert is loaded when >= n tokens wait or the oldest
            has waited w steps; one load serves the whole queue
A token's wait at a layer is the largest wait among its k experts (0 if all are resident) -- the "slowest expert".
Steps only, no time (the time model is Q1's simulator). Coupling between layers / steps is ignored: a waiting
token is assumed not to delay its own later layers.

  python experiments/moe/queue_stats.py --trace outputs/moe/traces_v3/GLM-4.5-Air_cwq_M0_cot_chat_cap8 \
      --N 16 32 64 128 200 --cache 0.25 0.5 0.75 --out outputs/moe/queue_stats.json
"""
import argparse
import glob
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_R = {}


def load(trace):
    reqs = []
    for f in sorted(glob.glob(trace + "/q*.npz")):
        d = np.load(f)["dec_idx"]                           # [L, D, k]
        if d.shape[1] >= 64:
            reqs.append(d.transpose(1, 0, 2).astype(np.int16))
    return reqs


def run(args):
    N, frac, n, w, steps = args
    reqs = _R["reqs"][:N]
    L, k = reqs[0].shape[1], reqs[0].shape[2]
    E = _R["E"]
    C = int(frac * E)
    freq = _R["freq"]                                       # [L, E] uses over the whole trace
    resident = np.zeros((L, E), bool)
    np.put_along_axis(resident, np.argsort(-freq, 1)[:, :C], True, 1)
    rng = np.random.default_rng(0)
    pos = np.array([rng.integers(0, max(1, len(r) // 2)) for r in reqs])
    T = min(steps, min(len(r) - p for r, p in zip(reqs, pos)))
    loads, served, tok = 0, 0, 0
    waits = []                                              # per (token, layer) slowest-expert wait, in steps
    hit_tok_layers = 0
    for l in range(L):
        q = {}                                              # expert -> list of (arrival step, token id)
        tw = {}                                             # (step, req) -> max wait at this layer
        for s in range(T + (w if n else 0) + 1):
            if s < T:
                for i, r in enumerate(reqs):
                    ex = r[pos[i] + s, l]
                    miss = ex[~resident[l, ex]]
                    tok += 1 if l == 0 else 0
                    if len(miss) == 0:
                        hit_tok_layers += 1
                        tw[(s, i)] = 0
                        continue
                    tw[(s, i)] = 0
                    for e in miss:
                        q.setdefault(int(e), []).append((s, i))
            fire = []
            for e, lst in q.items():
                if not lst:
                    continue
                if n == 0:                                   # sync: load whatever is needed now
                    fire.append(e)
                elif len(lst) >= n or s - lst[0][0] >= w or s >= T:
                    fire.append(e)
            for e in fire:
                lst = q[e]
                loads += 1
                served += len(lst)
                for (s0, i) in lst:
                    tw[(s0, i)] = max(tw[(s0, i)], s - s0)
                q[e] = []
        waits.extend(tw.values())
    wv = np.array(waits)
    return dict(N=N, cache=frac, n=n, w=w, steps=T, tokens_per_load=served / max(loads, 1),
                loads_per_token_layer=loads / max(len(wv), 1), frac_layer_all_resident=hit_tok_layers / max(len(wv), 1),
                wait_mean=float(wv.mean()), wait_p50=float(np.percentile(wv, 50)), wait_p99=float(np.percentile(wv, 99)),
                frac_waiting=float((wv > 0).mean()),
                # a token crosses L layers; if waits added up along the layers (no overlap), its per-step delay:
                token_delay_steps_mean=float(wv.mean() * L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--E", type=int, default=128)
    ap.add_argument("--N", type=int, nargs="+", default=[16, 32, 64, 128, 200])
    ap.add_argument("--cache", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    reqs = load(a.trace)
    _R["reqs"], _R["E"] = reqs, a.E
    L = reqs[0].shape[1]
    freq = np.zeros((L, a.E))
    for r in reqs:
        for l in range(L):
            freq[l] += np.bincount(r[:, l].ravel(), minlength=a.E)
    _R["freq"] = freq
    # arrival-rate picture: share of uses taken by the hottest experts per layer
    sh = np.sort(freq, 1)[:, ::-1].cumsum(1) / freq.sum(1, keepdims=True)
    print(f"{len(reqs)} requests, L={L}; share of uses by the hottest 25/50/75% experts per layer: "
          f"{sh[:, a.E // 4 - 1].mean():.2f} / {sh[:, a.E // 2 - 1].mean():.2f} / {sh[:, 3 * a.E // 4 - 1].mean():.2f}", flush=True)
    jobs = [(N, c, 0, 0, a.steps) for N in a.N for c in a.cache]
    jobs += [(N, c, n, w, a.steps) for N in a.N for c in a.cache for n in (2, 4, 8, 16) for w in (1, 2, 4)]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(run, jobs))
    json.dump(rows, open(a.out, "w"), indent=1)
    base = {(r["N"], r["cache"]): r for r in rows if r["n"] == 0}
    for r in rows:
        b = base[(r["N"], r["cache"])]
        tag = "sync" if r["n"] == 0 else f"n={r['n']:2} w={r['w']}"
        print(f"N={r['N']:3} cache {r['cache']:.0%} {tag:10} tokens/load {r['tokens_per_load']:5.2f} "
              f"(x{r['tokens_per_load'] / b['tokens_per_load']:.2f})  loads/(token*layer) {r['loads_per_token_layer']:.3f} "
              f"(x{r['loads_per_token_layer'] / max(b['loads_per_token_layer'], 1e-9):.2f})  waiting {r['frac_waiting']:.0%}  "
              f"wait mean {r['wait_mean']:.2f} p99 {r['wait_p99']:.0f} steps/layer", flush=True)


if __name__ == "__main__":
    main()
