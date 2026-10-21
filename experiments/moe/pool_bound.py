"""Affinity decode-pool plan A0/A1: how much does picking WHICH requests decode together (per step) shrink the
number of distinct experts per layer?

Pool of N decode streams (requests of a trace, each a sequence of per-token routes [L, k]). Every step B of them
advance one token; the cost of a step is sum over layers of the number of distinct experts the B tokens use.
Policies:
  fcfs     the B requests that have waited longest (round robin; every request moves at the same pace)
  oracle   greedy: start from requests forced by the wait limit (skipped ceil(N/B)-1+k steps in a row, i.e. k steps
           more than round robin would make them wait), then repeatedly add
           the request whose next-token route adds the fewest new (layer, expert) pairs -- using the TRUE routes
  prev     the same greedy, but scoring each request by the route of its PREVIOUS token (known before the step)
Cost is always computed on the true routes. Reported: distinct experts per layer per token (lower is better),
relative to fcfs, and the wait (steps skipped) distribution.

  python experiments/moe/pool_bound.py --trace outputs/moe/traces_v3/GLM-4.5-Air_cwq_M0_cot_chat_cap8 \
      --N 32 64 128 200 --B 4 8 16 32 --k 0 1 2 4 1000 --steps 400 --out outputs/moe/pool_bound.json
"""
import argparse
import glob
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_R = {}


def load(trace, E):
    reqs = []
    for f in sorted(glob.glob(trace + "/q*.npz")):
        z = np.load(f)
        dec = z["dec_idx"]                                  # [L, D, k]
        if dec.shape[1] < 2:
            continue
        first_prev = z["pre_idx"][:, -1:, :]                # route of the last prompt token
        reqs.append((dec.transpose(1, 0, 2), np.concatenate([first_prev, dec[:, :-1]], 1).transpose(1, 0, 2)))
    return reqs


def masks(routes, E):
    """routes [T, L, k] -> bool [T, L, E]."""
    T, L, k = routes.shape
    m = np.zeros((T, L, E), bool)
    np.put_along_axis(m, routes.astype(np.int64), True, -1)
    return m


def run(args):
    N, B, k, policy, steps, E = args
    reqs = _R["reqs"][:N]
    true = [masks(r[0], E) for r in reqs]
    prev = [masks(r[1], E) for r in reqs]
    # requests sit at random points of their decode (continuous batching: a pool mixes old and new requests);
    # starting everyone at token 0 would put all streams in the same templated opening text
    rng = np.random.default_rng(0)
    pos = np.array([rng.integers(0, max(1, len(t) // 2)) for t in true]) if _R["offset"] else np.zeros(N, int)
    wait = np.zeros(N, int)
    L = true[0].shape[1]
    cost, toks, waits = 0, 0, []
    for _ in range(steps):
        alive = np.flatnonzero(pos < np.array([len(t) for t in true]))
        if len(alive) == 0:
            break
        if len(alive) <= B:
            pick = list(alive)
        elif policy == "fcfs":
            pick = list(alive[np.lexsort((alive, -wait[alive]))][:B])      # longest-waiting first: round robin
        else:
            src = true if policy == "oracle" else prev
            lim = int(np.ceil(len(alive) / B)) - 1 + k        # fair share of waiting (round robin) + k extra steps
            forced = alive[wait[alive] >= lim]
            forced = forced[np.argsort(-wait[forced])][:B]
            pick = list(forced)
            U = np.zeros((L, E), bool)
            for r in pick:
                U |= src[r][pos[r]]
            cand = np.array([r for r in alive if r not in set(pick)])
            if len(cand):
                C = np.stack([src[r][pos[r]] for r in cand])            # [c, L, E]
                while len(pick) < B and len(cand):
                    inc = (C & ~U).sum((1, 2))
                    j = int(np.argmin(inc - 1e-6 * wait[cand]))          # tie-break: longest-waiting first
                    pick.append(int(cand[j]))
                    U |= C[j]
                    cand = np.delete(cand, j)
                    C = np.delete(C, j, 0)
        U = np.zeros((L, E), bool)
        for r in pick:
            U |= true[r][pos[r]]
        cost += int(U.sum())
        toks += len(pick)
        sel = np.zeros(N, bool)
        sel[pick] = True
        waits.extend(wait[alive[~sel[alive]]] + 1)
        wait[alive[~sel[alive]]] += 1
        wait[sel] = 0
        pos[sel] += 1
    w = np.array(waits) if waits else np.zeros(1)
    return dict(N=N, B=B, k=k, policy=policy, experts_per_layer_per_token=cost / max(toks, 1) / L,
                experts_per_layer_per_step=cost / max(toks / B, 1) / L, tokens=toks,
                wait_max=int(w.max()), wait_p99=float(np.percentile(w, 99)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--E", type=int, default=128)
    ap.add_argument("--N", type=int, nargs="+", default=[32, 64, 128, 200])
    ap.add_argument("--B", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--k", type=int, nargs="+", default=[0, 1, 2, 4, 1000])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--no_offset", action="store_true", help="start every request at its first decode token")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    _R["reqs"] = load(a.trace, a.E)
    _R["offset"] = not a.no_offset
    print(f"{len(_R['reqs'])} requests", flush=True)
    jobs = []
    for N in a.N:
        for B in a.B:
            if B >= N:
                continue
            jobs.append((N, B, 0, "fcfs", a.steps, a.E))
            for k in a.k:
                for pol in ("oracle", "prev"):
                    jobs.append((N, B, k, pol, a.steps, a.E))
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(run, jobs))
    json.dump(rows, open(a.out, "w"), indent=1)
    base = {(r["N"], r["B"]): r["experts_per_layer_per_token"] for r in rows if r["policy"] == "fcfs"}
    for r in rows:
        b = base[(r["N"], r["B"])]
        print(f"N={r['N']:3} B={r['B']:2} {r['policy']:6} k={r['k']:4}  experts/layer/token {r['experts_per_layer_per_token']:.3f}"
              f" ({r['experts_per_layer_per_token'] / b - 1:+.0%})  per step {r['experts_per_layer_per_step']:.1f}"
              f"  wait max {r['wait_max']} p99 {r['wait_p99']:.0f}", flush=True)


if __name__ == "__main__":
    main()
