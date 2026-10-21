"""Phase A / E2 offline bound: segment-wise regrouping of decode batches.

A long decode is treated as a sequence of segments of S steps. N requests are active (continuous batching:
a finished request is replaced by the next one in the queue); each step only B of them can run together.
At every segment boundary the active requests are partitioned into N/B groups and every group decodes S
steps. Cost metric (no cache): per step and layer, the number of distinct experts the group needs
(= expert loads if nothing is resident), divided by the tokens produced -> expert loads / layer / token.

Grouping keys (how the next segment's expert profile of a request is predicted):
  random      random partition at every boundary
  static      random groups formed when a request is admitted and kept (classic static batching)
  temporal    profile of the request's previous segment (known online)
  recent8     profile of its last 8 decode steps
  oracle      actual profile of its next segment (upper bound)

  python experiments/moe_edge/segment_regroup.py --trace outputs/moe/traces/GLM-4.5-Air_cwq_M0_cot_cap8 \
      --N 32 --B 8 --S 8,16,32,64 --out outputs/moe_edge/segment_regroup_glm_w2.json
"""

import argparse
import glob
import json
import os

import numpy as np


def load(tdir):
    out = []
    for f in sorted(glob.glob(os.path.join(tdir, "q*.npz"))):
        d = np.load(f)["dec_idx"].astype(np.int32)          # [L, D, k]
        if d.shape[1] > 0:
            out.append(d)
    return out


def profile(dec, a, b, E):
    """Unit-norm per-layer expert histogram of decode steps [a, b)."""
    a, b = max(a, 0), min(b, dec.shape[1])
    L = dec.shape[0]
    c = np.zeros((L, E))
    if b > a:
        for l in range(L):
            np.add.at(c[l], dec[l, a:b].ravel(), 1)
    v = c.ravel()
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def partition(ids, B, vecs, rng):
    """Greedy: random seed, add its B-1 most similar remaining requests. vecs[i] None -> random filler."""
    rem = list(ids)
    rng.shuffle(rem)
    groups = []
    while rem:
        seed = rem.pop(0)
        if not rem or vecs is None or vecs.get(seed) is None:
            groups.append([seed] + rem[:B - 1])
            rem = rem[B - 1:]
            continue
        cand = [j for j in rem if vecs.get(j) is not None]
        sims = sorted(cand, key=lambda j: -float(vecs[j] @ vecs[seed]))
        pick = sims[:B - 1]
        if len(pick) < B - 1:                                 # fill with requests without a profile
            pick += [j for j in rem if j not in pick][:B - 1 - len(pick)]
        rem = [j for j in rem if j not in pick]
        groups.append([seed] + pick)
    return groups


def simulate(reqs, N, B, S, key, E, seed=0):
    rng = np.random.default_rng(seed)
    L = reqs[0].shape[0]
    queue = list(range(len(reqs)))
    rng.shuffle(queue)
    pos, active, static_groups = {}, [], []
    loads = toks = 0
    while queue or active:
        while len(active) < N and queue:                     # continuous batching: refill
            r = queue.pop(0)
            active.append(r)
            pos[r] = 0
        if key == "static":
            # keep groups; new arrivals fill the free slots of existing groups, else form new groups
            static_groups = [[r for r in g if r in active] for g in static_groups]
            placed = {r for g in static_groups for r in g}
            new = [r for r in active if r not in placed]
            for g in static_groups:
                while len(g) < B and new:
                    g.append(new.pop(0))
            while new:
                static_groups.append(new[:B])
                new = new[B:]
            static_groups = [g for g in static_groups if g]
            groups = static_groups
        else:
            vecs = None
            if key == "oracle":
                vecs = {r: profile(reqs[r], pos[r], pos[r] + S, E) for r in active}
            elif key == "temporal":
                vecs = {r: profile(reqs[r], pos[r] - S, pos[r], E) for r in active}
            elif key == "recent8":
                vecs = {r: profile(reqs[r], pos[r] - 8, pos[r], E) for r in active}
            groups = partition(active, B, vecs, rng)
        for g in groups:
            for s in range(S):
                act = [r for r in g if pos[r] + s < reqs[r].shape[1]]
                if not act:
                    break
                toks += len(act)
                for l in range(L):
                    loads += len(np.unique(np.concatenate([reqs[r][l, pos[r] + s] for r in act])))
        for r in active:
            pos[r] += S
        active = [r for r in active if pos[r] < reqs[r].shape[1]]
    return loads / toks / L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--E", type=int, default=128)
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--S", default="8,16,32,64")
    ap.add_argument("--keys", default="random,static,temporal,recent8,oracle")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    reqs = load(args.trace)
    print(f"{args.trace}: {len(reqs)} requests, mean decode {np.mean([r.shape[1] for r in reqs]):.0f} steps, "
          f"N={args.N} B={args.B}", flush=True)
    rows = []
    for S in [int(x) for x in args.S.split(",")]:
        res = {k: float(np.mean([simulate(reqs, args.N, args.B, S, k, args.E, sd) for sd in range(args.seeds)]))
               for k in args.keys.split(",")}
        base, orc = res["random"], res.get("oracle")
        line = f"S={S:3d}: random {base:.3f}"
        for k, v in res.items():
            if k == "random":
                continue
            frac = f", {(base - v) / (base - orc):.0%} of oracle gain" if orc is not None and k != "oracle" and base != orc else ""
            line += f" | {k} {v:.3f} ({v / base - 1:+.1%}{frac})"
        print(line, flush=True)
        rows.append(dict(S=S, N=args.N, B=args.B, **res))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(rows, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
