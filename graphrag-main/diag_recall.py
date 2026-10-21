"""
Phase-1a diagnostic: does retrieval even surface the answer?

For a dataset's precomputed RoG-plan paths, compute ANSWER RECALL = fraction of
questions where some normalized gold answer appears as a node (entity) on a
retrieved path. This is the ceiling for retrieval-grounded Hit. Compare against
the gold a_entity too. NO model is loaded — pure CPU, fast.

  answer-recall >> current Hit  => generation-side gap (prompt/decoding)
  answer-recall ~ current Hit   => retrieval-side gap (need better paths)
"""
import argparse
import json
import numpy as np
from metrics import normalize_answer
from table2_worker import load_ds


def path_nodes(path):
    """All entity strings on a path [[h,r,t],...]."""
    nodes = []
    for trip in path:
        h, r, t = trip
        nodes.append(h)
        nodes.append(t)
    return nodes


def recall_hit(golds, paths):
    """1 if any normalized gold is a substring of any normalized path node."""
    gnorm = [normalize_answer(g) for g in golds]
    gnorm = [g for g in gnorm if g]
    if not gnorm:
        return None
    for p in paths:
        for node in path_nodes(p):
            nn = normalize_answer(node)
            for g in gnorm:
                if g and g in nn:          # gold is substring of a path node (Hit direction)
                    return 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rmanluo/RoG-cwq")
    ap.add_argument("--paths_file", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--max_paths", type=int, default=15)
    args = ap.parse_args()

    ds = load_ds(args.dataset)
    n = min(args.n, len(ds)) if args.n > 0 else len(ds)
    pdict = json.load(open(args.paths_file))

    rec, npaths, nuniq_ent, plen_max, empty = [], [], [], [], 0
    for i in range(n):
        ex = ds[i]
        golds = ex["answer"] if isinstance(ex["answer"], list) else [ex["answer"]]
        ser = pdict.get(ex["question"], [])[:args.max_paths]
        paths = [[list(t) for t in p] for p in ser]
        r = recall_hit(golds, paths)
        if r is not None:
            rec.append(r)
        npaths.append(len(paths))
        ents = set()
        mx = 0
        for p in paths:
            ents.update(path_nodes(p))
            mx = max(mx, len(p))
        nuniq_ent.append(len(ents))
        plen_max.append(mx)
        if not paths:
            empty += 1

    print(f"=== {args.dataset}  N={n}  max_paths={args.max_paths} ===")
    print(f"answer-recall (gold entity on some retrieved path): {np.mean(rec):.4f}  (n={len(rec)})")
    print(f"paths/q: mean {np.mean(npaths):.2f}  median {int(np.median(npaths))}  "
          f"frac_empty {empty/n:.3f}")
    print(f"unique entities/q: mean {np.mean(nuniq_ent):.1f}")
    print(f"max path hop-len/q: mean {np.mean(plen_max):.2f}  median {int(np.median(plen_max))}")


if __name__ == "__main__":
    main()
