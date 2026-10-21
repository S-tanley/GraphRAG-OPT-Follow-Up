"""Summarize Exp2/3/4 prediction files: Hit, F1, paired McNemar vs a reference condition,
path-region token counts, and buckets (number of paths, multi-source convergence).

  python experiments/exp2/analyze.py outputs/exp2 [--ref M0] [--md out.md]
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from math import comb

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "graphrag-main"))
from metrics import evaluate_metrics  # noqa: E402

PATHS = {"rmanluo/RoG-cwq": "paths_rmanluo-RoG-cwq.json",
         "rmanluo/RoG-webqsp": "paths_rmanluo-RoG-webqsp.json"}


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def per_q_hit(preds, gold):
    return [evaluate_metrics([p], [g])["hits_at_1"] for p, g in zip(preds, gold)]


def features(dataset, questions, cap):
    pdict = json.load(open(os.path.join(ROOT, "graphrag-main/precomputed_paths_v2", PATHS[dataset])))
    feats = []
    for q in questions:
        P = pdict.get(q, [])[:cap]
        term = defaultdict(set)
        for p in P:
            if p:
                term[p[-1][2]].add(p[0][0])
        feats.append(dict(npaths=len(P), multisrc=any(len(s) >= 2 for s in term.values())))
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--ref", default="M0")
    ap.add_argument("--md", default=None)
    args = ap.parse_args()

    groups = defaultdict(dict)                     # (model, dataset, cap) -> cond -> data
    for d in args.dirs:
        for f in glob.glob(os.path.join(d, "*.json")):
            x = json.load(open(f))
            if "cond" not in x:
                continue
            groups[(x["model"], x["dataset"], x["max_paths"])][x["cond"]] = x

    lines = []
    for key in sorted(groups):
        model, dataset, cap = key
        conds = groups[key]
        if args.ref not in conds:
            continue
        ref = conds[args.ref]
        rh = per_q_hit(ref["predictions"], ref["answers"])
        feats = features(dataset, ref["questions"], cap)
        lines.append(f"\n### {model} × {dataset.split('/')[-1]}（cap={cap}，n={len(rh)}）\n")
        lines.append("| 条件 | Hit | F1 | ΔHit vs " + args.ref + " | 变好/变差 | p | 路径区 token/q | "
                     "Hit（≤4 条路径） | Hit（≥5 条） | Hit（多起点汇合） |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        order = sorted(conds, key=lambda c: (c != args.ref, c))
        for c in order:
            x = conds[c]
            h = per_q_hit(x["predictions"], x["answers"])
            m = evaluate_metrics(x["predictions"], x["answers"])
            up = sum(1 for a, b in zip(rh, h) if b > a)
            dn = sum(1 for a, b in zip(rh, h) if b < a)
            ntok = sum(x["path_tokens"]) / len(x["path_tokens"])

            def bucket(pred):
                idx = [i for i, f in enumerate(feats) if pred(f)]
                return f"{sum(h[i] for i in idx) / len(idx):.3f} (n={len(idx)})" if idx else "-"
            lines.append(f"| {c} | {m['hits_at_1']:.3f} | {m['f1']:.3f} | "
                         f"{m['hits_at_1'] - sum(rh) / len(rh):+.3f} | {up}/{dn} | {mcnemar(up, dn):.3f} | "
                         f"{ntok:.0f} | {bucket(lambda f: f['npaths'] <= 4)} | "
                         f"{bucket(lambda f: f['npaths'] >= 5)} | {bucket(lambda f: f['multisrc'])} |")
    text = "\n".join(lines)
    print(text)
    if args.md:
        open(args.md, "w").write(text + "\n")


if __name__ == "__main__":
    main()
