"""
Assemble the corrected, protocol-consistent Table 2 (MICRO 2026 rebuttal).

Inputs:
  --tsv      one or more metric TSVs (model\\tdataset\\tmethod\\tHits@1\\tF1\\tsubstr\\tn\\tsecs).
             Later files override earlier on (model,dataset,method) conflicts, so
             pass the full-set banked file first and the N=500 subset file second
             (subset wins for cells we re-ran, e.g. MQA reuse).
  --preds    one or more dirs of per-shard prediction JSONs ({answers,predictions})
             named <model>_<DS>_<method>_s<k>.json. Used to recompute per-question
             hits and the PAIRED GraphRAG-Opt vs GraphRAG-R statistics (mid-p
             McNemar, paired-difference CI, non-inferiority at margin delta).

Outputs: a console accuracy table with Wilson 95% CIs, a paired reuse-fidelity
table, and a LaTeX body for the paper (subset cells flagged with \\dag).

    python assemble_table2.py --metric hits \\
        --tsv results/table2_overnight.tsv results/table2_subset.tsv \\
        --preds results/preds_full results/preds_subset
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

from metrics import hit_single
from table2_stats import (fmt_pct_ci, mcnemar_midp, noninferiority,
                          paired_counts, paired_diff_ci, wilson_halfwidth)

COLS = ["Direct", "GraphRAG-R", "GraphRAG-Opt", "GraphRAG-Opt+RC"]
# TSV method label -> prediction-file method token
FILE_METHOD = {"Direct": "direct", "GraphRAG-R": "graphrag_r",
               "GraphRAG-Opt": "graphrag_opt", "GraphRAG-Opt+RC": "graphrag_opt_rc"}
MODEL_ORDER = ["rmanluo/RoG", "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf",
               "google/gemma-2-9b", "google/gemma-2-27b",
               "Qwen/Qwen3-8B", "Qwen/Qwen3-32B", "Qwen/Qwen3-Next-80B-A3B-Thinking",
               "facebook/opt-6.7b", "facebook/opt-13b", "facebook/opt-30b"]
DS_ORDER = ["CWQ", "WEB", "MQA"]
FULL_N = {"CWQ": 3531, "WEB": 1628, "MQA": 14274}
SHORT = {"rmanluo/RoG": "RoG", "meta-llama/Llama-2-7b-hf": "Llama-2-7b",
         "meta-llama/Llama-2-13b-hf": "Llama-2-13b", "google/gemma-2-9b": "Gemma-2-9b",
         "google/gemma-2-27b": "Gemma-2-27b", "Qwen/Qwen3-8B": "Qwen3-8b",
         "Qwen/Qwen3-32B": "Qwen3-32B", "Qwen/Qwen3-Next-80B-A3B-Thinking": "Qwen3-80B",
         "facebook/opt-6.7b": "OPT-6.7b", "facebook/opt-13b": "OPT-13b",
         "facebook/opt-30b": "OPT-30b"}
METRIC_IDX = {"hits": 0, "f1": 1, "substr": 2}
# model dir-token (model.replace('/','-')) -> canonical model id
TOK2MODEL = {m.replace("/", "-"): m for m in MODEL_ORDER}


def load_tsv(paths, data):
    for path in paths:
        try:
            rows = open(path).read().splitlines()[1:]
        except FileNotFoundError:
            continue
        for r in rows:
            p = r.split("\t")
            if len(p) < 7 or p[3] == "FAILED":
                continue
            data[(p[0], p[1], p[2])] = (float(p[3]), float(p[4]), float(p[5]), int(p[6]))


def load_preds(dirs):
    """-> hits[(model, DS, file_method)] = list of 0/1 per question (shard-ordered)."""
    hits = {}
    for d in dirs:
        if not os.path.isdir(d):
            continue
        # group shard files by (model, DS, method)
        groups = defaultdict(list)
        for fp in glob.glob(os.path.join(d, "*_s*.json")):
            base = os.path.basename(fp)
            m = re.match(r"^(.*)_(CWQ|WEB|MQA)_(direct|graphrag_r|graphrag_opt|graphrag_opt_rc)_s(\d+)\.json$", base)
            if not m:
                continue
            tok, ds, meth, shard = m.group(1), m.group(2), m.group(3), int(m.group(4))
            model = TOK2MODEL.get(tok)
            if model is None:
                continue
            groups[(model, ds, meth)].append((shard, fp))
        for key, files in groups.items():
            if key in hits:   # earlier dir already provided it; keep (full preferred when listed first)
                continue
            files.sort()
            vec = []
            for _, fp in files:
                obj = json.load(open(fp))
                for pred, gold in zip(obj["predictions"], obj["answers"]):
                    golds = [gold] if isinstance(gold, str) else list(gold)
                    vec.append(hit_single(pred, golds))
            hits[key] = vec
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", nargs="+", default=["results/table2_overnight.tsv",
                                                 "results/table2_subset.tsv"])
    ap.add_argument("--preds", nargs="+", default=["results/preds_full",
                                                   "results/preds_subset"])
    ap.add_argument("--metric", default="hits", choices=["hits", "f1", "substr"])
    ap.add_argument("--delta", type=float, default=0.025,
                    help="non-inferiority margin for Opt vs R (default 0.025 = 2.5 pts)")
    ap.add_argument("--latex", action="store_true", help="also print LaTeX body")
    args = ap.parse_args()

    data = {}
    load_tsv(args.tsv, data)
    mi = METRIC_IDX[args.metric]
    # preds: list dirs so the FIRST listed wins per key. Put full-set preds first so
    # primary CWQ/WEB use full pairing; subset provides everything else.
    hits = load_preds(args.preds)

    models = [m for m in MODEL_ORDER if any((m, d, c) in data
              for d in DS_ORDER for c in COLS)]

    # ---- Accuracy table with Wilson 95% CI (half-width in points) ----
    print(f"\n=== Table 2 ({args.metric}) with Wilson 95% CI (pts); † = N<full subset ===")
    hdr = f"{'Model':<13}{'DS':<5}" + "".join(f"{c:>20}" for c in COLS) + f"{'n':>7}"
    print(hdr); print("-" * len(hdr))
    for m in models:
        for d in DS_ORDER:
            n = None
            for c in COLS:
                v = data.get((m, d, c))
                if v:
                    n = v[3]
            if n is None:
                continue
            sub = "†" if n < FULL_N[d] else " "
            cells = []
            for c in COLS:
                v = data.get((m, d, c))
                if not v:
                    cells.append("--")
                    continue
                k = round(v[mi] * v[3])
                hw = 100 * wilson_halfwidth(k, v[3])
                cells.append(f"{100*v[mi]:.2f}±{hw:.1f}")
            print(f"{SHORT.get(m,m):<13}{d:<5}" + "".join(f"{x:>20}" for x in cells) + f"{str(n)+sub:>7}")
        print()

    # ---- Paired reuse-fidelity: GraphRAG-Opt vs GraphRAG-R ----
    print(f"\n=== Reuse fidelity: GraphRAG-Opt vs GraphRAG-R (paired, {args.metric}=hits) ===")
    print("delta (non-inferiority margin) = %.1f pts\n" % (100 * args.delta))
    ph = (f"{'Model':<13}{'DS':<5}{'N':>6}{'b':>4}{'c':>4}{'n_d':>5}"
          f"{'Opt-R(pts)':>12}{'95%CI(pts)':>16}{'McNemar p':>11}{'NI':>5}")
    print(ph); print("-" * len(ph))
    paired_rows = []
    for m in models:
        for d in DS_ORDER:
            ro = hits.get((m, d, "graphrag_r")); op = hits.get((m, d, "graphrag_opt"))
            if not ro or not op or len(ro) != len(op):
                continue
            b, c, nd, N = paired_counts(op, ro)
            D, lo, hi = paired_diff_ci(b, c, N)
            pmid = mcnemar_midp(b, c)
            ni_ok, ni_lo = noninferiority(b, c, N, delta=args.delta)
            print(f"{SHORT.get(m,m):<13}{d:<5}{N:>6}{b:>4}{c:>4}{nd:>5}"
                  f"{100*D:>+11.2f} {100*lo:>+6.2f},{100*hi:>+6.2f}  {pmid:>10.4f}"
                  f"{'  ok' if ni_ok else ' NO':>5}")
            paired_rows.append((m, d, N, b, c, nd, D, lo, hi, pmid, ni_ok, ni_lo))
    npass = sum(1 for r in paired_rows if r[10])
    print(f"\nNon-inferiority PASSED {npass}/{len(paired_rows)} cells at delta={100*args.delta:.1f} pts.")

    # ---- LaTeX body ----
    if args.latex:
        print("\n=== LaTeX body (metric=%s; cells: pt est, \\dag=subset) ===" % args.metric)
        for m in models:
            vals = []
            for d in DS_ORDER:
                for c in COLS:
                    v = data.get((m, d, c))
                    if not v:
                        vals.append("--"); continue
                    sub = "$^\\dagger$" if v[3] < FULL_N[d] else ""
                    vals.append(f"{100*v[mi]:.2f}{sub}")
            print(f"{SHORT.get(m,m)} & " + " & ".join(vals) + r" \\")


if __name__ == "__main__":
    main()
