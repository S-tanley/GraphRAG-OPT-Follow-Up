#!/usr/bin/env python3
"""
exp_retrieval_latency.py — Reviewer-B Q1b (retrieval latency) + Q3 retrieval floor.

Times the three enumerators on the SAME q_entity->a_entity retrieval over each
query's provided KG subgraph, capped at --cap paths (the deployed operating point),
across hop bounds 1..4:

  BFS         : get_paths_bfs            (RoG's enumerator = the paper's "GraphRAG").
  JbiDFS      : get_paths_optimized      (ours, GraphRAG-R, Algorithm 1).
  exhaustive  : plain bounded DFS        (an optimal-DFS reference point).

Reports mean latency, mean #paths (output size), and speedups. The JbiDFS-vs-BFS
speedup is the paper's retrieval-acceleration claim; it is BATCH-INDEPENDENT
(per-(query,pair) algorithmic), i.e. the Q3 "retrieval floor". Output: CSV.

Usage:
  python exp_retrieval_latency.py --datasets rmanluo/RoG-cwq rmanluo/RoG-webqsp metaqa-3hop \
     --n 200 --cap 50 --out results/retrieval_latency.csv
"""
import argparse, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from graph_utils import build_graph, get_paths_bfs, get_paths_optimized, OptimizedPathEnumerator  # noqa


def load_ds(name, n, seed=42):
    if name == "metaqa-3hop":
        from metaqa_adapter import load_metaqa
        ds = load_metaqa(hops=3, split="test", data_dir=str(HERE / "metaQA")).shuffle(seed=seed)
    else:
        from datasets import load_dataset
        ds = load_dataset(name, split="test").shuffle(seed=seed)
    return ds.select(range(min(n, len(ds))))


def exhaustive(graph, s, t, k, cap):
    out = []
    def dfs(n, pn, vis):
        d = len(pn) - 1
        if d > k: return
        if n == t and d >= 1:
            out.append(1); return
        if d == k: return
        for nb in graph.neighbors(n):
            if nb not in vis and len(out) < cap:
                dfs(nb, pn + [nb], vis | {nb})
    if s in graph and t in graph and s != t:
        dfs(s, [s], {s})
    return out


def main():
    import csv
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["rmanluo/RoG-cwq", "rmanluo/RoG-webqsp", "metaqa-3hop"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--cap", type=int, default=50)
    ap.add_argument("--out", default="results/retrieval_latency.csv")
    a = ap.parse_args()

    rows = []
    for ds_name in a.datasets:
        ds = load_ds(ds_name, a.n)
        for hop in (1, 2, 3, 4):
            tb, tj, te, nb, nj = [], [], [], [], []
            for i in range(len(ds)):
                s = ds[i]; g = build_graph(s["graph"])
                t0 = time.perf_counter(); B = get_paths_bfs(s["q_entity"], s["a_entity"], g, hop, a.cap); tb.append((time.perf_counter()-t0)*1000); nb.append(len(B))
                t0 = time.perf_counter(); O, _ = get_paths_optimized(s["q_entity"], s["a_entity"], g, hop, "cpu", a.cap); tj.append((time.perf_counter()-t0)*1000); nj.append(len(O))
                t0 = time.perf_counter()
                tot = 0
                for src in s["q_entity"]:
                    for tgt in s["a_entity"]:
                        if src != tgt and src in g and tgt in g:
                            tot += len(exhaustive(g, src, tgt, hop, a.cap))
                te.append((time.perf_counter()-t0)*1000)
            row = {"dataset": ds_name, "hop": hop, "n": len(ds), "cap": a.cap,
                   "bfs_ms": float(np.mean(tb)), "jbidfs_ms": float(np.mean(tj)),
                   "exhaustive_ms": float(np.mean(te)),
                   "bfs_paths": float(np.mean(nb)), "jbidfs_paths": float(np.mean(nj)),
                   "speedup_vs_bfs": float(np.mean(tb)/max(1e-9, np.mean(tj))),
                   "speedup_vs_exhaustive": float(np.mean(te)/max(1e-9, np.mean(tj)))}
            rows.append(row)
            print(f"{ds_name.split('/')[-1]:12} hop={hop}: BFS={row['bfs_ms']:8.2f}ms "
                  f"JbiDFS={row['jbidfs_ms']:8.2f}ms (vs BFS {row['speedup_vs_bfs']:.2f}x, "
                  f"vs exhaustive {row['speedup_vs_exhaustive']:.2f}x)  "
                  f"paths BFS={row['bfs_paths']:.1f}/JbiDFS={row['jbidfs_paths']:.1f}", flush=True)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
