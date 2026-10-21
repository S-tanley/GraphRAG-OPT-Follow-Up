#!/usr/bin/env python3
"""
count_paths_vs_hops.py — Count how many paths BFS enumerates as hop depth grows.

Parallelized with multiprocessing for speed.  Each query gets a per-query
timeout so large graphs at high hops don't stall the run.

Usage:
    python count_paths_vs_hops.py                       # defaults
    python count_paths_vs_hops.py --samples 256 --max_hops 6
    python count_paths_vs_hops.py --workers 16 --timeout 10
"""

import argparse
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from datasets import load_dataset

# ── local imports ────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from graph_utils import build_graph, get_paths_bfs
from metaqa_adapter import load_metaqa


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_ds(name: str, split: str, n: int, seed: int):
    short = {"metaqa-1hop": 1, "metaqa-2hop": 2, "metaqa-3hop": 3}
    if name in short:
        hops = short[name]
        kb = SCRIPT_DIR / "metaQA" / "kb.txt"
        if kb.exists():
            ds = load_metaqa(hops=hops, split=split,
                             data_dir=str(SCRIPT_DIR / "metaQA"))
        elif hops == 3:
            ds = load_dataset("mhivanov123/metaqa-3hop",
                              split="validation" if split == "dev" else split)
        else:
            raise FileNotFoundError(
                f"MetaQA local files not found for metaqa-{hops}hop.")
    else:
        ds = load_dataset(name, split=split)
    if len(ds) > n:
        ds = ds.shuffle(seed=seed).select(range(n))
    return ds


# ═══════════════════════════════════════════════════════════════════════════════
# Worker function (runs in subprocess)
# ═══════════════════════════════════════════════════════════════════════════════

def _timeout_handler(signum, frame):
    raise TimeoutError()


def _worker(args):
    """Process one (query_idx, sample, hops_list) tuple.

    Builds the graph ONCE, then runs BFS for every hop in hops_list.
    Returns a list of result dicts (one per hop).
    """
    query_idx, sample, hops_list, timeout_s = args

    graph = build_graph(sample["graph"])
    nodes = graph.number_of_nodes()
    edges = graph.number_of_edges()
    degrees = [d for _, d in graph.degree()]
    avg_deg = float(np.mean(degrees)) if degrees else 0.0
    max_deg = max(degrees) if degrees else 0
    density = nx.density(graph)

    results = []
    for hop in hops_list:
        # Per-hop timeout
        timed_out = False
        if timeout_s and sys.platform != "win32":
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout_s)

        try:
            t0 = time.perf_counter()
            paths = get_paths_bfs(
                sample["q_entity"], sample["a_entity"],
                graph, hop_constraint=hop,
            )
            elapsed = time.perf_counter() - t0
        except TimeoutError:
            paths = []
            elapsed = timeout_s * 1000
            timed_out = True
        finally:
            if timeout_s and sys.platform != "win32":
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        n_paths = len(paths)
        path_lengths = [len(p) for p in paths]

        results.append({
            "query_idx":      query_idx,
            "hop_constraint": hop,
            "n_paths":        n_paths,
            "retrieval_ms":   elapsed * 1000 if not timed_out else timeout_s * 1000,
            "timed_out":      timed_out,
            "graph_nodes":    nodes,
            "graph_edges":    edges,
            "avg_degree":     avg_deg,
            "max_degree":     max_deg,
            "density":        density,
            "min_path_hops":  min(path_lengths) if path_lengths else 0,
            "max_path_hops":  max(path_lengths) if path_lengths else 0,
            "avg_path_hops":  float(np.mean(path_lengths)) if path_lengths else 0.0,
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main sweep (parallel)
# ═══════════════════════════════════════════════════════════════════════════════

def run_sweep(dataset_name, dataset, hops_range, workers, timeout_s):
    """Sweep all hops for all queries in parallel."""
    n = len(dataset)

    # Prepare work items: one per query, each carries all hops
    # Build plain dicts so they're picklable
    samples = []
    for i in range(n):
        s = dataset[i]
        samples.append({
            "graph":    s["graph"],
            "q_entity": s["q_entity"],
            "a_entity": s["a_entity"],
            "question": s["question"],
        })

    work = [(i, samples[i], hops_range, timeout_s) for i in range(n)]

    all_rows = []
    t0 = time.perf_counter()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, w): w[0] for w in work}
        done = 0
        for fut in as_completed(futures):
            done += 1
            if done % 50 == 0 or done == n:
                elapsed = time.perf_counter() - t0
                print(f"  [{dataset_name}] {done}/{n} queries  ({elapsed:.1f}s)")
            try:
                results = fut.result()
                for r in results:
                    r["dataset"] = dataset_name
                all_rows.extend(results)
            except Exception as e:
                qi = futures[fut]
                print(f"  WARN: query {qi} failed: {e}")
                for hop in hops_range:
                    all_rows.append({
                        "dataset": dataset_name, "query_idx": qi,
                        "hop_constraint": hop, "n_paths": 0,
                        "retrieval_ms": 0, "timed_out": True,
                        "graph_nodes": 0, "graph_edges": 0,
                        "avg_degree": 0, "max_degree": 0, "density": 0,
                        "min_path_hops": 0, "max_path_hops": 0,
                        "avg_path_hops": 0,
                    })

    elapsed = time.perf_counter() - t0
    print(f"  [{dataset_name}] Done in {elapsed:.1f}s")

    # Print per-hop summary
    df_tmp = pd.DataFrame(all_rows)
    for hop in hops_range:
        sub = df_tmp[df_tmp["hop_constraint"] == hop]
        arr = sub["n_paths"].values
        to = sub["timed_out"].sum()
        print(f"    hop={hop}:  paths/q  mean={arr.mean():.1f}  "
              f"med={np.median(arr):.0f}  max={arr.max()}  "
              f"total={arr.sum()}  timeout={to}")

    return all_rows


def main():
    import multiprocessing as mp

    p = argparse.ArgumentParser(description="Count BFS paths vs hop depth")
    p.add_argument("--datasets", nargs="+",
                   default=["rmanluo/RoG-cwq", "metaqa-3hop"])
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--max_hops", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=min(mp.cpu_count(), 16),
                   help="Parallel workers (default: min(ncpu, 16))")
    p.add_argument("--timeout", type=int, default=120,
                   help="Per-query timeout in seconds (default: 30)")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    hops_range = list(range(1, args.max_hops + 1))

    print(f"{'='*70}")
    print(f"  Path Enumeration vs Hop Depth")
    print(f"  Datasets:  {args.datasets}")
    print(f"  Samples:   {args.samples}")
    print(f"  Hops:      {hops_range}")
    print(f"  Workers:   {args.workers}")
    print(f"  Timeout:   {args.timeout}s per query")
    print(f"{'='*70}\n")

    all_rows = []

    for ds_name in args.datasets:
        print(f"Loading {ds_name} ...")
        ds = load_ds(ds_name, split="test", n=args.samples, seed=args.seed)
        print(f"  {len(ds)} samples loaded\n")

        rows = run_sweep(ds_name, ds, hops_range, args.workers, args.timeout)
        all_rows.extend(rows)
        print()

    # ── Build DataFrame ──────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)

    # ── Aggregate summary ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SUMMARY: mean paths per query")
    print(f"{'='*70}")

    agg = df.groupby(["dataset", "hop_constraint"]).agg(
        paths_mean=("n_paths", "mean"),
        paths_median=("n_paths", "median"),
        paths_max=("n_paths", "max"),
        paths_total=("n_paths", "sum"),
        paths_zero_pct=("n_paths", lambda x: 100 * (x == 0).sum() / len(x)),
        timed_out=("timed_out", "sum"),
        retrieval_ms_mean=("retrieval_ms", "mean"),
        avg_path_hops_mean=("avg_path_hops", "mean"),
        graph_nodes_mean=("graph_nodes", "mean"),
        graph_edges_mean=("graph_edges", "mean"),
    ).reset_index()

    print(agg.to_string(index=False, float_format="%.2f"))

    # ── Save ─────────────────────────────────────────────────────────────────
    if args.out:
        out_path = Path(args.out)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = SCRIPT_DIR / "profile_results" / f"path_counts_{ts}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_path, index=False)
    print(f"\nPer-query CSV  -> {out_path}")

    agg_path = out_path.with_name(out_path.stem + "_summary.csv")
    agg.to_csv(agg_path, index=False)
    print(f"Summary CSV    -> {agg_path}")

    # ── Growth ratios ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  GROWTH: paths at hop N / paths at hop 1")
    print(f"{'='*70}")

    for ds_name in args.datasets:
        sub = agg[agg["dataset"] == ds_name]
        base = sub[sub["hop_constraint"] == 1]["paths_mean"].values
        if len(base) == 0 or base[0] == 0:
            print(f"\n  {ds_name}: no paths at hop=1, skipping ratios")
            continue
        base_val = base[0]
        print(f"\n  {ds_name}:")
        for _, row in sub.iterrows():
            ratio = row["paths_mean"] / base_val
            to = int(row["timed_out"])
            print(f"    hop={int(row['hop_constraint'])}:  "
                  f"{row['paths_mean']:.1f} paths  ({ratio:.1f}x vs hop=1)"
                  f"{'  [' + str(to) + ' timed out]' if to else ''}")


if __name__ == "__main__":
    main()
