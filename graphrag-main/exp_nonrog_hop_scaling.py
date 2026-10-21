#!/usr/bin/env python3
"""
exp_nonrog_hop_scaling.py  —  Reviewer-B Q2.

Reproduce the hop-depth path-enumeration pathology (paper Fig:workload-c) with a
RETRIEVER-INDEPENDENT enumerator, to show the 1->4-hop latency/path-count blowup
is STRUCTURAL to hop-constrained s-t path enumeration, NOT an artifact of RoG's
implementation.

Primary enumerator  = networkx.all_simple_paths (textbook, zero project code, an
                      AC can reproduce it line-for-line).
Secondary (sanity)  = our generic BFS get_paths_bfs (also non-RoG; capped at 50).

Both run on the SAME provided subgraphs (the dataset's own `graph` field = the
Freebase / WikiMovies subgraph), the same graphs RoG uses — so the only thing that
changes vs the paper figure is the enumerator. If the blowup persists, it is a
property of the problem (HcPE), not of RoG.

Outputs: results/nonrog_hop_scaling.csv (per query x hop) + _summary.csv.

Usage:
    python exp_nonrog_hop_scaling.py --samples 200 --max_hops 4 \
        --datasets rmanluo/RoG-cwq rmanluo/RoG-webqsp metaqa-3hop \
        --out results/nonrog_hop_scaling.csv
"""
import argparse
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from datasets import load_dataset

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from graph_utils import build_graph, get_paths_bfs          # noqa: E402
from metaqa_adapter import load_metaqa                       # noqa: E402

# Hard ceiling on the number of simple paths we will actually materialize per
# (query, s, t, hop) — purely so a single explosive pair cannot OOM the box. We
# record whether the ceiling was hit; a hit is itself evidence of the blowup.
PATH_CEIL = 200_000


def load_ds(name, split, n, seed):
    short = {"metaqa-1hop": 1, "metaqa-2hop": 2, "metaqa-3hop": 3}
    if name in short:
        ds = load_metaqa(hops=short[name], split=split,
                         data_dir=str(SCRIPT_DIR / "metaQA"))
    else:
        ds = load_dataset(name, split=split)
    if len(ds) > n:
        ds = ds.shuffle(seed=seed).select(range(n))
    return ds


def _timeout_handler(signum, frame):
    raise TimeoutError()


def _count_simple_paths(G, sources, targets, cutoff, ceil):
    """Total simple paths (length 1..cutoff edges) over all s in sources, t in
    targets, using the textbook nx.all_simple_paths. cutoff is in EDGES."""
    total = 0
    hit_ceiling = False
    for s in sources:
        if s not in G:
            continue
        for t in targets:
            if t not in G or s == t:
                continue
            # all_simple_paths cutoff is in number of NODES-1 == edges == hops
            for _ in nx.all_simple_paths(G, s, t, cutoff=cutoff):
                total += 1
                if total >= ceil:
                    return total, True
    return total, hit_ceiling


def _worker(args):
    query_idx, sample, hops_list, timeout_s = args
    G = build_graph(sample["graph"])
    nodes = G.number_of_nodes()
    edges = G.number_of_edges()
    degs = [d for _, d in G.degree()]
    avg_deg = float(np.mean(degs)) if degs else 0.0
    max_deg = max(degs) if degs else 0

    sources = sample["q_entity"]
    targets = sample["a_entity"]

    out = []
    for hop in hops_list:
        # --- Primary: networkx all_simple_paths (uncapped, true enumeration) ---
        timed_out = False
        hit_ceiling = False
        if timeout_s and sys.platform != "win32":
            old = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout_s)
        try:
            t0 = time.perf_counter()
            n_nx, hit_ceiling = _count_simple_paths(G, sources, targets, hop, PATH_CEIL)
            nx_ms = (time.perf_counter() - t0) * 1000.0
        except TimeoutError:
            n_nx = -1
            nx_ms = timeout_s * 1000.0
            timed_out = True
        finally:
            if timeout_s and sys.platform != "win32":
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old)

        # --- Secondary: our generic BFS (non-RoG, capped at 50) ---
        bfs_to = False
        if timeout_s and sys.platform != "win32":
            old = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout_s)
        try:
            t0 = time.perf_counter()
            bfs_paths = get_paths_bfs(sources, targets, G, hop_constraint=hop)
            bfs_ms = (time.perf_counter() - t0) * 1000.0
            n_bfs = len(bfs_paths)
        except TimeoutError:
            n_bfs = -1
            bfs_ms = timeout_s * 1000.0
            bfs_to = True
        finally:
            if timeout_s and sys.platform != "win32":
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old)

        out.append({
            "query_idx": query_idx, "hop": hop,
            "nx_paths": n_nx, "nx_ms": nx_ms, "nx_timed_out": timed_out,
            "nx_hit_ceiling": hit_ceiling,
            "bfs_paths": n_bfs, "bfs_ms": bfs_ms, "bfs_timed_out": bfs_to,
            "graph_nodes": nodes, "graph_edges": edges,
            "avg_degree": avg_deg, "max_degree": max_deg,
        })
    return out


def run(ds_name, ds, hops, workers, timeout_s):
    n = len(ds)
    samples = [{"graph": ds[i]["graph"], "q_entity": ds[i]["q_entity"],
                "a_entity": ds[i]["a_entity"]} for i in range(n)]
    work = [(i, samples[i], hops, timeout_s) for i in range(n)]
    rows = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_worker, w): w[0] for w in work}
        done = 0
        for fut in as_completed(futs):
            done += 1
            if done % 25 == 0 or done == n:
                print(f"  [{ds_name}] {done}/{n}  ({time.perf_counter()-t0:.1f}s)", flush=True)
            try:
                for r in fut.result():
                    r["dataset"] = ds_name
                    rows.append(r)
            except Exception as e:
                print(f"  WARN q{futs[fut]} failed: {e}", flush=True)
    return rows


def main():
    import multiprocessing as mp
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+",
                   default=["rmanluo/RoG-cwq", "rmanluo/RoG-webqsp", "metaqa-3hop"])
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--max_hops", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=min(mp.cpu_count(), 24))
    p.add_argument("--timeout", type=int, default=90)
    p.add_argument("--out", type=str, default="results/nonrog_hop_scaling.csv")
    a = p.parse_args()

    hops = list(range(1, a.max_hops + 1))
    print(f"NON-RoG hop scaling | datasets={a.datasets} samples={a.samples} "
          f"hops={hops} workers={a.workers} timeout={a.timeout}s ceil={PATH_CEIL}", flush=True)

    all_rows = []
    for name in a.datasets:
        print(f"\nLoading {name} ...", flush=True)
        ds = load_ds(name, "test", a.samples, a.seed)
        print(f"  {len(ds)} samples", flush=True)
        all_rows.extend(run(name, ds, hops, a.workers, a.timeout))

    df = pd.DataFrame(all_rows)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    # Aggregate. For path counts, treat timeouts/ceilings as the recorded value
    # (timeouts excluded from the mean but counted) to keep the mean honest.
    def agg_one(g):
        nx_valid = g[(g.nx_paths >= 0)]
        bfs_valid = g[(g.bfs_paths >= 0)]
        return pd.Series({
            "nx_paths_mean": nx_valid.nx_paths.mean() if len(nx_valid) else float("nan"),
            "nx_paths_median": nx_valid.nx_paths.median() if len(nx_valid) else float("nan"),
            "nx_ms_mean": g.nx_ms.mean(),
            "nx_timeouts": int(g.nx_timed_out.sum()),
            "nx_ceilings": int(g.nx_hit_ceiling.sum()),
            "bfs_paths_mean": bfs_valid.bfs_paths.mean() if len(bfs_valid) else float("nan"),
            "bfs_ms_mean": g.bfs_ms.mean(),
            "graph_nodes_mean": g.graph_nodes.mean(),
            "graph_edges_mean": g.graph_edges.mean(),
            "avg_degree_mean": g.avg_degree.mean(),
            "n": len(g),
        })
    agg = df.groupby(["dataset", "hop"]).apply(agg_one).reset_index()
    agg.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)

    print(f"\n{'='*78}\nSUMMARY (networkx all_simple_paths, the textbook enumerator)\n{'='*78}")
    print(agg.to_string(index=False, float_format="%.2f"))

    print(f"\n{'='*78}\nGROWTH vs hop=1 (mean paths and mean latency)\n{'='*78}")
    for name in a.datasets:
        sub = agg[agg.dataset == name].sort_values("hop")
        if sub.empty:
            continue
        p1 = sub[sub.hop == 1]
        if p1.empty or not (p1.nx_paths_mean.values[0] > 0):
            print(f"\n  {name}: no hop-1 baseline")
            continue
        b_paths = p1.nx_paths_mean.values[0]
        b_ms = p1.nx_ms_mean.values[0]
        print(f"\n  {name}:")
        for _, r in sub.iterrows():
            pr = r.nx_paths_mean / b_paths if b_paths else float("nan")
            mr = r.nx_ms_mean / b_ms if b_ms else float("nan")
            note = []
            if r.nx_timeouts:
                note.append(f"{int(r.nx_timeouts)} timeout")
            if r.nx_ceilings:
                note.append(f"{int(r.nx_ceilings)} ceil")
            tag = ("  [" + ", ".join(note) + "]") if note else ""
            print(f"    hop={int(r.hop)}: {r.nx_paths_mean:9.1f} paths ({pr:7.1f}x)  "
                  f"{r.nx_ms_mean:9.2f} ms ({mr:8.1f}x){tag}")

    print(f"\nWrote {out} and {out.with_name(out.stem + '_summary.csv')}")


if __name__ == "__main__":
    main()
