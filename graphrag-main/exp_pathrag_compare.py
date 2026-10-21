#!/usr/bin/env python3
"""
exp_pathrag_compare.py  —  Reviewer-B Q1b (JbiDFS vs PathRAG) + Q3 retrieval floor.

Answers the reviewer's three requested metrics on KGQA — retrieval latency,
retrieved context size, final answer accuracy — while keeping the comparison
HONEST: JbiDFS is an exact, completeness-preserving ENUMERATION accelerator;
PathRAG's flow-pruning is a lossy relevance SELECTION rule. They occupy different
stages of the (F, k, Sel) retriever and are COMPOSABLE, not competing.

PART A (CPU) — enumeration latency & completeness (the JbiDFS contribution):
  For each query, over its provided KG subgraph, enumerate s-t paths (q_entity ->
  a_entity, <= k hops) with (i) exhaustive DFS and (ii) JbiDFS. Record per-query
  latency and path counts; verify JbiDFS returns the SAME path set (completeness).
  -> the enumeration speedup is BATCH-INDEPENDENT (the Q3 retrieval floor), and it
     accelerates the same primitive PathRAG's inner loop (bounded DFS) uses.

PART B (GPU) — selection: context size & Hits@1 (the PathRAG contribution as Sel):
  Candidate pool = the leakage-free RoG-plan retrieved paths (precomputed). Apply
  two selection rules to the SAME pool, feed the SAME reader (greedy), measure
  retrieved context size (#paths, #tokens) and Hits@1:
    rog-sel      : our default (planner order, top --max_paths).
    pathrag-flow : faithful port of PathRAG flow-pruning (pathrag_flow.py), cap 15.
  Because the pool is identical, this isolates Sel; because JbiDFS is
  completeness-preserving, PathRAG-flow over JbiDFS == over exhaustive DFS (same
  paths) but faster -> JbiDFS accelerates PathRAG rather than competing with it.

We do NOT reimplement PathRAG's text-corpus indexing (LLM entity/relation
extraction): on a given KG there is nothing to construct. We disclose that the
flow weight source (Freebase relation IDs) differs from PathRAG's LLM edge
keywords, and we report it faithfully from source.

Usage:
  python exp_pathrag_compare.py --part A --datasets rmanluo/RoG-cwq rmanluo/RoG-webqsp \
     --n 300 --hop 3 --out results/retrieval_jbidfs.csv
  python exp_pathrag_compare.py --part B --model rmanluo/RoG \
     --datasets rmanluo/RoG-cwq rmanluo/RoG-webqsp --n 300 --out results/pathrag_compare.csv
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from graph_utils import build_graph, OptimizedPathEnumerator   # noqa: E402

PATHS_FILE = {
    "rmanluo/RoG-cwq": "precomputed_paths/paths_rmanluo-RoG-cwq.json",
    "rmanluo/RoG-webqsp": "precomputed_paths/paths_rmanluo-RoG-webqsp.json",
    "metaqa-3hop": "precomputed_paths/paths_metaqa-3hop.json",
}


def load_ds(name, n, seed=42):
    if name == "metaqa-3hop":
        from metaqa_adapter import load_metaqa
        ds = load_metaqa(hops=3, split="test", data_dir=str(HERE / "metaQA")).shuffle(seed=seed)
    else:
        from datasets import load_dataset
        ds = load_dataset(name, split="test").shuffle(seed=seed)
    return ds.select(range(min(n, len(ds))))


# ───────────────────────── PART A: enumeration latency ─────────────────────────

def exhaustive_dfs(graph, source, target, k, max_paths=10_000):
    """Plain exhaustive bounded DFS s->t (the baseline enumerator). Returns list
    of (h,rel,t)-triple paths."""
    out = []

    def dfs(node, path_nodes, visited):
        depth = len(path_nodes) - 1
        if depth > k:
            return
        if node == target and depth >= 1:
            rp = []
            for i in range(len(path_nodes) - 1):
                u, v = path_nodes[i], path_nodes[i + 1]
                rp.append((u, graph[u][v].get("relation", "rel"), v))
            out.append(rp)
            return
        if depth == k:
            return
        for nb in graph.neighbors(node):
            if nb not in visited and len(out) < max_paths:
                dfs(nb, path_nodes + [nb], visited | {nb})

    if source in graph and target in graph and source != target:
        dfs(source, [source], {source})
    return out


def _pathset(paths):
    return set(tuple((h, r, t) for (h, r, t) in p) for p in paths)


def run_part_A(datasets, n, hop, out, cap=50):
    import numpy as np
    rows = []
    for ds_name in datasets:
        ds = load_ds(ds_name, n)
        per = {"exh_ms": [], "jbidfs_ms": [], "exh_n": [], "jbidfs_n": [],
               "equal": [], "nodes": [], "edges": []}
        for i in range(len(ds)):
            s = ds[i]
            graph = build_graph(s["graph"])
            qents, aents = s["q_entity"], s["a_entity"]
            enum = OptimizedPathEnumerator("cpu")
            for src in qents:
                for tgt in aents:
                    if src == tgt or src not in graph or tgt not in graph:
                        continue
                    t0 = time.perf_counter()
                    exh = exhaustive_dfs(graph, src, tgt, hop, max_paths=cap)
                    t_exh = (time.perf_counter() - t0) * 1000
                    enum.reset_statistics(); enum.path_cache.clear()
                    t0 = time.perf_counter()
                    jb = enum.enumerate_paths(graph, src, tgt, hop, max_paths=cap)
                    t_jb = (time.perf_counter() - t0) * 1000
                    per["exh_ms"].append(t_exh); per["jbidfs_ms"].append(t_jb)
                    per["exh_n"].append(len(exh)); per["jbidfs_n"].append(len(jb))
                    # completeness is only well-defined when neither hit the cap
                    if len(exh) < cap and len(jb) < cap:
                        per["equal"].append(_pathset(exh) == _pathset(jb))
                    per["nodes"].append(graph.number_of_nodes())
                    per["edges"].append(graph.number_of_edges())
            if (i + 1) % 50 == 0:
                print(f"  [{ds_name}] {i+1}/{len(ds)}", flush=True)
        if not per["exh_ms"]:
            continue
        row = {
            "dataset": ds_name, "n_pairs": len(per["exh_ms"]),
            "exh_ms_mean": float(np.mean(per["exh_ms"])),
            "jbidfs_ms_mean": float(np.mean(per["jbidfs_ms"])),
            "speedup": float(np.mean(per["exh_ms"]) / max(1e-9, np.mean(per["jbidfs_ms"]))),
            "exh_paths_mean": float(np.mean(per["exh_n"])),
            "jbidfs_paths_mean": float(np.mean(per["jbidfs_n"])),
            "pathset_equal_pct": 100.0 * float(np.mean(per["equal"])) if per["equal"] else float("nan"),
            "n_uncapped_checked": len(per["equal"]),
            "graph_nodes_mean": float(np.mean(per["nodes"])),
            "graph_edges_mean": float(np.mean(per["edges"])),
        }
        rows.append(row)
        print(f"\n[{ds_name}] pairs={row['n_pairs']}  exhaustive={row['exh_ms_mean']:.3f}ms  "
              f"JbiDFS={row['jbidfs_ms_mean']:.3f}ms  speedup={row['speedup']:.2f}x  "
              f"path-set identical={row['pathset_equal_pct']:.1f}%", flush=True)
    _write_csv(out, rows)


# ───────────────────────── PART B: selection + reader ─────────────────────────

def run_part_B(datasets, n, model_name, max_paths, max_new_tokens, out):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from hf_reuse import full_generate
    from faithful_pfkv import path_to_str_rog
    from metrics import evaluate_metrics
    from pathrag_flow import pathrag_flow_select, TOTAL_EDGES

    device = "cuda"
    template = "rog" if "rog" in model_name.lower() else "fewshot"
    tok_src = "meta-llama/Llama-2-7b-hf" if "rog" in model_name.lower() else model_name
    tok = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device).eval()

    def ntok(paths):
        return sum(len(tok.encode(path_to_str_rog(p) + "\n", add_special_tokens=False)) for p in paths)

    rows = []
    for ds_name in datasets:
        ds = load_ds(ds_name, n)
        pf = json.load(open(HERE / PATHS_FILE[ds_name]))
        arms = {"rog-sel": [], "pathrag-flow": []}
        ctx = {"rog-sel": {"paths": [], "toks": [], "sel_ms": []},
               "pathrag-flow": {"paths": [], "toks": [], "sel_ms": []}}
        gold = []
        for i in range(len(ds)):
            s = ds[i]
            q = s["question"]
            gold.append(s["answer"])
            pool = [[tuple(t) for t in p] for p in pf.get(q, [])]
            # rog-sel: planner order, top max_paths
            t0 = time.perf_counter()
            sel_rog = pool[:max_paths]
            ctx["rog-sel"]["sel_ms"].append((time.perf_counter() - t0) * 1000)
            # pathrag-flow: faithful flow pruning, matched budget = max_paths
            t0 = time.perf_counter()
            sel_flow = pathrag_flow_select(pool, max_select=max_paths)
            ctx["pathrag-flow"]["sel_ms"].append((time.perf_counter() - t0) * 1000)
            for arm, sel in (("rog-sel", sel_rog), ("pathrag-flow", sel_flow)):
                ctx[arm]["paths"].append(len(sel))
                ctx[arm]["toks"].append(ntok(sel))
                pred = full_generate(model, tok, q, sel, template,
                                     max_paths=max(max_paths, TOTAL_EDGES),
                                     max_new_tokens=max_new_tokens, device=device)
                arms[arm].append(pred)
            if (i + 1) % 50 == 0:
                print(f"  [{ds_name}] {i+1}/{len(ds)}", flush=True)
        for arm in arms:
            m = evaluate_metrics(arms[arm], gold)
            row = {
                "dataset": ds_name, "selector": arm, "n": len(gold),
                "n_paths_mean": float(np.mean(ctx[arm]["paths"])),
                "n_tokens_mean": float(np.mean(ctx[arm]["toks"])),
                "sel_ms_mean": float(np.mean(ctx[arm]["sel_ms"])),
                "hits_at_1": m["hits_at_1"], "f1": m["f1"], "substr": m["accuracy"],
            }
            rows.append(row)
            print(f"\n[{ds_name}] {arm:13} n_paths={row['n_paths_mean']:.1f} "
                  f"n_tokens={row['n_tokens_mean']:.1f} sel={row['sel_ms_mean']:.3f}ms "
                  f"Hits@1={row['hits_at_1']:.4f} F1={row['f1']:.4f}", flush=True)
    _write_csv(out, rows)


def _write_csv(out, rows):
    import csv
    if not rows:
        print("no rows"); return
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["A", "B"], required=True)
    ap.add_argument("--datasets", nargs="+", default=["rmanluo/RoG-cwq", "rmanluo/RoG-webqsp"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--hop", type=int, default=3)
    ap.add_argument("--cap", type=int, default=50, help="max paths per s-t pair (PART A)")
    ap.add_argument("--model", default="rmanluo/RoG")
    ap.add_argument("--max_paths", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.part == "A":
        run_part_A(a.datasets, a.n, a.hop, a.out, cap=a.cap)
    else:
        run_part_B(a.datasets, a.n, a.model, a.max_paths, a.max_new_tokens, a.out)


if __name__ == "__main__":
    main()
