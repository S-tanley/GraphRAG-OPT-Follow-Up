"""R0.4: the two suspected issues in graphrag-main/retrieval.py BatchOptimizedRetrieval.retrieve_batch.

(1) pair_redundancy_ratio: total_pairs and unique_pairs are both len(pair_to_graph), so it is always 0.
    Here: the returned stats next to the correct ratio 1 - unique pairs / pair occurrences.
(2) pair_to_graph[pair] = graph: a (q_entity, a_entity) pair shared by several questions of a batch is
    enumerated only on the LAST question's subgraph (RoG gives every question its own subgraph).
    Here: BatchOptimizedRetrieval as is, against the same code with each question's paths enumerated on its
    own subgraph and assembled in the same (global pair) order, so only the graph overwrite differs.

The original class is imported and called unchanged. Settings of its callers (hop 3; ablation_kgqa / hf_reuse
use max_paths 8 with a whole shard as one batch; main / main_benchmark / experiment_scheduling use no cap and
batches of 16-64). Questions: the first --n of the test split in dataset order, cut into consecutive batches.

Also (--jbidfs_vs_bfs): the rebuttal says JbiDFS returns the same path set as BFS. Compare get_paths_bfs and
get_paths_optimized per question on the K0 subset (seed-42 shuffle, first 200), cap 50, hops 2-4.

  python experiments/kv_prefetch/r0_batch_check.py --n 512
  python experiments/kv_prefetch/r0_batch_check.py --jbidfs_vs_bfs
"""
import argparse
import contextlib
import io
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "graphrag-main"))
import torch  # noqa: E402

torch.cuda.is_available = lambda: False      # CPU only; see traversal_profile.py
from graph_utils import OptimizedPathEnumerator, build_graph, get_paths_bfs, get_paths_optimized  # noqa: E402
from retrieval import BatchOptimizedRetrieval  # noqa: E402


def fixed_retrieve(batch, hop, cap):
    """retrieve_batch's logic with each question's pairs enumerated on its own subgraph."""
    enum = OptimizedPathEnumerator("cpu")
    pair_order, pair_to_indices = {}, defaultdict(list)
    graphs = [build_graph(q["graph"]) for q in batch]
    for idx, q in enumerate(batch):
        for src in q["q_entity"]:
            for tgt in q["a_entity"]:
                if src != tgt:
                    pair_order.setdefault((src, tgt), len(pair_order))
                    pair_to_indices[(src, tgt)].append(idx)
    query_paths = defaultdict(list)
    for pair in pair_order:
        for idx in pair_to_indices[pair]:
            enum.path_cache.clear()                  # the cache key has no graph identity
            query_paths[idx].extend(enum.enumerate_paths(graphs[idx], pair[0], pair[1], hop, max_paths=50))
    return [query_paths[i][:cap] if cap else query_paths[i] for i in range(len(batch))], graphs


def jbidfs_vs_bfs(out):
    from datasets import load_dataset
    res = []
    for name in ("rmanluo/RoG-cwq", "rmanluo/RoG-webqsp"):
        ds = load_dataset(name, split="test").shuffle(seed=42).select(range(200))
        for hop in (2, 3, 4):
            c = defaultdict(int)
            for i in range(len(ds)):
                s = ds[i]
                G = build_graph(s["graph"])
                b = {tuple(map(tuple, p)) for p in get_paths_bfs(s["q_entity"], s["a_entity"], G, hop, 50)}
                j_list, _ = get_paths_optimized(s["q_entity"], s["a_entity"], G, hop, "cpu", 50)
                j = {tuple(map(tuple, p)) for p in j_list}
                nonsimple = sum(len({t[0] for t in p} | {p[-1][2]}) < len(p) + 1 for p in j)
                c["q"] += 1
                c["equal"] += b == j
                full = len(b) < 50 and len(j_list) < 50          # neither hit the cap
                c["uncapped"] += full
                c["uncapped_equal"] += full and b == j
                c["j_nonsimple_q"] += nonsimple > 0
                c["j_nonsimple_paths"] += nonsimple
                c["j_paths"] += len(j_list)
                c["j_not_in_bfs_uncapped"] += len(j - b) if full else 0
                c["b_not_in_j_uncapped"] += len(b - j) if full else 0
            row = dict(dataset=name.split("-")[-1], hop=hop, **c)
            res.append(row)
            print(json.dumps(row), flush=True)
    json.dump(res, open(out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jbidfs_vs_bfs", action="store_true")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--hop", type=int, default=3)
    ap.add_argument("--batches", type=int, nargs="+", default=[16, 64, 512])
    ap.add_argument("--caps", type=int, nargs="+", default=[8, 0])
    ap.add_argument("--out", default="outputs/kv_prefetch/v2/R0/batch_check.json")
    a = ap.parse_args()
    if a.jbidfs_vs_bfs:
        return jbidfs_vs_bfs(os.path.join(os.path.dirname(a.out), "jbidfs_vs_bfs.json"))
    from datasets import load_dataset
    res = []
    for name in ("rmanluo/RoG-cwq", "rmanluo/RoG-webqsp"):
        ds = load_dataset(name, split="test")
        qs = [{k: ds[i][k] for k in ("question", "q_entity", "a_entity", "graph")} for i in range(min(a.n, len(ds)))]
        for B in a.batches:
            for cap in a.caps:
                agg = defaultdict(float)
                for s in range(0, len(qs), B):
                    batch = qs[s:s + B]
                    with contextlib.redirect_stdout(io.StringIO()):
                        got, stats = BatchOptimizedRetrieval(batch, device="cpu").retrieve_batch(a.hop, True, cap or None)
                    ref, graphs = fixed_retrieve(batch, a.hop, cap)
                    occ = sum(1 for q in batch for x in q["q_entity"] for y in q["a_entity"] if x != y)
                    uniq = len({(x, y) for q in batch for x in q["q_entity"] for y in q["a_entity"] if x != y})
                    agg["occurrences"] += occ
                    agg["unique"] += uniq
                    agg["reported_pair_redundancy_sum"] += stats["pair_redundancy_ratio"]
                    agg["reported_path_redundancy_sum"] += stats["path_redundancy_ratio"]
                    agg["batches"] += 1
                    for i, q in enumerate(batch):
                        g_paths = [[tuple(t) for t in p] for p in got[i]["paths"]]
                        r_paths = [[tuple(t) for t in p] for p in ref[i]]
                        agg["questions"] += 1
                        agg["differ"] += g_paths != r_paths
                        agg["differ_set"] += {tuple(p) for p in g_paths} != {tuple(p) for p in r_paths}
                        agg["lost_all"] += bool(r_paths) and not g_paths
                        agg["paths"] += len(g_paths)
                        G = graphs[i]
                        agg["paths_foreign"] += sum(any(not G.has_edge(u, v) for u, _r, v in p) for p in g_paths)
                        aset = set(q["a_entity"])
                        hit = lambda ps: any(t[0] in aset or t[2] in aset for p in ps for t in p)  # noqa: E731
                        agg["cover_asis"] += hit(g_paths)
                        agg["cover_fixed"] += hit(r_paths)
                n = agg["questions"]
                row = dict(dataset=name.split("-")[-1], batch=B, cap=cap or None, hop=a.hop, questions=int(n),
                           true_pair_redundancy=1 - agg["unique"] / max(1, agg["occurrences"]),
                           reported_pair_redundancy_mean=agg["reported_pair_redundancy_sum"] / agg["batches"],
                           reported_path_redundancy_mean=agg["reported_path_redundancy_sum"] / agg["batches"],
                           q_paths_differ=agg["differ"] / n, q_pathset_differ=agg["differ_set"] / n,
                           q_lost_all=agg["lost_all"] / n,
                           foreign_path_frac=agg["paths_foreign"] / max(1, agg["paths"]),
                           cover_asis=agg["cover_asis"] / n, cover_fixed=agg["cover_fixed"] / n)
                res.append(row)
                print(json.dumps(row), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
