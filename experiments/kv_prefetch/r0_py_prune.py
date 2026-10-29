"""R0 follow-up: separate "language" from "algorithm" in the BFS speed-up.

Times graph_utils.get_paths_bfs against a pure-Python copy with target-distance pruning (same idea as
trav.cpp bfs_gold prune=1), on the K0 / R0 subset, and checks the two return the same list. Together with
traversal_profile.py (C++ without / with pruning) this fills the 2x2: {Python, C++} x {no pruning, pruning}.

  python experiments/kv_prefetch/r0_py_prune.py --workers 6 --tag mac
"""
import argparse
import json
import os
import sys
import time
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from traversal_profile import DATASETS, OUT_DIR, load_subset  # noqa: E402
from graph_utils import build_graph, get_paths_bfs  # noqa: E402


def bfs_prune_py(q_entities, a_entities, graph, hop, cap):
    """get_paths_bfs with target-distance pruning; returns the same list."""
    dist = {}
    q = deque()
    for t in a_entities:
        if t in graph and t not in dist:
            dist[t] = 0
            q.append(t)
    while q:
        u = q.popleft()
        if dist[u] + 1 > hop - 1:
            continue
        for v in graph.neighbors(u):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    far = hop + 1
    result = []
    for start in q_entities:
        if start not in graph:
            continue
        queue = deque([(start, [start], {start})])
        while queue:
            cur, nodes, vis = queue.popleft()
            d = len(nodes) - 1
            if cur in a_entities and 1 <= d <= hop:
                result.append([(nodes[i], graph[nodes[i]][nodes[i + 1]]["relation"], nodes[i + 1])
                               for i in range(d)])
                if len(result) >= cap:
                    return result
                continue
            if d < hop:
                for nb in graph.neighbors(cur):
                    if nb not in vis and dist.get(nb, far) <= hop - d - 1:
                        queue.append((nb, nodes + [nb], vis | {nb}))
    return result


ITEMS = None


def work(i):
    name, idx, s = ITEMS[i]
    G = build_graph(s["graph"])
    rows = []
    for hop in (3, 4):
        t0 = time.perf_counter()
        ref = get_paths_bfs(s["q_entity"], s["a_entity"], G, hop, 50)
        t_ref = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        got = bfs_prune_py(s["q_entity"], s["a_entity"], G, hop, 50)
        t_pr = (time.perf_counter() - t0) * 1000
        rows.append(dict(dataset=name, idx=idx, hop=hop, ms_py=t_ref, ms_py_prune=t_pr, match=int(got == ref)))
    return rows


def main():
    global ITEMS
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    ITEMS = [(name, i, s) for name in DATASETS for i, s in enumerate(load_subset(name, a.n))]
    import multiprocessing as mp
    out = os.path.join(OUT_DIR, f"py_prune_{a.tag}.jsonl")
    with open(out, "w") as f, mp.get_context("fork").Pool(a.workers) as pool:
        for rows in pool.imap_unordered(work, range(len(ITEMS))):
            for r in rows:
                f.write(json.dumps(r) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
