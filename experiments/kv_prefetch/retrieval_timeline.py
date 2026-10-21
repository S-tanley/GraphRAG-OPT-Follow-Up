"""K0 (S-A): k-hop path enumeration on RoG subgraphs with per-path arrival times.

Same enumerator, sampling and cap as graphrag-main/exp_retrieval_latency.py (RoG's get_paths_bfs,
shuffle seed 42, cap 50), i.e. the setting behind the proposal's CWQ 4-hop number. The BFS is copied
here only to timestamp each path as it is appended; the returned path list is checked against
get_paths_bfs.

  python experiments/kv_prefetch/retrieval_timeline.py --datasets rmanluo/RoG-cwq rmanluo/RoG-webqsp \
      --n 200 --cap 50 --workers 16 --out outputs/kv_prefetch/retrieval_timeline.jsonl
"""
import argparse, json, os, sys, time
from collections import deque
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "graphrag-main"))
from graph_utils import build_graph, get_paths_bfs  # noqa: E402


def bfs_timed(q_entities, a_entities, graph, hop, cap):
    """get_paths_bfs with a timestamp (ms since start) for every emitted path."""
    t0 = time.perf_counter()
    out, arr = [], []
    for start in q_entities:
        if start not in graph:
            continue
        queue = deque([(start, [start], {start})])
        while queue:
            cur, nodes, vis = queue.popleft()
            if cur in a_entities and 1 <= len(nodes) - 1 <= hop:
                out.append([(nodes[i], graph[nodes[i]][nodes[i + 1]]["relation"], nodes[i + 1])
                            for i in range(len(nodes) - 1)])
                arr.append((time.perf_counter() - t0) * 1000)
                if len(out) >= cap:
                    return out, arr, (time.perf_counter() - t0) * 1000
                continue
            if len(nodes) - 1 < hop:
                for nb in graph.neighbors(cur):
                    if nb not in vis:
                        queue.append((nb, nodes + [nb], vis | {nb}))
    return out, arr, (time.perf_counter() - t0) * 1000


def work(args):
    ds_name, idx, s, hops, cap, check = args
    t0 = time.perf_counter()
    g = build_graph(s["graph"])
    build_ms = (time.perf_counter() - t0) * 1000
    rows = []
    for hop in hops:
        paths, arr, ms = bfs_timed(s["q_entity"], s["a_entity"], g, hop, cap)
        if check:
            assert paths == get_paths_bfs(s["q_entity"], s["a_entity"], g, hop, cap), (idx, hop)
        rows.append(dict(dataset=ds_name, idx=idx, question=s["question"], answer=s["answer"], hop=hop, cap=cap,
                         n_paths=len(paths), retrieval_ms=ms, build_ms=build_ms, arrivals_ms=arr, paths=paths,
                         nodes=g.number_of_nodes(), edges=g.number_of_edges()))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["rmanluo/RoG-cwq", "rmanluo/RoG-webqsp"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--cap", type=int, default=50)
    ap.add_argument("--hops", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--check", type=int, default=20, help="verify against get_paths_bfs on the first N queries")
    ap.add_argument("--out", default="outputs/kv_prefetch/retrieval_timeline.jsonl")
    a = ap.parse_args()
    from datasets import load_dataset
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for name in a.datasets:
            ds = load_dataset(name, split="test").shuffle(seed=42)
            ds = ds.select(range(min(a.n, len(ds))))
            items = [(name, i, {k: ds[i][k] for k in ("question", "answer", "q_entity", "a_entity", "graph")},
                      a.hops, a.cap, i < a.check) for i in range(len(ds))]
            with ProcessPoolExecutor(a.workers) as ex:
                res = [r for rows in ex.map(work, items) for r in rows]
            for r in res:
                f.write(json.dumps(r) + "\n")
            for hop in a.hops:
                rs = [r for r in res if r["hop"] == hop]
                ms = sorted(r["retrieval_ms"] for r in rs)
                print(f"{name.split('/')[-1]:12} hop={hop}: mean {sum(ms) / len(ms):8.1f} ms  p50 {ms[len(ms) // 2]:8.1f}"
                      f"  p90 {ms[int(len(ms) * .9)]:8.1f}  paths {sum(r['n_paths'] for r in rs) / len(rs):5.1f}"
                      f"  build {sum(r['build_ms'] for r in rs) / len(rs):5.1f} ms", flush=True)


if __name__ == "__main__":
    main()
