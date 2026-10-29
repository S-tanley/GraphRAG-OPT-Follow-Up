"""R0 (KV prefetch v2): how much of KG path-retrieval time is implementation overhead, and how long
retrieval takes when the answer entity is not known.

Same questions and subgraphs as K0 (RoG per-question subgraphs, test split shuffled with seed 42,
first --n), hops 3 / 4, cap 50. Per (question, hop) and mode:

  gold (target = gold answer entities; the RoG / GraphRAG-Opt setting)
    gold_py         graph_utils.get_paths_bfs as is (RoG BFS), one run
    gold_py_ins     instrumented copy: per-path arrival times, states dequeued, edges scanned
    jbidfs_py       graph_utils.get_paths_optimized (GraphRAG-Opt Algorithm 1 / JbiDFS), one run
    jbidfs_ins      instrumented copy: DFS calls, edges scanned, when each pair's paths come out
    gold_cpp        the same BFS in C++ (trav.cpp), no pruning
    gold_cpp_prune  C++ BFS with target-distance pruning (R0.2)
                    both C++ versions must return exactly get_paths_bfs's list (order and cap cut)
  open (a) answer unknown: paths from the topic entities to any node
    open_cap_py     BFS over simple paths, first `cap` (Python, networkx)
    open_cap        the same in C++ (checked against open_cap_py)
    open_top        C++: all simple paths up to `hop` (DFS), keep the top `cap` by mean relation score;
                    stops at --limit enumerated paths (flag `truncated`)
    open_beam       C++: beam, keep the top --beam extensions per hop by mean relation score, then the top `cap`
  plan (b) RoG relation-plan grounding (graph_utils._bfs_with_rule, step (1) of get_paths_rog_plan)
    rog_plan        the relation plans are read back from the saved RoG-plan retrieval
                    (graphrag-main/precomputed_paths_v2: hop 3, 30 paths); the planner LLM is not run,
                    so this is traversal time only. Run once per question (hop column = 3), at cap 30 and 50.

Relation scores for open_top / open_beam: cosine(question, relation) with all-MiniLM-L6-v2 (the model
graph_utils.get_paths_dense uses), relation text with '.' / '_' as spaces; a path scores the mean over its
relations. `relscore` computes them once (and times one question embedding on CPU); the scores are inputs
here, so the traversal timings do not include the encoder.

C++ timings are the median of up to --reps runs of the enumerator itself (ms_core; a call >= 50 ms is not
repeated); ms adds the Python glue (mapping entity names to ids, turning node ids back into triples). Graph build (networkx) and CSR build are
reported separately and excluded from all retrieval times, as in K0.

  python experiments/kv_prefetch/traversal_profile.py relscore
  python experiments/kv_prefetch/traversal_profile.py run --workers 6 --tag mac
"""
import argparse
import ctypes
import heapq
import json
import os
import platform
import subprocess
import sys
import time
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "graphrag-main"))
from graph_utils import (OptimizedPathEnumerator, _bfs_with_rule, build_graph,  # noqa: E402
                         get_paths_bfs, get_paths_optimized)
import torch  # noqa: E402

# OptimizedPathEnumerator.__init__ calls torch.cuda.is_available(); on the cluster the first call initialises
# the CUDA driver (~20 s per process), which landed in the first JbiDFS timing of every worker. This is a
# CPU-only job: never touch the driver.
torch.cuda.is_available = lambda: False

DATASETS = ["rmanluo/RoG-cwq", "rmanluo/RoG-webqsp"]
OUT_DIR = os.path.join(ROOT, "outputs", "kv_prefetch", "v2", "R0")
PLAN_DIR = os.path.join(ROOT, "graphrag-main", "precomputed_paths_v2")


def load_subset(name, n):
    from datasets import load_dataset
    ds = load_dataset(name, split="test").shuffle(seed=42)
    ds = ds.select(range(min(n, len(ds))))
    return [{k: ds[i][k] for k in ("question", "answer", "q_entity", "a_entity", "graph")} for i in range(len(ds))]


def ms_since(t0):
    return (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------- C++ enumerators (trav.cpp)

_LIB = None


def lib():
    global _LIB
    if _LIB is None:
        ext = "dylib" if sys.platform == "darwin" else "so"
        src, so = os.path.join(HERE, "trav.cpp"), os.path.join(HERE, "build", f"libtrav.{ext}")
        if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
            os.makedirs(os.path.dirname(so), exist_ok=True)
            cxx = "clang++" if sys.platform == "darwin" else "g++"
            subprocess.check_call([cxx, "-O3", "-std=c++17", "-shared", "-fPIC", src, "-o", so])
        L = ctypes.CDLL(so)
        i32 = np.ctypeslib.ndpointer(np.int32, flags="C")
        f32 = np.ctypeslib.ndpointer(np.float32, flags="C")
        f64 = np.ctypeslib.ndpointer(np.float64, flags="C")
        i64 = np.ctypeslib.ndpointer(np.int64, flags="C")
        ci, cll = ctypes.c_int, ctypes.c_longlong
        L.bfs_gold.argtypes = [ci, i32, i32, i32, ci, i32, ci, ci, ci, ci, i32, i32, f64, i64, f64]
        L.bfs_gold.restype = ci
        L.expand.argtypes = [ci, i32, i32, f32, i32, ci, ci, ci, ci, ci, cll, i32, i32, f64, f64, i64, f64]
        L.expand.restype = ci
        _LIB = L
    return _LIB


class CSR:
    """networkx graph -> CSR with G.adj's neighbour order, so traversal order matches graph.neighbors()."""

    def __init__(self, G):
        t0 = time.perf_counter()
        self.G = G
        self.names = list(G.nodes)
        self.idx = {u: i for i, u in enumerate(self.names)}
        deg = np.fromiter((len(G.adj[u]) for u in self.names), np.int32, count=len(self.names))
        self.indptr = np.zeros(len(self.names) + 1, np.int32)
        np.cumsum(deg, out=self.indptr[1:])
        idx = self.idx
        self.indices = np.fromiter((idx[v] for u in self.names for v in G.adj[u]), np.int32,
                                   count=int(self.indptr[-1]))
        self.edge_rel = [G.adj[u][v]["relation"] for u in self.names for v in G.adj[u]]
        self.build_ms = ms_since(t0)

    def triples(self, nodes):
        nm, G = self.names, self.G
        return [(nm[a], G[nm[a]][nm[b]]["relation"], nm[b]) for a, b in zip(nodes[:-1], nodes[1:])]

    def edge_scores(self, relscore):
        """Per-edge float32 score for one question (timed by the caller: part of open_* retrieval)."""
        rels = sorted(set(self.edge_rel))
        rid = {r: i for i, r in enumerate(rels)}
        table = np.array([relscore.get(r, 0.0) for r in rels], np.float32)
        return table[np.fromiter((rid[r] for r in self.edge_rel), np.int32, count=len(self.edge_rel))]


def _run_cpp(fn, reps, cap, hop):
    """Call fn(buffers); repeat up to `reps` times while a call takes < 50 ms; return (n, buffers, median core ms)."""
    cores = []
    while True:
        b = dict(nodes=np.zeros(cap * (hop + 1), np.int32), length=np.zeros(cap, np.int32),
                 t=np.zeros(cap, np.float64), score=np.zeros(cap, np.float64),
                 cnt=np.zeros(8, np.int64), tot=np.zeros(1, np.float64))
        n = fn(b)
        cores.append(b["tot"][0] / 1000)
        if len(cores) >= reps or cores[0] >= 50:
            b["first_ms"] = cores[0]
            return n, b, float(np.median(cores))


def _paths_from(csr, n, b, hop):
    return [csr.triples(b["nodes"][i * (hop + 1): i * (hop + 1) + b["length"][i]].tolist()) for i in range(n)]


def gold_cpp(csr, q_ent, a_ent, hop, cap, prune, reps):
    t0 = time.perf_counter()
    starts = np.array([csr.idx[s] for s in q_ent if s in csr.idx], np.int32)
    targets = np.array(sorted({csr.idx[t] for t in a_ent if t in csr.idx}), np.int32)
    L = lib()

    def fn(b):
        return L.bfs_gold(len(csr.names), csr.indptr, csr.indices, starts, len(starts), targets, len(targets),
                          hop, cap, prune, b["nodes"], b["length"], b["t"], b["cnt"], b["tot"])
    pre = ms_since(t0)
    n, b, core = _run_cpp(fn, reps, cap, hop)
    t1 = time.perf_counter()
    paths = _paths_from(csr, n, b, hop)
    b["cnt"][7] = int(b["first_ms"] * 1000)              # first (cold) call, in us
    return paths, (b["t"][:n] / 1000).tolist(), core, core + pre + ms_since(t1), b["cnt"]


def expand_cpp(csr, escore, q_ent, hop, mode, N, W, limit, reps):
    t0 = time.perf_counter()
    starts = np.array([csr.idx[s] for s in q_ent if s in csr.idx], np.int32)
    L = lib()

    def fn(b):
        return L.expand(len(csr.names), csr.indptr, csr.indices, escore, starts, len(starts), hop, mode, N, W,
                        limit, b["nodes"], b["length"], b["score"], b["t"], b["cnt"], b["tot"])
    pre = ms_since(t0)
    n, b, core = _run_cpp(fn, reps, N, hop)
    t1 = time.perf_counter()
    paths = _paths_from(csr, n, b, hop)
    b["cnt"][7] = int(b["first_ms"] * 1000)              # first (cold) call, in us
    return paths, b["score"][:n].tolist(), (b["t"][:n] / 1000).tolist(), core, core + pre + ms_since(t1), b["cnt"]


# ---------------------------------------------------------------- Python references / instrumented copies

def bfs_ins(q_entities, a_entities, graph, hop, cap):
    """get_paths_bfs plus per-path arrival times (ms) and work counters."""
    t0 = time.perf_counter()
    out, arr, deq, scan, uniq = [], [], 0, 0, set()
    for start in q_entities:
        if start not in graph:
            continue
        queue = deque([(start, [start], {start})])
        while queue:
            cur, nodes, vis = queue.popleft()
            deq += 1
            uniq.add(cur)
            if cur in a_entities and 1 <= len(nodes) - 1 <= hop:
                out.append([(nodes[i], graph[nodes[i]][nodes[i + 1]]["relation"], nodes[i + 1])
                            for i in range(len(nodes) - 1)])
                arr.append(ms_since(t0))
                if len(out) >= cap:
                    return out, arr, dict(states=deq, edges=scan, nodes=len(uniq))
                continue
            if len(nodes) - 1 < hop:
                for nb in graph.neighbors(cur):
                    scan += 1
                    if nb not in vis:
                        queue.append((nb, nodes + [nb], vis | {nb}))
    return out, arr, dict(states=deq, edges=scan, nodes=len(uniq))


class JbiDFSIns(OptimizedPathEnumerator):
    """OptimizedPathEnumerator with counters in its two DFS routines (bodies copied from graph_utils)."""

    def __init__(self):
        super().__init__("cpu")
        self.calls = 0
        self.scan = 0
        self.indep_time = 0.0

    def bidirectional_dfs(self, graph, source, target, join_node, hop_constraint, max_paths=50):
        start_time = time.time()

        def dfs_paths(g, start, end, max_hops, current_path, all_paths, visited):
            self.calls += 1
            if len(current_path) - 1 > max_hops:
                return
            if start == end and len(current_path) > 1:
                rel_path = []
                for i in range(len(current_path) - 1):
                    u, v = current_path[i], current_path[i + 1]
                    if g.has_edge(u, v):
                        rel_path.append((u, g[u][v].get("relation", "related_to"), v))
                if rel_path:
                    all_paths.append(rel_path)
                return
            if len(all_paths) >= max_paths:
                return
            for neighbor in g.neighbors(start):
                self.scan += 1
                if neighbor not in visited:
                    dfs_paths(g, neighbor, end, max_hops, current_path + [neighbor], all_paths, visited | {neighbor})

        source_to_join = []
        if join_node != source:
            dfs_paths(graph, source, join_node, hop_constraint, [source], source_to_join, {source})
        join_to_target = []
        if join_node != target:
            dfs_paths(graph, join_node, target, hop_constraint, [join_node], join_to_target, {join_node})
        self.stats["bidirectional_dfs_time"] += time.time() - start_time
        return source_to_join, join_to_target

    def independent_dfs(self, graph, source, target, hop_constraint, max_paths=50):
        start_time = time.time()

        def dfs_all(g, start, end, max_hops, current_path, visited):
            self.calls += 1
            if len(current_path) - 1 > max_hops:
                return []
            if start == end and len(current_path) > 1:
                rel_path = []
                for i in range(len(current_path) - 1):
                    u, v = current_path[i], current_path[i + 1]
                    if g.has_edge(u, v):
                        rel_path.append((u, g[u][v].get("relation", "related_to"), v))
                return [rel_path] if rel_path else []
            all_paths = []
            for neighbor in g.neighbors(start):
                self.scan += 1
                if neighbor not in visited:
                    all_paths.extend(dfs_all(g, neighbor, end, max_hops, current_path + [neighbor],
                                             visited | {neighbor}))
                    if len(all_paths) >= max_paths:
                        break
            return all_paths

        res = dfs_all(graph, source, target, hop_constraint, [source], {source})
        self.indep_time += time.time() - start_time
        return res


def jbidfs_ins(q_entities, a_entities, graph, hop, cap):
    """get_paths_optimized (same loop) with the time each (source, target) pair's paths become available."""
    t0 = time.perf_counter()
    enum = JbiDFSIns()
    out, arr = [], []
    for start in q_entities:
        for end in a_entities:
            if start != end:
                paths = enum.enumerate_paths(graph, start, end, hop, max_paths=50)
                out.extend(paths)
                arr.extend([ms_since(t0)] * len(paths))
                if len(out) >= cap:
                    out, arr = out[:cap], arr[:cap]
                    return out, arr, enum
    return out, arr, enum


def open_cap_py(q_entities, graph, hop, cap):
    """Answer unknown: BFS over simple paths from the topic entities; every node is an endpoint; first `cap`."""
    t0 = time.perf_counter()
    out, arr = [], []
    for start in q_entities:
        if start not in graph:
            continue
        queue = deque([(start, [start], {start})])
        while queue:
            cur, nodes, vis = queue.popleft()
            if len(nodes) > 1:
                out.append([(nodes[i], graph[nodes[i]][nodes[i + 1]]["relation"], nodes[i + 1])
                            for i in range(len(nodes) - 1)])
                arr.append(ms_since(t0))
                if len(out) >= cap:
                    return out, arr
            if len(nodes) - 1 < hop:
                for nb in graph.neighbors(cur):
                    if nb not in vis:
                        queue.append((nb, nodes + [nb], vis | {nb}))
    return out, arr


def open_top_py(csr, escore, q_ent, hop, N):
    """Reference for expand mode 1 (same DFS order, same tie-break)."""
    ip, ix = csr.indptr, csr.indices
    heap, order = [], [0]

    def dfs(path, s):
        u = path[-1]
        for e in range(ip[u], ip[u + 1]):
            v = int(ix[e])
            if v in path:
                continue
            s1 = s + float(escore[e])
            p1 = path + [v]
            item = (s1 / (len(p1) - 1), -order[0], p1)
            order[0] += 1
            if len(heap) < N:
                heapq.heappush(heap, item)
            elif item[:2] > heap[0][:2]:
                heapq.heapreplace(heap, item)
            if len(p1) - 1 < hop:
                dfs(p1, s1)

    for s in q_ent:
        if s in csr.idx:
            dfs([csr.idx[s]], 0.0)
    return [(sc, p) for sc, _, p in sorted(heap, key=lambda x: x[:2], reverse=True)]


def open_beam_py(csr, escore, q_ent, hop, N, W):
    """Reference for expand mode 2."""
    ip, ix = csr.indptr, csr.indices
    frontier = [([csr.idx[s]], 0.0) for s in q_ent if s in csr.idx]
    cand, k = [], 0
    for _ in range(hop):
        children = []
        for path, s in frontier:
            u = path[-1]
            for e in range(ip[u], ip[u + 1]):
                v = int(ix[e])
                if v in path:
                    continue
                children.append((path + [v], s + float(escore[e]), k))
                k += 1
        children.sort(key=lambda c: (-c[1] / (len(c[0]) - 1), c[2]))
        children = children[:W]
        frontier = [(p, s) for p, s, _ in children]
        cand += children
    cand.sort(key=lambda c: (-c[1] / (len(c[0]) - 1), c[2]))
    return [(s / (len(p) - 1), p) for p, s, _ in cand[:N]]


def rog_ground(adj, q_entities, rules, max_paths):
    """Step (1) of graph_utils.get_paths_rog_plan: ground each planned relation path from every topic entity."""
    t0 = time.perf_counter()
    paths, seen, arr = [], set(), []
    for rule in rules:
        for start in q_entities:
            for p in _bfs_with_rule(adj, start, list(rule)):
                key = tuple(p)
                if p and key not in seen:
                    seen.add(key)
                    paths.append(p)
                    arr.append(ms_since(t0))
        if len(paths) >= max_paths:
            return paths[:max_paths], arr[:max_paths]
    return paths, arr


def rog_adj(graph_triples):
    """The adjacency get_paths_rog_plan builds from the raw directed triples (both directions)."""
    from collections import defaultdict
    adj = defaultdict(list)
    for h, r, t in graph_triples:
        r = r.strip()
        adj[h].append((r, t))
        adj[t].append((r, h))
    return adj


def plan_rules(saved_paths):
    """Relation sequences of the saved RoG-plan paths, in first-appearance order."""
    rules, seen = [], set()
    for p in saved_paths:
        r = tuple(t[1] for t in p)
        if r not in seen:
            seen.add(r)
            rules.append(r)
    return rules


# ---------------------------------------------------------------- per-question work

G_ARGS = None
G_ITEMS = None


def covers(paths, a_set):
    return int(any(t[0] in a_set or t[2] in a_set for p in paths for t in p))


def nonsimple(paths):
    return sum(len({t[0] for t in p} | {p[-1][2]}) < len(p) + 1 for p in paths)


def arrival_stats(arr):
    a = list(arr)
    return dict(t_first=a[0] if a else None, t10=a[9] if len(a) >= 10 else None,
                t50=a[49] if len(a) >= 50 else None)


def work(i):
    a = G_ARGS
    item = G_ITEMS[i]
    s, ds, idx = item["s"], item["dataset"], item["idx"]
    q_ent, a_ent, a_set = s["q_entity"], s["a_entity"], set(s["a_entity"])
    rows = []
    base = dict(dataset=ds, idx=idx)
    t0 = time.perf_counter()
    G = build_graph(s["graph"])
    build_ms = ms_since(t0)
    csr = CSR(G)
    t0 = time.perf_counter()
    escore = csr.edge_scores(item["relscore"])
    escore_ms = ms_since(t0)
    graph_info = dict(nodes=G.number_of_nodes(), edges=G.number_of_edges(), build_ms=build_ms,
                      csr_ms=csr.build_ms, escore_ms=escore_ms)

    def row(hop, mode, paths, ms, **kw):
        r = dict(base, hop=hop, mode=mode, n_paths=len(paths), ms=ms, cover=covers(paths, a_set), **kw)
        if mode in a.keep_paths:
            r["paths"] = paths
        return r

    for hop in a.hops:
        t0 = time.perf_counter()
        ref = get_paths_bfs(q_ent, a_ent, G, hop, a.cap)
        rows.append(row(hop, "gold_py", ref, ms_since(t0), **graph_info))
        t0 = time.perf_counter()
        p, arr, c = bfs_ins(q_ent, a_ent, G, hop, a.cap)
        rows.append(row(hop, "gold_py_ins", p, ms_since(t0), match=int(p == ref), arrivals=arr, **c,
                        **arrival_stats(arr)))
        for prune in (0, 1):
            p, arr, core, ms, cnt = gold_cpp(csr, q_ent, a_ent, hop, a.cap, prune, a.reps)
            rows.append(row(hop, "gold_cpp_prune" if prune else "gold_cpp", p, ms, ms_core=core, ms_first=cnt[7] / 1000,
                            match=int(p == ref), states=int(cnt[0]), edges=int(cnt[1]), enq=int(cnt[2]),
                            dist_edges=int(cnt[3]), **arrival_stats(arr)))
        t0 = time.perf_counter()
        jref, _ = get_paths_optimized(q_ent, a_ent, G, hop, "cpu", a.cap)
        rows.append(row(hop, "jbidfs_py", jref, ms_since(t0), nonsimple=nonsimple(jref)))
        t0 = time.perf_counter()
        p, arr, enum = jbidfs_ins(q_ent, a_ent, G, hop, a.cap)
        st = enum.get_statistics()
        rows.append(row(hop, "jbidfs_ins", p, ms_since(t0), match=int(p == jref), calls=enum.calls,
                        edges=enum.scan, prune_ms=st["graph_pruning_time"] * 1000,
                        join_ms=st["join_selection_time"] * 1000, bidi_ms=st["bidirectional_dfs_time"] * 1000,
                        combine_ms=st["path_combination_time"] * 1000, indep_ms=enum.indep_time * 1000,
                        **arrival_stats(arr)))
        # (a) answer unknown
        t0 = time.perf_counter()
        pref, arr_py = open_cap_py(q_ent, G, hop, a.cap)
        ms_py = ms_since(t0)
        rows.append(row(hop, "open_cap_py", pref, ms_py, **arrival_stats(arr_py)))
        p, _, arr, core, ms, cnt = expand_cpp(csr, escore, q_ent, hop, 0, a.cap, a.beam, a.limit, a.reps)
        rows.append(row(hop, "open_cap", p, ms, ms_core=core, ms_first=cnt[7] / 1000, match=int(p == pref), states=int(cnt[0]),
                        edges=int(cnt[1]), **arrival_stats(arr)))
        p, sc, _, core, ms, cnt = expand_cpp(csr, escore, q_ent, hop, 1, a.cap, a.beam, a.limit, a.reps)
        match = None
        if not cnt[4] and cnt[0] <= a.check_top:
            r_ = open_top_py(csr, escore, q_ent, hop, a.cap)
            match = int([x[0] for x in r_] == sc and [csr.triples(x[1]) for x in r_] == p)
        rows.append(row(hop, "open_top", p, ms + escore_ms, ms_core=core, ms_first=cnt[7] / 1000, match=match, all_paths=int(cnt[0]),
                        edges=int(cnt[1]), truncated=int(cnt[4]), top_score=sc[:1]))
        p, sc, _, core, ms, cnt = expand_cpp(csr, escore, q_ent, hop, 2, a.cap, a.beam, a.limit, a.reps)
        r_ = open_beam_py(csr, escore, q_ent, hop, a.cap, a.beam)
        match = int([x[0] for x in r_] == sc and [csr.triples(x[1]) for x in r_] == p)
        rows.append(row(hop, "open_beam", p, ms + escore_ms, ms_core=core, ms_first=cnt[7] / 1000, match=match, states=int(cnt[0]),
                        edges=int(cnt[1])))
    # (b) RoG plan grounding, hop <= 3 (the plans were made with hop 3)
    saved = item["plan_paths"]
    if saved is not None:
        rules = plan_rules(saved)
        t0 = time.perf_counter()
        adj = rog_adj(s["graph"])
        adj_ms = ms_since(t0)
        saved_set = {tuple(tuple(t) for t in p) for p in saved}
        for cap in (30, a.cap):
            t0 = time.perf_counter()
            p, arr = rog_ground(adj, q_ent, rules, cap)
            ms = ms_since(t0)
            got = {tuple(tuple(t) for t in x) for x in p}
            rows.append(row(3, f"rog_plan_cap{cap}", p, ms, n_rules=len(rules), adj_ms=adj_ms,
                            saved_n=len(saved), saved_recall=len(got & saved_set) / max(1, len(saved_set)),
                            **arrival_stats(arr)))
    return rows


def cmd_run(a):
    global G_ARGS, G_ITEMS
    G_ARGS = a
    lib()                                   # compile once before forking
    rel = json.load(open(os.path.join(OUT_DIR, "relscore.json")))
    items = []
    for name in DATASETS:
        sub = load_subset(name, a.n)
        tag = name.replace("/", "-")
        plans = json.load(open(os.path.join(PLAN_DIR, f"paths_{tag}.json")))
        k0 = {}
        k0f = os.path.join(ROOT, "outputs", "kv_prefetch", a.k0_timeline)
        if os.path.exists(k0f):
            for line in open(k0f):
                r = json.loads(line)
                if r["dataset"] == name:
                    k0[r["idx"]] = r["question"]
        for i, s in enumerate(sub):
            assert not k0 or k0.get(i, s["question"]) == s["question"], (name, i)
            items.append(dict(dataset=name, idx=i, s=s, relscore=rel[name][str(i)],
                              plan_paths=plans.get(s["question"])))
    if a.limit_items:
        items = items[:a.limit_items]
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"profile_{a.tag}.jsonl")
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    G_ITEMS = items
    t0 = time.time()
    with open(out, "w") as f, ctx.Pool(a.workers) as pool:
        for k, rows in enumerate(pool.imap_unordered(work, range(len(items)))):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            if (k + 1) % 50 == 0:
                print(f"{k + 1}/{len(items)} questions, {time.time() - t0:.0f} s", flush=True)
    meta = dict(host=platform.node(), machine=platform.machine(), processor=platform.processor(),
                python=sys.version.split()[0], args=vars(a), wall_s=time.time() - t0)
    json.dump(meta, open(out.replace(".jsonl", "_meta.json"), "w"), indent=1)
    print("wrote", out)


def cmd_relscore(a):
    """MiniLM cosine(question, relation) for every relation in each question's subgraph."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    enc_name = "sentence-transformers/all-MiniLM-L6-v2"
    tok, model = AutoTokenizer.from_pretrained(enc_name), AutoModel.from_pretrained(enc_name).eval()
    torch.set_num_threads(a.threads)

    @torch.no_grad()
    def embed(texts, bs=256):
        out = []
        for i in range(0, len(texts), bs):
            x = tok(texts[i:i + bs], padding=True, truncation=True, max_length=64, return_tensors="pt")
            h = model(**x).last_hidden_state
            m = x["attention_mask"].unsqueeze(-1).float()
            out.append(torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1))
        return torch.cat(out).numpy()

    res, cache = {}, {}
    questions = []
    for name in DATASETS:
        sub = load_subset(name, a.n)
        res[name] = {}
        rels_all = sorted({r.strip() for s in sub for _, r, _ in s["graph"]} - set(cache))
        for r, v in zip(rels_all, embed([r.replace(".", " ").replace("_", " ") for r in rels_all])):
            cache[r] = v
        qv = embed([s["question"] for s in sub])
        for i, s in enumerate(sub):
            rels = sorted({r.strip() for _, r, _ in s["graph"]})
            res[name][str(i)] = {r: float(cache[r] @ qv[i]) for r in rels}
            questions.append(s["question"])
    # one question embedding at batch 1 on CPU: the encoder cost a real (a)-style retriever would add
    lat = []
    for q in questions[:60]:
        t0 = time.perf_counter()
        embed([q])
        lat.append(ms_since(t0))
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(res, open(os.path.join(OUT_DIR, "relscore.json"), "w"))
    meta = dict(model=enc_name, threads=a.threads, qembed_ms_median=float(np.median(lat[10:])),
                qembed_ms_p90=float(np.percentile(lat[10:], 90)), n_relations=len(cache), host=platform.node())
    json.dump(meta, open(os.path.join(OUT_DIR, "relscore_meta.json"), "w"), indent=1)
    print(meta)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("relscore")
    r.add_argument("--n", type=int, default=200)
    r.add_argument("--threads", type=int, default=4)
    r = sub.add_parser("run")
    r.add_argument("--n", type=int, default=200)
    r.add_argument("--hops", type=int, nargs="+", default=[3, 4])
    r.add_argument("--cap", type=int, default=50)
    r.add_argument("--beam", type=int, default=50)
    r.add_argument("--limit", type=int, default=50_000_000, help="open_top stops after this many paths")
    r.add_argument("--check_top", type=int, default=200_000, help="check open_top against Python up to this many paths")
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--workers", type=int, default=6)
    r.add_argument("--tag", required=True)
    r.add_argument("--limit_items", type=int, default=0)
    r.add_argument("--k0_timeline", default="retrieval_timeline.jsonl",
                   help="K0 timeline under outputs/kv_prefetch/ used to check the question subset")
    r.add_argument("--keep_paths", nargs="+", default=["gold_py", "open_cap", "open_top", "open_beam",
                                                       "rog_plan_cap30", "rog_plan_cap50"])
    a = ap.parse_args()
    {"relscore": cmd_relscore, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
