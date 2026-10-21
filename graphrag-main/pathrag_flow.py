"""
pathrag_flow.py — faithful port of PathRAG's flow-based path pruning.

Ported DIRECTLY from the official repo BUPT-GAMMA/PathRAG
(PathRAG/operate.py, commit 32567bf), functions:
  - find_paths_and_edges_with_stats  (brute-force <=3-hop DFS between seed pairs)
  - bfs_weighted_paths               (resource-decay flow weighting; alpha=0.8,
                                      threshold=0.3; path score = avg edge weight)
  - _find_most_related_edges_from_entities3  (sort by weight, dedup, cap 15)

We isolate PathRAG's RETRIEVAL contribution — the flow-based relevance pruning —
as a Sel() rule in our (F, k, Sel) formulation, so it can be applied to the SAME
candidate path set a KGQA enumerator (exhaustive DFS or our JbiDFS) produces over
a given knowledge graph. We do NOT reimplement PathRAG's text-corpus INDEXING
stage (LLM entity/relation extraction with NL descriptions/keywords): on a *given*
KG (Freebase/WikiMovies) there is nothing to construct, so that stage is
inapplicable. The flow-weighting math below is reproduced verbatim from source;
only the WEIGHT SOURCE differs (Freebase relation IDs instead of LLM-written edge
keywords), which we disclose and ablate.

Constants are PathRAG's own (operate.py:1124-1125, 1160).
"""
from collections import defaultdict

ALPHA = 0.8        # PathRAG operate.py:1125 — resource-flow decay
THRESHOLD = 0.3    # PathRAG operate.py:1124 — flow expansion threshold
TOTAL_EDGES = 15   # PathRAG operate.py:1160 — final path/edge cap


def _bfs_weighted_paths(paths, source, target, threshold=THRESHOLD, alpha=ALPHA):
    """Verbatim port of PathRAG.operate.bfs_weighted_paths.

    `paths` is the enumerated set of node-paths (lists of node ids) from `source`
    to `target`. Returns [(path, weight)] where weight is the avg edge flow-weight
    along the path. (Edge weights accumulate resource flow with alpha decay,
    distributed uniformly over a node's followers, gated by `threshold`.)"""
    results = []
    edge_weights = defaultdict(float)
    node = source
    follow_dict = {}
    for p in paths:
        for i in range(len(p) - 1):
            cur, nxt = p[i], p[i + 1]
            follow_dict.setdefault(cur, set()).add(nxt)

    if node not in follow_dict:
        # no outgoing transitions from source in this path set
        return [(p, 0.0) for p in paths]

    for neighbor in follow_dict[node]:
        edge_weights[(node, neighbor)] += 1 / len(follow_dict[node])
        if neighbor == target:
            results.append([node, neighbor]); continue
        if edge_weights[(node, neighbor)] > threshold and neighbor in follow_dict:
            for second in follow_dict[neighbor]:
                w = edge_weights[(node, neighbor)] * alpha / len(follow_dict[neighbor])
                edge_weights[(neighbor, second)] += w
                if second == target:
                    results.append([node, neighbor, second]); continue
                if edge_weights[(neighbor, second)] > threshold and second in follow_dict:
                    for third in follow_dict[second]:
                        w2 = edge_weights[(neighbor, second)] * alpha / len(follow_dict[second])
                        edge_weights[(second, third)] += w2
                        if third == target:
                            results.append([node, neighbor, second, third]); continue

    path_weights = []
    for p in paths:
        pw = 0.0
        for i in range(len(p) - 1):
            pw += edge_weights.get((p[i], p[i + 1]), 0.0)
        path_weights.append(pw / (len(p) - 1) if len(p) > 1 else 0.0)
    return [(p, w) for p, w in zip(paths, path_weights)]


def pathrag_flow_select(rel_paths, max_select=TOTAL_EDGES):
    """Apply PathRAG flow-pruning as a Sel() rule over a candidate path set.

    rel_paths : list of paths, each a list of (h, rel, t) triples (our path unit).
    Returns the top-`max_select` paths ranked by PathRAG's flow weight (their
    cap is 15), preserving PathRAG's sort-by-weight + dedup-by-edge logic.
    """
    if not rel_paths:
        return []
    # Convert (h,rel,t) triple-paths -> node-paths, grouped by (source, target).
    # The source is the first node of every path (the query/topic entity); the
    # target is the last node. PathRAG scores per (source,target) bucket.
    buckets = defaultdict(list)               # (s,t) -> list[node-path]
    nodepath_to_relpath = {}
    for rp in rel_paths:
        if not rp:
            continue
        nodes = [rp[0][0]] + [t for (_h, _r, t) in rp]
        s, t = nodes[0], nodes[-1]
        buckets[(s, t)].append(nodes)
        nodepath_to_relpath[tuple(nodes)] = rp

    scored = []
    for (s, t), npaths in buckets.items():
        for nodes, w in _bfs_weighted_paths(npaths, s, t):
            scored.append((nodepath_to_relpath.get(tuple(nodes)), w))
    # PathRAG: sort by weight desc, dedup, cap
    scored.sort(key=lambda x: x[1], reverse=True)
    seen, out = set(), []
    for rp, w in scored:
        if rp is None:
            continue
        key = tuple((h, r, t) for (h, r, t) in rp)
        if key in seen:
            continue
        seen.add(key)
        out.append(rp)
        if len(out) >= max_select:
            break
    return out
