"""
Graph construction, path serialization, and path-finding algorithms.

Exports:
  build_graph          – build a NetworkX graph from triplet data
  path_to_str          – convert a path (list of (h,r,t)) to a dot-joined string
  get_paths_bfs        – simple BFS retrieval
  get_paths_optimized  – Algorithm-1 optimized retrieval (via OptimizedPathEnumerator)
  OptimizedPathEnumerator – the full enumeration class
"""

import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import networkx as nx
import torch


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(graph_data: list) -> nx.Graph:
    """Build a NetworkX graph from a list of (head, relation, tail) triplets."""
    G = nx.Graph()
    for triplet in graph_data:
        h, r, t = triplet
        G.add_edge(h, t, relation=r.strip())
    return G


# ---------------------------------------------------------------------------
# Path serialization
# ---------------------------------------------------------------------------

def path_to_str(path: List[Tuple[str, str, str]]) -> str:
    """Convert a path (list of (h, r, t) tuples) to a dot-joined readable string."""
    if not path:
        return ""
    parts = [path[0][0]]  # start entity
    for (u, rel, v) in path:
        parts.append(rel)
        parts.append(v)
    return ".".join(parts)


# ---------------------------------------------------------------------------
# Simple BFS retrieval
# ---------------------------------------------------------------------------

def get_paths_bfs(
    q_entities: list,
    a_entities: list,
    graph: nx.Graph,
    hop_constraint: int = 2,
    max_paths_per_query: int = None,
) -> list:
    """
    Simple BFS to find paths from question entities to answer entities
    within the hop constraint.  Returns paths as lists of (h, r, t) tuples.
    """
    result_paths = []

    for start in q_entities:
        if start not in graph:
            continue

        # BFS queue: (current_node, path_nodes, visited_set)
        queue = deque([(start, [start], {start})])

        while queue:
            current_node, path_nodes, visited = queue.popleft()

            if current_node in a_entities and 1 <= len(path_nodes) - 1 <= hop_constraint:
                path_with_rel = []
                for i in range(len(path_nodes) - 1):
                    u = path_nodes[i]
                    v = path_nodes[i + 1]
                    rel = graph[u][v]["relation"]
                    path_with_rel.append((u, rel, v))
                result_paths.append(path_with_rel)

                path_limit = max_paths_per_query if max_paths_per_query else 50
                if len(result_paths) >= path_limit:
                    return result_paths
                continue

            if len(path_nodes) - 1 < hop_constraint:
                for neighbor in graph.neighbors(current_node):
                    if neighbor not in visited:
                        queue.append((
                            neighbor,
                            path_nodes + [neighbor],
                            visited | {neighbor},
                        ))

    return result_paths


# ---------------------------------------------------------------------------
# Algorithm-1 optimized path enumeration
# ---------------------------------------------------------------------------

def get_paths_optimized(
    q_entities: list,
    a_entities: list,
    graph: nx.Graph,
    hop_constraint: int = 3,
    device: str = "cuda",
    max_paths_per_query: int = None,
) -> Tuple[list, dict]:
    """
    Optimized path enumeration using Algorithm 1 from the paper.
    Returns (paths, statistics_dict).
    """
    enumerator = OptimizedPathEnumerator(device)
    result_paths = []

    for start in q_entities:
        for end in a_entities:
            if start != end:
                paths = enumerator.enumerate_paths(graph, start, end, hop_constraint, max_paths=50)
                result_paths.extend(paths)

                path_limit = max_paths_per_query if max_paths_per_query else 100
                if len(result_paths) >= path_limit:
                    return result_paths[:path_limit], enumerator.get_statistics()

    return result_paths, enumerator.get_statistics()


# ---------------------------------------------------------------------------
# Optimized Path Enumerator (Algorithm 1)
# ---------------------------------------------------------------------------

class OptimizedPathEnumerator:
    """
    Optimized Path Enumeration implementing Algorithm 1 from the paper
    with torch integration for GPU acceleration where applicable.
    """

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.stats: Dict = {
            "graph_pruning_time": 0.0,
            "join_selection_time": 0.0,
            "bidirectional_dfs_time": 0.0,
            "path_combination_time": 0.0,
            "nodes_pruned": 0,
            "paths_found": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        self.distance_cache: Dict = {}
        self.path_cache: Dict = {}

    # ------------------------------------------------------------------
    # Step 1: Graph pruning
    # ------------------------------------------------------------------

    def graph_prune(self, graph: nx.Graph, source: str, target: str, hop_constraint: int) -> Dict:
        """Lightweight graph pruning using BFS; creates candidate node sets."""
        start_time = time.time()

        def bfs_distances(g, start_node, max_hops):
            distances = {start_node: 0}
            queue = deque([(start_node, 0)])
            while queue:
                node, dist = queue.popleft()
                if dist < max_hops:
                    for neighbor in g.neighbors(node):
                        if neighbor not in distances:
                            distances[neighbor] = dist + 1
                            queue.append((neighbor, dist + 1))
            return distances

        source_distances = bfs_distances(graph, source, hop_constraint)
        target_distances = bfs_distances(graph, target, hop_constraint)

        all_nodes = set(graph.nodes())
        candidate_nodes = {
            n for n in all_nodes
            if source_distances.get(n, float("inf")) + target_distances.get(n, float("inf")) <= hop_constraint
        }

        pruned_graph = graph.subgraph(candidate_nodes).copy()
        self.stats["graph_pruning_time"] += time.time() - start_time
        self.stats["nodes_pruned"] += len(all_nodes) - len(candidate_nodes)

        return {
            "graph": pruned_graph,
            "source_distances": source_distances,
            "target_distances": target_distances,
            "candidate_nodes": candidate_nodes,
            "pruning_ratio": len(candidate_nodes) / len(all_nodes) if all_nodes else 0,
        }

    # ------------------------------------------------------------------
    # Step 2: Join-node selection
    # ------------------------------------------------------------------

    def select_join_node(self, graph: nx.Graph, source: str, target: str) -> Optional[str]:
        """Select the midpoint of the shortest path as the join node."""
        start_time = time.time()
        join_node = None
        try:
            shortest = nx.shortest_path(graph, source, target)
            if len(shortest) > 1:
                join_node = shortest[len(shortest) // 2]
        except nx.NetworkXNoPath:
            pass
        self.stats["join_selection_time"] += time.time() - start_time
        return join_node

    # ------------------------------------------------------------------
    # Step 3: Bidirectional DFS
    # ------------------------------------------------------------------

    def bidirectional_dfs(
        self,
        graph: nx.Graph,
        source: str,
        target: str,
        join_node: str,
        hop_constraint: int,
        max_paths: int = 50,
    ) -> Tuple[List, List]:
        start_time = time.time()

        def dfs_paths(g, start, end, max_hops, current_path, all_paths, visited):
            if len(current_path) - 1 > max_hops:
                return
            if start == end and len(current_path) > 1:
                rel_path = []
                for i in range(len(current_path) - 1):
                    u, v = current_path[i], current_path[i + 1]
                    if g.has_edge(u, v):
                        rel = g[u][v].get("relation", "related_to")
                        rel_path.append((u, rel, v))
                if rel_path:
                    all_paths.append(rel_path)
                return
            if len(all_paths) >= max_paths:
                return
            for neighbor in g.neighbors(start):
                if neighbor not in visited:
                    dfs_paths(g, neighbor, end, max_hops,
                              current_path + [neighbor], all_paths, visited | {neighbor})

        source_to_join: List = []
        if join_node != source:
            dfs_paths(graph, source, join_node, hop_constraint, [source], source_to_join, {source})

        join_to_target: List = []
        if join_node != target:
            dfs_paths(graph, join_node, target, hop_constraint, [join_node], join_to_target, {join_node})

        self.stats["bidirectional_dfs_time"] += time.time() - start_time
        return source_to_join, join_to_target

    # ------------------------------------------------------------------
    # Step 4: Path combination
    # ------------------------------------------------------------------

    def combine_paths(self, source_paths: List, target_paths: List, hop_constraint: int) -> List:
        start_time = time.time()
        combined = [
            s + t
            for s in source_paths
            for t in target_paths
            if len(s) + len(t) <= hop_constraint
        ]
        self.stats["path_combination_time"] += time.time() - start_time
        return combined

    # ------------------------------------------------------------------
    # Step 5: Independent DFS (fallback / supplement)
    # ------------------------------------------------------------------

    def independent_dfs(
        self,
        graph: nx.Graph,
        source: str,
        target: str,
        hop_constraint: int,
        max_paths: int = 50,
    ) -> List:
        def dfs_all(g, start, end, max_hops, current_path, visited):
            if len(current_path) - 1 > max_hops:
                return []
            if start == end and len(current_path) > 1:
                rel_path = []
                for i in range(len(current_path) - 1):
                    u, v = current_path[i], current_path[i + 1]
                    if g.has_edge(u, v):
                        rel = g[u][v].get("relation", "related_to")
                        rel_path.append((u, rel, v))
                return [rel_path] if rel_path else []
            all_paths: List = []
            for neighbor in g.neighbors(start):
                if neighbor not in visited:
                    all_paths.extend(dfs_all(g, neighbor, end, max_hops,
                                             current_path + [neighbor], visited | {neighbor}))
                    if len(all_paths) >= max_paths:
                        break
            return all_paths

        return dfs_all(graph, source, target, hop_constraint, [source], {source})

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def enumerate_paths(
        self,
        graph: nx.Graph,
        source: str,
        target: str,
        hop_constraint: int = 3,
        max_paths: int = 100,
    ) -> List:
        """Main path enumeration function (Algorithm 1)."""
        if source not in graph or target not in graph or source == target:
            return []

        cache_key = (source, target, hop_constraint)
        if cache_key in self.path_cache:
            self.stats["cache_hits"] += 1
            return self.path_cache[cache_key]

        self.stats["cache_misses"] += 1

        pruning_result = self.graph_prune(graph, source, target, hop_constraint)
        pruned_graph = pruning_result["graph"]

        if target not in pruning_result["candidate_nodes"]:
            self.path_cache[cache_key] = []
            return []

        join_node = self.select_join_node(pruned_graph, source, target)
        final_paths: List = []

        if join_node and join_node != source and join_node != target:
            src_paths, tgt_paths = self.bidirectional_dfs(
                pruned_graph, source, target, join_node, hop_constraint, max_paths
            )
            if src_paths and tgt_paths:
                final_paths.extend(self.combine_paths(src_paths, tgt_paths, hop_constraint))

        if len(final_paths) < max_paths:
            independent = self.independent_dfs(
                pruned_graph, source, target, hop_constraint, max_paths - len(final_paths)
            )
            seen = {str(p) for p in final_paths}
            for path in independent:
                if str(path) not in seen:
                    final_paths.append(path)
                    seen.add(str(path))
                if len(final_paths) >= max_paths:
                    break

        self.stats["paths_found"] += len(final_paths)
        self.path_cache[cache_key] = final_paths
        return final_paths

    def get_statistics(self) -> Dict:
        return self.stats.copy()

    def reset_statistics(self) -> None:
        for key in self.stats:
            self.stats[key] = 0.0 if isinstance(self.stats[key], float) else 0


# ---------------------------------------------------------------------------
# Dense, no-leakage retrieval (literature-standard)
# ---------------------------------------------------------------------------
# Ranks paths emanating from the QUESTION entities by semantic similarity of
# their relation sequence to the question (MiniLM cosine), WITHOUT using the
# answer entity. This mirrors RoG/GNN-RAG relation-path retrieval (which use a
# learned planner) with an unsupervised dense proxy, so numbers are comparable
# to the literature and there is no gold-answer leakage.

_EMBEDDER = None


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev)
    return _EMBEDDER


def get_paths_dense(q_entities, graph, question, hop_constraint: int = 3,
                    max_paths: int = 8, max_cand: int = 400):
    """Top-`max_paths` shortest paths from question entities, ranked by
    question-vs-relation similarity. No answer entity is used."""
    from collections import deque
    import numpy as np

    cand_triples, cand_text, seen = [], [], set()
    for start in q_entities:
        if start not in graph:
            continue
        queue = deque([(start, [start])])
        visited = {start}
        while queue and len(cand_triples) < max_cand:
            node, pnodes = queue.popleft()
            depth = len(pnodes) - 1
            if depth >= 1 and node not in seen:
                seen.add(node)
                triples = [(pnodes[i], graph[pnodes[i]][pnodes[i + 1]]["relation"], pnodes[i + 1])
                           for i in range(len(pnodes) - 1)]
                cand_triples.append(triples)
                cand_text.append(" ".join(t[1] for t in triples).replace(".", " "))
            if depth < hop_constraint:
                for nb in graph.neighbors(node):
                    if nb not in visited:
                        visited.add(nb)
                        queue.append((nb, pnodes + [nb]))
    if not cand_triples:
        return []
    if len(cand_triples) <= max_paths:
        return cand_triples

    emb = _get_embedder()
    qv = emb.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]
    pv = emb.encode(cand_text, convert_to_numpy=True, normalize_embeddings=True, batch_size=256)
    sims = pv @ qv
    top = np.argsort(-sims)[:max_paths]
    return [cand_triples[i] for i in top]


def get_paths_rog(q_entities, graph, draft_answer, hop_constraint: int = 3,
                  max_paths: int = 8):
    """RoG-style retrieval: targets are entities mentioned in a draft LLM answer
    (NOT the gold answer), then BFS paths from question entities to those targets.
    Answer-aware but leakage-free; matches RoG's 'question + initial answer' method."""
    import re as _re
    import string as _string

    def _norm(s):
        s = str(s).lower().replace("<pad>", " ")
        s = "".join(c for c in s if c not in _string.punctuation)
        return " ".join(s.split())

    dl = _norm(draft_answer)
    targets = [n for n in graph.nodes
               if n not in q_entities and len(_norm(n)) >= 3 and _norm(n) in dl]
    if not targets:
        return []
    return get_paths_bfs(q_entities, targets, graph, hop_constraint, max_paths)


def _greedy_parse_rule(gen_text, rel_vocab_sorted, max_len):
    """Parse a (possibly concatenated / extra-prefixed) generated relation string
    into an ordered rule by locating KG relations as SUBSTRINGS, in order of
    appearance, non-overlapping (longest match wins). Robust to the RoG model
    emitting relations with doubled domain prefixes."""
    occ = []  # (position, relation)
    for r in rel_vocab_sorted:           # longest first
        start = 0
        while True:
            idx = gen_text.find(r, start)
            if idx < 0:
                break
            occ.append((idx, idx + len(r), r))
            start = idx + 1
    occ.sort(key=lambda x: (x[0], -x[1]))   # by position; longest match first at ties
    rule, last_end = [], -1
    for s, e, r in occ:
        if s >= last_end and len(rule) < max_len:
            rule.append(r)
            last_end = e
    return rule


def _bfs_with_rule(adj, start, rule):
    """RoG's exact grounding: follow the relation sequence `rule` from `start`."""
    from collections import deque
    res, queue = [], deque([(start, [])])
    while queue:
        node, path = queue.popleft()
        if len(path) == len(rule):
            res.append(path)
            continue
        for rel, nb in adj.get(node, ()):
            if rel == rule[len(path)]:
                queue.append((nb, path + [(node, rel, nb)]))
    return res


def get_paths_rog_plan(planner, tok, q_entities, graph_triples, question,
                       hop_constraint: int = 3, max_paths: int = 8, num_beam: int = 3):
    """Faithful RoG retrieval: the fine-tuned RoG model plans relation paths; each
    is greedy-parsed into a relation sequence and grounded with bfs_with_rule
    (RoG's exact instantiation). No answer entity used. `graph_triples` is the raw
    directed (h, r, t) list. `planner` is the RoG model (retrieval is fixed)."""
    import torch
    from collections import defaultdict
    adj, rels = defaultdict(list), set()
    for h, r, t in graph_triples:
        r = r.strip()
        adj[h].append((r, t))
        adj[t].append((r, h))   # allow traversing relations from either endpoint
        rels.add(r)
    rel_sorted = sorted(rels, key=len, reverse=True)

    INSTR = "Please generate a valid relation path that can be helpful for answering the following question: "
    ids = tok(f"<s>[INST] {INSTR}{question} [/INST]", return_tensors="pt").to(planner.device)
    with torch.no_grad():
        out = planner.generate(**ids, max_new_tokens=80, num_beams=num_beam,
                               num_return_sequences=num_beam, do_sample=False,
                               pad_token_id=tok.eos_token_id)
    rules = []
    for j in range(out.shape[0]):
        gen = tok.decode(out[j, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        rule = _greedy_parse_rule(gen, rel_sorted, hop_constraint)
        if rule:
            rules.append(rule)

    paths, seen = [], set()

    def _add(p):
        key = tuple(p)
        if p and key not in seen:
            seen.add(key)
            paths.append(p)

    # (1) ground full planned rules
    for rule in rules:
        for start in q_entities:
            for p in _bfs_with_rule(adj, start, rule):
                _add(p)
        if len(paths) >= max_paths:
            return paths[:max_paths]

    # (2) fallback: ground rule prefixes (shorter paths) when the full rule
    #     didn't reach far enough in this subgraph
    if len(paths) < max_paths:
        for rule in rules:
            for L in range(min(len(rule), hop_constraint) - 1, 0, -1):
                for start in q_entities:
                    for p in _bfs_with_rule(adj, start, rule[:L]):
                        _add(p)
            if len(paths) >= max_paths:
                return paths[:max_paths]

    # (3) fallback: topic-entity neighborhood (BFS up to hop, any relation) when
    #     no planned relation grounds at all (RoG-style backoff to local context)
    if not paths:
        from collections import deque
        for start in q_entities:
            if start not in adj:
                continue
            queue = deque([(start, [])])
            visited = {start}
            while queue and len(paths) < max_paths:
                node, path = queue.popleft()
                if path:
                    _add(path)
                if len(path) < hop_constraint:
                    for rel, nb in adj.get(node, ()):
                        if nb not in visited:
                            visited.add(nb)
                            queue.append((nb, path + [(node, rel, nb)]))
    return paths[:max_paths]
