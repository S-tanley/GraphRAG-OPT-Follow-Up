"""
Graph-path retrieval classes.

SimpleGraphRAGRetrieval  – plain BFS (baseline)
BatchOptimizedRetrieval  – Algorithm-1 enumeration + optional batch deduplication
"""

import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from graph_utils import (
    OptimizedPathEnumerator,
    build_graph,
    get_paths_bfs,
    get_paths_optimized,
    path_to_str,
)


# ---------------------------------------------------------------------------
# Baseline BFS retrieval
# ---------------------------------------------------------------------------

class SimpleGraphRAGRetrieval:
    """Simplified GraphRAG retrieval using plain BFS (no optimizations)."""

    def __init__(self, dataset):
        self.dataset = dataset
        print("SimpleGraphRAGRetrieval initialized")

    def retrieve_batch(
        self,
        hop_constraint: int = 3,
        max_paths_per_query: Optional[int] = None,
    ) -> Tuple[List[Dict], Dict]:
        t0 = time.time()
        results = []
        stats = {"processing_time": 0, "total_paths": 0, "total_queries": len(self.dataset)}

        if max_paths_per_query:
            print(f"[Path Control] Limiting to {max_paths_per_query} paths per query")

        for query in self.dataset:
            graph = build_graph(query["graph"])
            paths = get_paths_bfs(
                query["q_entity"],
                query["a_entity"],
                graph,
                hop_constraint,
                max_paths_per_query,
            )
            results.append({"query": query["question"], "paths": paths})
            stats["total_paths"] += len(paths)

        stats["processing_time"] = time.time() - t0
        return results, stats


# ---------------------------------------------------------------------------
# Optimized batch retrieval
# ---------------------------------------------------------------------------

class BatchOptimizedRetrieval:
    """
    Batch-optimized retrieval implementing the paper's batch query optimization.
    Uses Algorithm-1 path enumeration and deduplicates repeated entity pairs
    across queries so each (source, target) pair is searched only once.
    """

    def __init__(self, dataset, device: str = "cuda"):
        self.dataset = dataset
        self.device = device
        self.enumerator = OptimizedPathEnumerator(device)
        print("BatchOptimizedRetrieval initialized")

    def retrieve_batch(
        self,
        hop_constraint: int = 3,
        enable_batch_opt: bool = True,
        max_paths_per_query: Optional[int] = None,
    ) -> Tuple[List[Dict], Dict]:
        if not enable_batch_opt:
            return self._simple_retrieve_batch(hop_constraint, max_paths_per_query)

        print(f"[Batch Optimization] Processing {len(self.dataset)} queries...")
        if max_paths_per_query:
            print(f"[Path Control] Limiting to {max_paths_per_query} paths per query")

        t0 = time.time()

        # Collect unique (source, target) entity pairs and their graphs
        pair_to_graph: Dict = {}
        pair_to_indices: Dict[Tuple, List[int]] = defaultdict(list)

        for idx, query in enumerate(self.dataset):
            graph = build_graph(query["graph"])
            for src in query["q_entity"]:
                for tgt in query["a_entity"]:
                    if src != tgt:
                        pair = (src, tgt)
                        pair_to_graph[pair] = graph
                        pair_to_indices[pair].append(idx)

        total_pairs = len(pair_to_graph)
        unique_pairs = len(pair_to_graph)
        pair_redundancy = 1 - unique_pairs / total_pairs if total_pairs > 0 else 0

        print(f"[Batch Opt] {total_pairs} pairs → {unique_pairs} unique "
              f"(redundancy {pair_redundancy:.1%})")

        # Compute paths once per unique pair
        pair_to_paths: Dict[Tuple, List] = {}
        for pair, graph in pair_to_graph.items():
            src, tgt = pair
            paths = self.enumerator.enumerate_paths(
                graph, src, tgt, hop_constraint, max_paths=50
            )
            pair_to_paths[pair] = paths

        # Assemble per-query results
        query_paths: Dict[int, List] = defaultdict(list)
        for pair, paths in pair_to_paths.items():
            for idx in pair_to_indices[pair]:
                query_paths[idx].extend(paths)

        results = []
        for idx, query in enumerate(self.dataset):
            paths = query_paths[idx]
            if max_paths_per_query:
                paths = paths[:max_paths_per_query]
            results.append({"query": query["question"], "paths": paths})

        all_paths_count = sum(len(r["paths"]) for r in results)
        path_redundancy = 1 - unique_pairs / all_paths_count if all_paths_count > 0 else 0

        stats = self.enumerator.get_statistics()
        stats.update({
            "total_pairs": total_pairs,
            "unique_pairs": unique_pairs,
            "pair_redundancy_ratio": pair_redundancy,
            "path_redundancy_ratio": max(0.0, path_redundancy),
            "efficiency_gain": total_pairs / unique_pairs if unique_pairs > 0 else 1.0,
            "processing_time": time.time() - t0,
        })
        return results, stats

    def _simple_retrieve_batch(
        self,
        hop_constraint: int,
        max_paths_per_query: Optional[int],
    ) -> Tuple[List[Dict], Dict]:
        """Fallback: per-query optimized enumeration without batch deduplication."""
        t0 = time.time()
        results = []
        for query in self.dataset:
            graph = build_graph(query["graph"])
            paths, _ = get_paths_optimized(
                query["q_entity"],
                query["a_entity"],
                graph,
                hop_constraint,
                self.device,
                max_paths_per_query,
            )
            results.append({"query": query["question"], "paths": paths})
        stats = self.enumerator.get_statistics()
        stats["processing_time"] = time.time() - t0
        return results, stats
