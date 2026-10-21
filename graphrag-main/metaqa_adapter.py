"""
MetaQA Adapter for GraphRAG-KV
================================
Converts MetaQA (1-hop, 2-hop, 3-hop) into the RoG-compatible dict format
used throughout this project:

    {
        "question":  str,
        "answer":    List[str],
        "q_entity":  List[str],   # seed entity from [brackets] in the question
        "a_entity":  List[str],   # same as answer (list of answer entities)
        "graph":     List[[head, relation, tail]]   # local subgraph triplets
    }

Data layout (download once, then point --metaqa_dir here)
----------------------------------------------------------
  <metaqa_dir>/
      kb.txt               # WikiMovies KG: subject|relation|object  (one per line)
      1-hop/
          qa_train.txt     # question<TAB>answer1|answer2|...
          qa_dev.txt
          qa_test.txt
      2-hop/ ...
      3-hop/ ...

Download
--------
  Official repo: https://github.com/yuyuz/MetaQA
  The "MetaQA_KGC.zip" release (or Google Drive link in that README) contains
  kb.txt and all QA splits. Unzip so that kb.txt lives at the root of metaqa_dir.

Quick usage
-----------
  from metaqa_adapter import load_metaqa
  ds = load_metaqa(hops=2, split="test", data_dir="data/metaqa")
  # ds is a HuggingFace Dataset with columns: question, answer, q_entity, a_entity, graph
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import networkx as nx
from datasets import Dataset


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_ENTITY_RE = re.compile(r"\[([^\]]+)\]")


def _parse_kb(kb_path: Path) -> nx.Graph:
    """Build an undirected NetworkX graph from MetaQA kb.txt.

    Each line:  subject|relation|object
    """
    G = nx.Graph()
    with kb_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) == 3:
                h, r, t = parts[0].strip(), parts[1].strip(), parts[2].strip()
                G.add_edge(h, t, relation=r)
    return G


def _parse_qa(qa_path: Path) -> list[dict]:
    """Parse a MetaQA QA .txt file.

    Each line:  raw_question<TAB>answer1|answer2|...
    Entities inside the question are wrapped in [square brackets].
    """
    samples: list[dict] = []
    with qa_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) < 2:
                continue
            raw_q   = parts[0]
            answers = [a.strip() for a in parts[1].split("|") if a.strip()]
            q_ents  = _ENTITY_RE.findall(raw_q)
            # Remove brackets but keep entity text in the displayed question
            clean_q = _ENTITY_RE.sub(lambda m: m.group(1), raw_q)
            samples.append({
                "question": clean_q,
                "answer":   answers,
                "q_entity": q_ents,
                "a_entity": answers,
            })
    return samples


def _extract_subgraph(
    kg: nx.Graph,
    seeds: list[str],
    hops: int,
    max_nodes: int = 500,
) -> list[list[str]]:
    """BFS-expand from `seeds` for `hops` steps and return the induced triplets.

    max_nodes is enforced continuously during expansion (not just at hop
    boundaries) so hub entities never blow up the subgraph size.
    """
    visited: set[str] = {s for s in seeds if s in kg}
    frontier = list(visited)

    for _ in range(hops):
        nxt: list[str] = []
        for node in frontier:
            for nb in kg.neighbors(node):
                if nb not in visited:
                    visited.add(nb)
                    nxt.append(nb)
                    if len(visited) >= max_nodes:
                        break
            if len(visited) >= max_nodes:
                break
        frontier = nxt
        if not frontier or len(visited) >= max_nodes:
            break

    # Collect all edges whose both endpoints are inside the visited set
    triplets: list[list[str]] = []
    seen: set[tuple] = set()
    for u in visited:
        for v in kg.neighbors(u):
            if v in visited:
                key = (min(u, v), max(u, v))
                if key not in seen:
                    seen.add(key)
                    triplets.append([u, kg[u][v]["relation"], v])

    return triplets


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_metaqa(
    hops: int = 2,
    split: str = "test",
    data_dir: str = "data/metaqa",
    neighborhood_hops: Optional[int] = None,
    max_nodes_per_subgraph: int = 500,
) -> Dataset:
    """Load a MetaQA split and return a HuggingFace Dataset in RoG format.

    Parameters
    ----------
    hops : int
        Which hop variant to load (1, 2, or 3).
    split : str
        "train", "dev", or "test".
    data_dir : str
        Path to the MetaQA directory containing kb.txt and N-hop/ subdirs.
    neighborhood_hops : int | None
        Hops for subgraph BFS extraction.  Defaults to the same value as `hops`.
    max_nodes_per_subgraph : int
        Hard cap on subgraph size (prevents memory blowup for hub entities).

    Returns
    -------
    datasets.Dataset  with columns: question, answer, q_entity, a_entity, graph
    """
    if hops not in (1, 2, 3):
        raise ValueError(f"hops must be 1, 2, or 3; got {hops}")
    if split not in ("train", "dev", "test"):
        raise ValueError(f"split must be 'train', 'dev', or 'test'; got {split}")

    root     = Path(data_dir)
    kb_path  = root / "kb.txt"
    qa_path  = root / f"{hops}-hop" / "vanilla" / f"qa_{split}.txt"

    if not kb_path.exists():
        raise FileNotFoundError(
            f"MetaQA kb.txt not found at {kb_path}.\n"
            "Expected layout:\n"
            f"  {root}/kb.txt\n"
            f"  {root}/1-hop/vanilla/qa_test.txt\n"
            f"  {root}/2-hop/vanilla/qa_test.txt  (etc.)\n"
        )
    if not qa_path.exists():
        raise FileNotFoundError(f"MetaQA QA file not found: {qa_path}")

    n_hops = neighborhood_hops if neighborhood_hops is not None else hops

    print(f"[MetaQA] Loading WikiMovies KG from {kb_path} ...")
    kg = _parse_kb(kb_path)
    print(
        f"[MetaQA] KG ready: {kg.number_of_nodes():,} nodes, "
        f"{kg.number_of_edges():,} edges"
    )

    print(f"[MetaQA] Loading {hops}-hop {split} split ...")
    raw = _parse_qa(qa_path)
    print(
        f"[MetaQA] {len(raw):,} QA pairs. "
        f"Extracting subgraphs ({n_hops}-hop BFS, max {max_nodes_per_subgraph} nodes/graph) ..."
    )

    records: list[dict] = []
    for s in raw:
        # Expand only from q_entity — using a_entity as a seed would leak the
        # answer at inference time and is inconsistent with RoG-style evaluation.
        subgraph = _extract_subgraph(kg, s["q_entity"], n_hops, max_nodes_per_subgraph)
        records.append({
            "question": s["question"],
            "answer":   s["answer"],
            "q_entity": s["q_entity"],
            "a_entity": s["a_entity"],
            "graph":    subgraph,
        })

    print(f"[MetaQA] Done — {len(records):,} samples ready.")
    return Dataset.from_list(records)
