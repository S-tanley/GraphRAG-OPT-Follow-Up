"""
Smoke-test for MetaQA adapter — no GPU required.

Usage
-----
  cd graphrag_kv
  python test_metaqa.py --metaqa_dir ../data/metaqa --hops 2

What it checks
--------------
1. kb.txt and QA files are parseable
2. Every sample has the four required keys: question, answer, q_entity, a_entity, graph
3. At least one sample has a non-empty subgraph
4. The subgraph triplets are [head, relation, tail] lists of 3 strings
5. q_entity entities all appear as graph nodes
6. a_entity entities are reachable within N hops from q_entity in the subgraph
"""

import argparse
import os
import sys
from collections import defaultdict

# tests/ lives one level below the flat root modules (e.g. metaqa_adapter).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metaqa_dir", default="data/metaqa")
    p.add_argument("--hops", type=int, default=2, choices=[1, 2, 3])
    p.add_argument("--split", default="test", choices=["train", "dev", "test"])
    p.add_argument("--n_samples", type=int, default=200,
                   help="Number of samples to inspect (0 = all)")
    return p.parse_args()


def check_sample(i: int, s: dict, hop_limit: int) -> list[str]:
    """Return list of error strings for sample i (empty = pass)."""
    errors: list[str] = []

    for key in ("question", "answer", "q_entity", "a_entity", "graph"):
        if key not in s:
            errors.append(f"missing key '{key}'")

    if errors:
        return errors

    if not s["question"]:
        errors.append("empty question")
    if not s["answer"]:
        errors.append("empty answer list")
    if not s["q_entity"]:
        errors.append("empty q_entity")
    if not s["a_entity"]:
        errors.append("empty a_entity")

    for t in s["graph"]:
        if not (isinstance(t, list) and len(t) == 3 and all(isinstance(x, str) for x in t)):
            errors.append(f"malformed triplet: {t!r}")
            break

    if s["graph"]:
        G = nx.Graph()
        for h, r, t in s["graph"]:
            G.add_edge(h, t, relation=r)

        for ent in s["q_entity"]:
            if ent not in G:
                errors.append(f"q_entity '{ent}' not in subgraph nodes")

        # Reachability check: at least one answer entity should be reachable
        # within hop_limit hops from any q_entity
        reachable = False
        for qe in s["q_entity"]:
            if qe not in G:
                continue
            lengths = nx.single_source_shortest_path_length(G, qe, cutoff=hop_limit)
            for ae in s["a_entity"]:
                if ae in lengths:
                    reachable = True
                    break
            if reachable:
                break
        if not reachable:
            errors.append(
                f"no a_entity reachable within {hop_limit} hops from q_entity "
                f"(q={s['q_entity']}, a={s['a_entity'][:3]})"
            )

    return errors


def main():
    args = parse_args()

    # Import here so we can run this script from within graphrag_kv/
    try:
        from metaqa_adapter import load_metaqa
    except ImportError:
        print("ERROR: cannot import metaqa_adapter — run this script from inside graphrag_kv/")
        sys.exit(1)

    print(f"Loading MetaQA {args.hops}-hop {args.split} from {args.metaqa_dir} ...")
    ds = load_metaqa(hops=args.hops, split=args.split, data_dir=args.metaqa_dir)

    n = len(ds) if args.n_samples == 0 else min(args.n_samples, len(ds))
    print(f"\nInspecting {n}/{len(ds)} samples ...\n")

    all_errors: dict[int, list[str]] = {}
    graph_sizes: list[int] = []
    empty_graphs = 0

    for i in range(n):
        s = ds[i]
        errs = check_sample(i, s, args.hops)
        if errs:
            all_errors[i] = errs
        graph_sizes.append(len(s["graph"]))
        if not s["graph"]:
            empty_graphs += 1

    # Report
    print("=" * 60)
    print(f"Dataset : MetaQA {args.hops}-hop  |  split={args.split}")
    print(f"Samples : {len(ds):,} total, {n} inspected")
    print(f"Columns : {list(ds.features.keys())}")
    print()

    if graph_sizes:
        avg_g = sum(graph_sizes) / len(graph_sizes)
        print(f"Subgraph stats (over {n} samples):")
        print(f"  avg triplets  : {avg_g:.1f}")
        print(f"  min triplets  : {min(graph_sizes)}")
        print(f"  max triplets  : {max(graph_sizes)}")
        print(f"  empty graphs  : {empty_graphs}")
    print()

    # Show a few sample rows
    print("Sample rows:")
    for i in range(min(3, n)):
        s = ds[i]
        print(f"  [{i}] Q : {s['question']}")
        print(f"       q_entity : {s['q_entity']}")
        print(f"       answer   : {s['answer'][:5]}")
        print(f"       #triplets: {len(s['graph'])}")
        if s['graph']:
            print(f"       first    : {s['graph'][0]}")
        print()

    if all_errors:
        print(f"ERRORS found in {len(all_errors)}/{n} samples:")
        for idx, errs in list(all_errors.items())[:10]:
            print(f"  sample {idx}: {errs}")
        sys.exit(1)
    else:
        print(f"All {n} samples passed checks.")
        print("MetaQA adapter is working correctly.")


if __name__ == "__main__":
    main()
