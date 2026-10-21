"""Build LoRA training data from the TRAIN splits (no test questions).

Mirrors the test-time retrieval distribution: relation plans are grounded with RoG's
bfs_with_rule from the question entities (same fan-out, same order, same cap) — but the
plans come from RoG's own training supervision (relation sequences of shortest
question->answer paths in the per-question graph) instead of the planner model.

  python experiments/exp3/build_train_paths.py --out outputs/exp3/train_paths.jsonl
"""

import argparse
import ast
import json
import os
import random
import sys
from collections import defaultdict, deque
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "graphrag-main"))
from graph_utils import _bfs_with_rule  # noqa: E402

MAX_HOP, CAP, MAX_RULES = 3, 8, 3


def as_list(x):
    return ast.literal_eval(x) if isinstance(x, str) else list(x)


def gold_rules(adj, q_ents, a_ents):
    """Relation sequences of shortest q->a paths (<= MAX_HOP), at most MAX_RULES."""
    rules = []
    targets = set(a_ents)
    for s in q_ents:
        if s not in adj:
            continue
        frontier, seen, found = deque([(s, [])]), {s}, []
        while frontier:
            node, rel = frontier.popleft()
            if found and len(rel) > len(found[0]):
                break
            if node in targets and rel:
                found.append(rel)
                continue
            if len(rel) == MAX_HOP:
                continue
            for r, nb in adj[node]:
                if nb not in seen or nb in targets:
                    if nb not in targets:
                        seen.add(nb)
                    frontier.append((nb, rel + [r]))
        for rel in found:
            if rel not in rules:
                rules.append(rel)
    return rules[:MAX_RULES]


def one(ex):
    q_ents, a_ents = as_list(ex["q_entity"]), as_list(ex["a_entity"])
    adj = defaultdict(list)
    for h, r, t in as_list(ex["graph"]):
        r = r.strip()
        adj[h].append((r, t))
        adj[t].append((r, h))
    rules = gold_rules(adj, q_ents, a_ents)
    paths, seen = [], set()
    for rule in rules:
        for s in q_ents:
            for p in _bfs_with_rule(adj, s, rule):
                if tuple(p) not in seen:
                    seen.add(tuple(p))
                    paths.append(p)
    if len(paths) > CAP:                       # keep answer-bearing paths, fill with siblings
        hit = [p for p in paths if p[-1][2] in a_ents]
        rest = [p for p in paths if p[-1][2] not in a_ents]
        rnd = random.Random(hash(ex["question"]) & 0xffff)
        rnd.shuffle(hit); rnd.shuffle(rest)
        keep = set(map(tuple, hit[:max(1, CAP // 2)] + rest))
        paths = [p for p in paths if tuple(p) in keep][:CAP]
    return dict(question=ex["question"], answer=as_list(ex["answer"]), paths=paths)


def main():
    from datasets import load_dataset
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_cwq", type=int, default=4000)
    ap.add_argument("--n_web", type=int, default=2826)
    args = ap.parse_args()
    rows = []
    for name, n in [("rmanluo/RoG-cwq", args.n_cwq), ("rmanluo/RoG-webqsp", args.n_web)]:
        ds = load_dataset(name, split="train").shuffle(seed=0)
        test_q = set(load_dataset(name, split="test")["question"])
        exs = [ds[i] for i in range(min(n, len(ds))) if ds[i]["question"] not in test_q]
        with Pool(64) as pool:
            out = pool.map(one, exs, chunksize=16)
        kept = [o for o in out if o["paths"]]
        for o in kept:
            o["dataset"] = name
        print(f"{name}: {len(exs)} train questions -> {len(kept)} with grounded paths; "
              f"paths/q {sum(len(o['paths']) for o in kept) / max(1, len(kept)):.1f}", flush=True)
        rows += kept
    random.Random(0).shuffle(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for o in rows:
            f.write(json.dumps(o) + "\n")
    print("wrote", len(rows), "->", args.out)


if __name__ == "__main__":
    main()
