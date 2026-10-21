"""
Test MetaQA recall working from the FULL WikiMovies KG (bypass the 500-node
subgraph cap that truncates the hop-3 neighborhood). Leak-free: expansion uses
only the topic entity.

Measures:
  - answer_in_3hop : gold answer reachable within 3 hops of topic in full KG (ceiling)
  - recall@cap for dense-ranked depth<=3 paths, and depth==3-first ranking
"""
import argparse
import numpy as np
from collections import deque, defaultdict
from pathlib import Path
from metrics import normalize_answer
from table2_worker import load_ds


def parse_kb(kb_path):
    adj = defaultdict(list)
    for line in open(kb_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        p = line.split("|")
        if len(p) == 3:
            h, r, t = p[0].strip(), p[1].strip(), p[2].strip()
            adj[h].append((r, t))
            adj[t].append((r, h))
    return adj


def enum_paths(adj, start, max_hop, hard_cap=4000):
    """All simple paths up to max_hop from start (bounded)."""
    out = []
    q = deque([(start, [], {start})])
    while q and len(out) < hard_cap:
        node, path, seen = q.popleft()
        d = len(path)
        if d >= 1:
            out.append(path)
        if d < max_hop:
            for rel, nb in adj.get(node, ()):
                if nb not in seen:
                    q.append((nb, path + [(node, rel, nb)], seen | {nb}))
    return out


def nodes_within(adj, start, max_hop):
    seen = {start}
    frontier = [start]
    for _ in range(max_hop):
        nxt = []
        for n in frontier:
            for _r, nb in adj.get(n, ()):
                if nb not in seen:
                    seen.add(nb)
                    nxt.append(nb)
        frontier = nxt
    return seen


def rel_text(path):
    return " ".join(t[1].replace("_", " ") for t in path)


def recall_of(paths, golds, cap):
    gnorm = [normalize_answer(g) for g in golds if normalize_answer(g)]
    if not gnorm:
        return None
    for p in paths[:cap]:
        for trip in p:
            for node in (trip[0], trip[2]):
                nn = normalize_answer(node)
                if any(g in nn for g in gnorm):
                    return 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--hop", type=int, default=3)
    ap.add_argument("--caps", default="15,30")
    args = ap.parse_args()
    caps = [int(c) for c in args.caps.split(",")]

    HERE = Path(__file__).resolve().parent
    adj = parse_kb(HERE / "metaQA" / "kb.txt")
    print(f"full KG adj: {len(adj):,} nodes")

    ds = load_ds("metaqa-3hop")
    n = min(args.n, len(ds))

    from sentence_transformers import SentenceTransformer
    import torch
    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                              device="cuda" if torch.cuda.is_available() else "cpu")

    in3, recall = [], {("upto", c): [] for c in caps}
    recall.update({("d3first", c): [] for c in caps})
    for i in range(n):
        ex = ds[i]
        golds = ex["answer"] if isinstance(ex["answer"], list) else [ex["answer"]]
        gnorm = [normalize_answer(g) for g in golds if normalize_answer(g)]
        starts = [s for s in ex["q_entity"] if s in adj]
        # ceiling: answer node within 3 hops
        reach = set()
        for s in starts:
            reach |= nodes_within(adj, s, args.hop)
        hit_reach = any(any(g in normalize_answer(node) for g in gnorm) for node in reach)
        in3.append(int(hit_reach))

        cand = []
        for s in starts:
            cand.extend(enum_paths(adj, s, args.hop))
        if not cand:
            for c in caps:
                recall[("upto", c)].append(0)
                recall[("d3first", c)].append(0)
            continue
        qv = emb.encode([ex["question"]], normalize_embeddings=True)[0]
        pv = emb.encode([rel_text(p) for p in cand], normalize_embeddings=True, batch_size=512)
        sims = pv @ qv
        order = np.argsort(-sims)
        ranked = [cand[j] for j in order]
        # depth-3-first ranking
        d3 = [cand[j] for j in order if len(cand[j]) == args.hop]
        dl = [cand[j] for j in order if len(cand[j]) < args.hop]
        d3first = d3 + dl
        for c in caps:
            recall[("upto", c)].append(recall_of(ranked, golds, c))
            recall[("d3first", c)].append(recall_of(d3first, golds, c))

    print(f"=== MetaQA-3hop FULL-KG  N={n}  hop={args.hop} ===")
    print(f"answer within {args.hop} hops of topic (ceiling): {np.mean(in3):.4f}")
    for k in sorted(recall):
        vals = [v for v in recall[k] if v is not None]
        print(f"  rank={k[0]:8s} cap={k[1]:3d}  recall={np.mean(vals):.4f}")


if __name__ == "__main__":
    main()
