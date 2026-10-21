"""
MetaQA recall with DEPTH-TARGETED enumeration + keyword-relation prior.

Root cause of the recall ceiling: BFS enumeration capped at N paths exhausts
shallow (depth-1/2) paths before reaching depth-3, where 3-hop answers live.
Fix: enumerate depth-EXACTLY-3 paths first (DFS), and bias ranking toward paths
whose relations match question keywords (MetaQA's 9-relation schema is keyworded).
Leak-free (no answer entity).
"""
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path
from metrics import normalize_answer
from table2_worker import load_ds

KEYMAP = {
    "directed_by": ["direct", "director"],
    "starred_actors": ["star", "actor", "act ", "acted", "appear", "cast"],
    "written_by": ["wrote", "writer", "written"],
    "has_genre": ["genre"],
    "in_language": ["language", "languages"],
    "release_year": ["year", "when", "release"],
    "has_tags": ["tag", "topic", "about", "theme"],
    "has_imdb_rating": ["rating", "rated"],
    "has_imdb_votes": ["votes", "voted", "popular"],
}


def parse_kb(kb_path):
    adj = defaultdict(list)
    for line in open(kb_path, encoding="utf-8"):
        p = line.strip().split("|")
        if len(p) == 3:
            h, r, t = p[0].strip(), p[1].strip(), p[2].strip()
            adj[h].append((r, t)); adj[t].append((r, h))
    return adj


def dfs_depth(adj, start, depth, hard_cap=4000):
    """All simple paths of length EXACTLY `depth` from start (DFS, bounded)."""
    out = []
    stack = [(start, [], {start})]
    while stack and len(out) < hard_cap:
        node, path, seen = stack.pop()
        if len(path) == depth:
            out.append(path); continue
        for rel, nb in adj.get(node, ()):
            if nb not in seen:
                stack.append((nb, path + [[node, rel, nb]], seen | {nb}))
    return out


def kw_relations(question):
    ql = question.lower()
    return {rel for rel, kws in KEYMAP.items() if any(k in ql for k in kws)}


def rel_text(path):
    return " ".join(t[1].replace("_", " ") for t in path)


def recall_of(paths, golds, cap):
    gnorm = [normalize_answer(g) for g in golds if normalize_answer(g)]
    if not gnorm:
        return None
    for p in paths[:cap]:
        for trip in p:
            for node in (trip[0], trip[2]):
                if any(g in normalize_answer(node) for g in gnorm):
                    return 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--hop", type=int, default=3)
    ap.add_argument("--caps", default="15,30")
    args = ap.parse_args()
    caps = [int(c) for c in args.caps.split(",")]

    adj = parse_kb(Path(__file__).resolve().parent / "metaQA" / "kb.txt")
    ds = load_ds("metaqa-3hop")
    n = min(args.n, len(ds))
    from sentence_transformers import SentenceTransformer
    import torch
    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                              device="cuda" if torch.cuda.is_available() else "cpu")

    res = {("d3_dense", c): [] for c in caps}
    res.update({("d3_kw", c): [] for c in caps})
    for i in range(n):
        ex = ds[i]
        golds = ex["answer"] if isinstance(ex["answer"], list) else [ex["answer"]]
        starts = [s for s in ex["q_entity"] if s in adj]
        cand = []
        for s in starts:
            cand.extend(dfs_depth(adj, s, args.hop))
        if not cand:
            for c in caps:
                res[("d3_dense", c)].append(0); res[("d3_kw", c)].append(0)
            continue
        qv = emb.encode([ex["question"]], normalize_embeddings=True)[0]
        pv = emb.encode([rel_text(p) for p in cand], normalize_embeddings=True, batch_size=512)
        sims = pv @ qv
        # plain dense
        order = np.argsort(-sims)
        ranked = [cand[j] for j in order]
        # keyword-boosted: +0.5 per path-relation that matches a question keyword
        kws = kw_relations(ex["question"])
        boost = np.array([0.5 * sum(1 for t in p if t[1] in kws) for p in cand])
        order_kw = np.argsort(-(sims + boost))
        ranked_kw = [cand[j] for j in order_kw]
        for c in caps:
            res[("d3_dense", c)].append(recall_of(ranked, golds, c))
            res[("d3_kw", c)].append(recall_of(ranked_kw, golds, c))

    print(f"=== MetaQA-3hop DEPTH-3 enumeration  N={n} ===")
    for k in sorted(res):
        v = [x for x in res[k] if x is not None]
        print(f"  {k[0]:10s} cap={k[1]:3d}  recall={np.mean(v):.4f}")


if __name__ == "__main__":
    main()
