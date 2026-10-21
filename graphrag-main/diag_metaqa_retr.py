"""
Phase-1 MetaQA retrieval recall experiment (CPU, no LLM). MetaQA's RoG-plan
retrieval fails (recall 0.35) because the RoG planner was tuned on Freebase, not
WikiMovies' 9-relation schema. Test leak-free, schema-appropriate strategies:

  A) bfs_upto3       : BFS paths up to 3 hops from topic entity, cap K (shallow-first)
  B) dfs_depth3      : DFS paths of length EXACTLY 3 from topic, cap K
  C) dense_upto3     : all paths up to 3 hops, dense-ranked (MiniLM) by question, top K
  D) dense_mixed     : depth-3 paths first (dense-ranked), then fill with shorter

Recall = gold answer entity appears as a node on some retrieved path (Hit dir).
None use the answer entity for retrieval -> leak-free.
"""
import argparse
import numpy as np
from collections import deque, defaultdict
from metrics import normalize_answer
from table2_worker import load_ds


def build_adj(triples):
    adj = defaultdict(list)
    for h, r, t in triples:
        r = r.strip()
        adj[h].append((r, t))
        adj[t].append((r, h))
    return adj


def paths_upto(adj, start, max_hop, cap, exact=False):
    """BFS enumerate paths from start; exact=True keeps only length==max_hop."""
    out = []
    q = deque([(start, [])])
    visited_paths = 0
    while q and len(out) < cap * 20:   # bounded exploration
        node, path = q.popleft()
        d = len(path)
        if d >= 1 and (not exact or d == max_hop):
            out.append(path)
        if d < max_hop:
            for rel, nb in adj.get(node, ()):
                if all(nb != p[2] for p in path) and nb != start:  # no revisit
                    q.append((nb, path + [(node, rel, nb)]))
    return out


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
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--cap", type=int, default=15)
    ap.add_argument("--hop", type=int, default=3)
    ap.add_argument("--strategies", default="A,B,C,D")
    args = ap.parse_args()

    ds = load_ds("metaqa-3hop")
    n = min(args.n, len(ds))
    strinclude = set(args.strategies.split(","))

    emb = None
    if {"C", "D"} & strinclude:
        from sentence_transformers import SentenceTransformer
        import torch
        emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                                  device="cuda" if torch.cuda.is_available() else "cpu")

    rec = defaultdict(list)
    reach3 = []
    for i in range(n):
        ex = ds[i]
        golds = ex["answer"] if isinstance(ex["answer"], list) else [ex["answer"]]
        adj = build_adj(ex["graph"])
        starts = [s for s in ex["q_entity"] if s in adj]
        cand = []
        for s in starts:
            cand.extend(paths_upto(adj, s, args.hop, args.cap))
        cand3 = [p for p in cand if len(p) == args.hop]
        reach3.append(len(cand3) > 0)

        if "A" in strinclude:
            rec["A_bfs_upto3"].append(recall_of(cand, golds, args.cap))
        if "B" in strinclude:
            c3 = []
            for s in starts:
                c3.extend(paths_upto(adj, s, args.hop, args.cap, exact=True))
            rec["B_dfs_depth3"].append(recall_of(c3, golds, args.cap))
        if "C" in strinclude and cand:
            qv = emb.encode([ex["question"]], normalize_embeddings=True)[0]
            texts = [rel_text(p) for p in cand]
            pv = emb.encode(texts, normalize_embeddings=True, batch_size=256)
            order = np.argsort(-(pv @ qv))
            ranked = [cand[j] for j in order]
            rec["C_dense_upto3"].append(recall_of(ranked, golds, args.cap))
        if "D" in strinclude and cand:
            qv = emb.encode([ex["question"]], normalize_embeddings=True)[0]
            d3 = [p for p in cand if len(p) == args.hop]
            dl = [p for p in cand if len(p) < args.hop]
            ranked = []
            for grp in (d3, dl):
                if not grp:
                    continue
                pv = emb.encode([rel_text(p) for p in grp], normalize_embeddings=True, batch_size=256)
                order = np.argsort(-(pv @ qv))
                ranked.extend(grp[j] for j in order)
            rec["D_dense_mixed"].append(recall_of(ranked, golds, args.cap))

    print(f"=== MetaQA-3hop retrieval recall  N={n}  cap={args.cap}  hop={args.hop} ===")
    print(f"frac questions with a depth-{args.hop} path available: {np.mean(reach3):.3f}")
    for k in sorted(rec):
        vals = [v for v in rec[k] if v is not None]
        print(f"  {k:18s} recall={np.mean(vals):.4f}  (n={len(vals)})")


if __name__ == "__main__":
    main()
