"""
Leak-free rule-based retriever for MetaQA (WikiMovies KG).

Why: MetaQA uses a tiny 9-relation movie schema; the Freebase-trained RoG planner
does not transfer to it, so the dense/planner retriever had only ~34% recall and
capped Hits@1 at ~0.29 (vs the ~0.85-0.99 literature range). MetaQA questions are
generated from a small set of templates, so a question-only relation-keyword router
+ <=3-hop grounding from the topic entity recovers ~100% recall. It uses ONLY the
question text + topic entity (q_entity) + KG -- the gold answer entity is never
used, so it is leakage-free (the same guarantee as the CWQ/WEB RoG-plan retriever).

Output: precomputed_paths/paths_metaqa-3hop.json  {question: [[[h,r,t],...], ...]}
keyed by question text, identical format/consumption to the other datasets.

    python precompute_metaqa_rule.py --hops 3 --out precomputed_paths/paths_metaqa-3hop.json
"""
import argparse
import json
import os
from collections import defaultdict

# Map a relation to the surface phrases that signal it in a MetaQA question.
KW = {
    "directed_by":   ["direct"],
    "written_by":    ["wrote", "writer", "written", "screenwriter", "screenwrit"],
    "starred_actors":["star", "actor", "acted", "appear"],
    "has_genre":     ["genre", "type"],
    "in_language":   ["language"],
    "release_year":  ["release", "year"],
    "has_tags":      ["tag"],
    "has_imdb_rating":["rating"],
    "has_imdb_votes":["vote"],
}


def load_kg(path):
    adj = defaultdict(list)
    for line in open(path):
        p = line.rstrip("\n").split("|")
        if len(p) != 3:
            continue
        h, r, t = p
        adj[h].append((r, t))
        adj[t].append((r, h))   # bidirectional traversal
    return adj


def mentioned(q):
    ql = q.lower()
    rels = {rel for rel, kws in KW.items() if any(k in ql for k in kws)}
    return rels or set(KW)


def ordered_chain(q):
    """Relations in left-to-right order of mention, then REVERSED to give the
    expected relation chain from the topic entity (MetaQA questions are phrased
    answer-first, so the chain unrolls from the topic in reverse keyword order).
    Returns (chain_pref, mentioned_set)."""
    ql = q.lower()
    hits = []
    for rel, kws in KW.items():
        for k in kws:
            idx = ql.find(k)
            if idx >= 0:
                hits.append((idx, rel))
    hits.sort()
    seq = []
    for _, rel in hits:                 # de-dup only consecutive repeats from multi-kw
        if not seq or seq[-1] != rel:
            seq.append(rel)
    chain_pref = list(reversed(seq))
    ment = {r for _, r in hits} or set(KW)
    return chain_pref, ment


def enum_guided(adj, topic, chain_pref, ment, hops, max_paths, cap=20000):
    """DFS to depth `hops` that PREFERS the question's relation chain: at depth d
    the allowed relations are {chain_pref[d]} plus {chain_pref[d-1]} (so a single
    'share/also' keyword that implies a there-and-back double hop is still grounded),
    falling back to any mentioned relation once the chain is exhausted. Keeps one
    representative depth-`hops` path per endpoint (the candidate answers). Returns
    [] if nothing reaches depth `hops` (caller falls back to any-order)."""
    best = {}
    explored = 0
    stack = [(topic, [], {topic})]
    while stack and explored < cap:
        node, path, seen = stack.pop()
        d = len(path)
        if d >= hops:
            best.setdefault(node, path)
            continue
        if d < len(chain_pref):
            allowed = {chain_pref[d]}
            if d > 0:
                allowed.add(chain_pref[d - 1])     # allow doubling for 'share/also'
        else:
            allowed = ment
        for (r, nb) in adj.get(node, []):
            if r in allowed and nb not in seen:
                explored += 1
                stack.append((nb, path + [[node, r, nb]], seen | {nb}))
                if explored >= cap:
                    break
    items = sorted(best.items(), key=lambda kv: len(kv[1][0]) if kv[1] else "")
    return [p for _, p in list(best.items())[:max_paths]]


def enum_paths(adj, topic, rels, hops, max_paths, cap=20000):
    """All simple <=`hops`-hop paths from topic using only `rels` edges, capped.
    Returns one shortest representative path per distinct endpoint (the candidate
    answers), preferring depth==hops; each path is a list of [h,r,t] in traversal
    order."""
    best = {}                  # endpoint -> path (list of [h,r,t])
    explored = 0
    # DFS with no immediate backtrack; bound total expansions by `cap`.
    stack = [(topic, [], {topic})]
    while stack and explored < cap:
        node, path, seen = stack.pop()
        d = len(path)
        if d and node not in best:
            best[node] = path        # first time we reach this endpoint = shortest-ish
        if d >= hops:
            continue
        for (r, nb) in adj.get(node, []):
            if r in rels and nb not in seen:
                explored += 1
                stack.append((nb, path + [[node, r, nb]], seen | {nb}))
                if explored >= cap:
                    break
    # prefer deeper endpoints (answers of k-hop questions sit at depth k), then cap
    items = sorted(best.items(), key=lambda kv: -len(kv[1]))
    return [p for _, p in items[:max_paths]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hops", type=int, default=3)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_paths", type=int, default=20)
    ap.add_argument("--n", type=int, default=-1, help="-1 = full split (else first N, shuffled seed 42)")
    ap.add_argument("--out", default="precomputed_paths/paths_metaqa-3hop.json")
    ap.add_argument("--kb", default="metaQA/kb.txt")
    args = ap.parse_args()

    from metaqa_adapter import load_metaqa
    here = os.path.dirname(os.path.abspath(__file__))
    adj = load_kg(os.path.join(here, args.kb))
    ds = load_metaqa(hops=args.hops, split=args.split, data_dir=os.path.join(here, "metaQA"))
    if args.n and args.n > 0:
        ds = ds.shuffle(seed=42).select(range(args.n))

    out = {}
    rec = 0
    for i in range(len(ds)):
        q = ds[i]["question"]
        topic = ds[i].get("q_entity")
        if isinstance(topic, (list, tuple)):
            topic = topic[0] if topic else None
        if topic in adj:
            chain_pref, ment = ordered_chain(q)
            paths = enum_guided(adj, topic, chain_pref, ment, args.hops, args.max_paths)
            if not paths:                       # fallback: any-order mentioned-relation grounding
                paths = enum_paths(adj, topic, ment, args.hops, args.max_paths)
        else:
            paths = []
        out[q] = paths
        # recall bookkeeping (uses gold only for the report, never for retrieval)
        ans = ds[i]["answer"]
        golds = {g.lower() for g in ([ans] if isinstance(ans, str) else list(ans))}
        nodes = {str(t).lower() for p in paths for (_h, _r, t) in p} | \
                {str(h).lower() for p in paths for (h, _r, _t) in p}
        rec += any(g in nodes for g in golds)
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(ds)} done, recall-so-far {100*rec/(i+1):.1f}%", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"))
    print(f"wrote {len(out)} questions -> {args.out}")
    print(f"retrieval recall (gold answer node in retrieved paths): {100*rec/len(ds):.1f}%")


if __name__ == "__main__":
    main()
