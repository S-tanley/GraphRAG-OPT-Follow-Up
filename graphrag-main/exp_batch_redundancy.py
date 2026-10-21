"""
Within-batch redundancy vs batch-formation policy (reviewer Q1 + Observation 2).

The reviewer correctly notes that redundancy/speedup depend on HOW batches are
formed: sequential file order can inflate redundancy via benchmark ordering, while
arrival-time (random) batching across independent users would show less. We
therefore measure path-level and token-level redundancy under THREE policies:

  sequential : consecutive questions in benchmark file order (what inflates).
  random     : uniform shuffle (seed 42) = arrival-time across independent users.
  topical    : entity-clustered (questions sharing a topic entity batched
               together) = real knowledge-intensive workloads with the entity
               locality established in Fig 4 (Zipfian popularity + session sim).

Redundancy in a batch B (over the retrieved path units of all queries):
  path-level  = 1 - |distinct tok(pi)| / |total path instances|
  token-level = (sum_inst |tok(pi)| - sum_distinct |tok(pi)|) / sum_inst |tok(pi)|
                = the W_red fraction (redundant prefill tokens).
Averaged over all size-B batches. Output: results/redundancy.csv + summary.
"""
import json, os, csv, random
from collections import defaultdict
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = {"CWQ": "precomputed_paths/paths_rmanluo-RoG-cwq.json",
            "WEB": "precomputed_paths/paths_rmanluo-RoG-webqsp.json",
            "MQA": "precomputed_paths/paths_metaqa-3hop.json"}
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
MAX_PATHS = 8            # paths injected per query (matches the eval worker)
SEED = 42


def path_to_str(p):
    if isinstance(p, str):
        return p
    if not p:
        return ""
    parts = [p[0][0]]
    for h, r, t in p:
        parts += [r, t]
    return " -> ".join(map(str, parts))


def topic_entity(paths):
    """Most frequent entity (node) across a query's paths = its topic anchor."""
    c = defaultdict(int)
    for p in paths:
        for h, r, t in p:
            c[h] += 1; c[t] += 1
    return max(c, key=c.get) if c else ""


def order(qlist, policy):
    if policy == "sequential":
        return qlist
    if policy == "random":
        r = random.Random(SEED); q = qlist[:]; r.shuffle(q); return q
    if policy == "topical":
        # sort so queries sharing a topic entity are adjacent -> topical batches
        return sorted(qlist, key=lambda x: x[2])
    raise ValueError(policy)


def batch_redundancy(batch):
    """batch = list of (q, pathstrs, toklens). Returns (path_red, token_red)."""
    inst = 0; tok_inst = 0
    distinct_tok = {}
    for _q, pstrs, tlens in batch:
        for s, tl in zip(pstrs, tlens):
            inst += 1; tok_inst += tl
            distinct_tok[s] = tl
    if inst == 0:
        return 0.0, 0.0
    distinct = len(distinct_tok)
    tok_distinct = sum(distinct_tok.values())
    return (inst - distinct) / inst, (tok_inst - tok_distinct) / max(1, tok_inst)


def main():
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True)
    rows = []
    for ds, path in DATASETS.items():
        d = json.load(open(os.path.join(HERE, path)))
        items = []
        for q, paths in d.items():
            paths = [[tuple(t) for t in p] for p in paths][:MAX_PATHS]
            pstrs = [path_to_str(p) for p in paths]
            tlens = [len(tok.encode(s + "\n", add_special_tokens=False)) for s in pstrs]
            items.append((q, pstrs, tlens, topic_entity(paths)))
        # repack with topic for ordering
        items = [(q, pstrs, tlens, te) for (q, pstrs, tlens, te) in items]
        for policy in ["sequential", "random", "topical"]:
            ordered = order([(q, ps, tl, te) for (q, ps, tl, te) in items],
                            policy) if policy != "topical" else \
                      sorted(items, key=lambda x: x[3])
            if policy == "random":
                r = random.Random(SEED); ordered = items[:]; r.shuffle(ordered)
            elif policy == "sequential":
                ordered = items
            packed = [(q, ps, tl) for (q, ps, tl, te) in ordered]
            for B in BATCH_SIZES:
                if B > len(packed):
                    continue
                prs, trs = [], []
                for i in range(0, len(packed) - B + 1, B):   # non-overlapping
                    pr, tr = batch_redundancy(packed[i:i + B])
                    prs.append(pr); trs.append(tr)
                if prs:
                    rows.append((ds, policy, B, len(prs),
                                 sum(prs) / len(prs), sum(trs) / len(trs)))
                    print(f"{ds:4} {policy:10} B={B:<4} n_batches={len(prs):<5} "
                          f"path_red={sum(prs)/len(prs)*100:5.1f}%  "
                          f"token_red={sum(trs)/len(trs)*100:5.1f}%", flush=True)

    with open(os.path.join(HERE, "results", "redundancy.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "batching", "batch_size", "n_batches",
                    "path_redundancy", "token_redundancy"])
        w.writerows(rows)
    print("\n[wrote results/redundancy.csv]")


if __name__ == "__main__":
    main()
