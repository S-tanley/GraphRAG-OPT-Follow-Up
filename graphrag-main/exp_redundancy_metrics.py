"""
Dig-deep: what well-defined within-batch redundancy metrics exist for GraphRAG,
and what does each mean? Computes, per dataset (CWQ/WEB) under entity-clustered
(topical) batching, across batch sizes, FOUR candidate metrics:

  (1) path_red   = fraction of retrieved PATH INSTANCES that are duplicates
                   = sum(r-1)/sum(r) over distinct paths. "of all path-prefills
                   the batch would do, what fraction is redundant work?"
  (2) token_red  = same, weighted by path token length = the W_red fraction.
                   "of all path-prefill TOKENS, what fraction is redundant?"
  (3) query_cov  = fraction of QUERIES that contain >=1 path shared with another
                   query in the batch (path appears in >=2 distinct queries).
                   "what fraction of queries benefit from cross-query reuse?"
  (4) shared_path_frac = distinct shared paths / distinct paths. "of the unique
                   paths present, what fraction recur across queries?"

These are the candidates for a 'multiple bars per batch size' figure. Note:
path_red and token_red nearly coincide (within ~2 pts). 'Batch redundancy' in the
original figure had NO formula and was a hardcoded ~80% (template-inflated).
"""
import json, os, random
from collections import defaultdict, Counter
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = {"CWQ": "precomputed_paths/paths_rmanluo-RoG-cwq.json",
            "WEB": "precomputed_paths/paths_rmanluo-RoG-webqsp.json"}
BATCHES = [16, 32, 64, 128, 256, 512]
MAX_PATHS = 8
SEED = 42


def path_to_str(p):
    if isinstance(p, str): return p
    if not p: return ""
    parts = [p[0][0]]
    for h, r, t in p:
        parts += [r, t]
    return " -> ".join(map(str, parts))


def topic_entity(paths):
    c = defaultdict(int)
    for p in paths:
        for h, r, t in p:
            c[h] += 1; c[t] += 1
    return max(c, key=c.get) if c else ""


def batch_metrics(batch):
    # batch = list of (qid, pstrs, tlens)
    inst = 0; tok_inst = 0
    distinct_tok = {}
    qfreq = Counter()            # distinct-query count per path string
    for qi, (q, pstrs, tlens) in enumerate(batch):
        for s, tl in zip(pstrs, tlens):
            inst += 1; tok_inst += tl
            distinct_tok[s] = tl
        for s in set(pstrs):     # count each path once per query
            qfreq[s] += 1
    if inst == 0:
        return 0, 0, 0, 0
    distinct = len(distinct_tok)
    tok_distinct = sum(distinct_tok.values())
    path_red = (inst - distinct) / inst
    token_red = (tok_inst - tok_distinct) / max(1, tok_inst)
    shared = {s for s, c in qfreq.items() if c >= 2}
    shared_path_frac = len(shared) / max(1, distinct)
    nq_benefit = sum(1 for (q, pstrs, tlens) in batch if any(s in shared for s in pstrs))
    query_cov = nq_benefit / len(batch)
    return path_red, token_red, query_cov, shared_path_frac


def main():
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True)
    print(f"{'ds':4}{'B':>5}{'path%':>8}{'token%':>8}{'qcov%':>8}{'shpath%':>9}")
    for ds, path in DATASETS.items():
        d = json.load(open(os.path.join(HERE, path)))
        items = []
        for q, paths in d.items():
            paths = [[tuple(t) for t in p] for p in paths][:MAX_PATHS]
            pstrs = [path_to_str(p) for p in paths]
            tlens = [len(tok.encode(s + "\n", add_special_tokens=False)) for s in pstrs]
            items.append((q, pstrs, tlens, topic_entity(paths)))
        ordered = sorted(items, key=lambda x: x[3])           # topical (entity-clustered)
        packed = [(q, ps, tl) for (q, ps, tl, te) in ordered]
        for B in BATCHES:
            if B > len(packed): continue
            prs = trs = qcs = sps = 0.0; n = 0
            for i in range(0, len(packed) - B + 1, B):
                pr, tr, qc, sp = batch_metrics(packed[i:i + B])
                prs += pr; trs += tr; qcs += qc; sps += sp; n += 1
            if n:
                print(f"{ds:4}{B:>5}{prs/n*100:>8.1f}{trs/n*100:>8.1f}{qcs/n*100:>8.1f}{sps/n*100:>9.1f}")


if __name__ == "__main__":
    main()
