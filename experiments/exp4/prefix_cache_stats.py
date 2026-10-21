"""How much path-region prefill does each scheme skip, per batch (CPU only)?

  flat+radix : M0 prompt, prefix caching (RadixAttention): a token is skipped if the whole
               prompt prefix up to it already appeared in an earlier query of the batch
  T+radix    : trie-as-text prompt, prefix caching
  T          : trie-as-text alone (intra-query dedup only, no cross-query reuse)
  M2 nodes   : trie mask, a trie node is computed once per batch (intra + cross query)

All numbers = 1 - computed path-region tokens / flat path-region tokens (M0, no caching).
The shared preamble is excluded (every scheme caches it).
"""

import json
import os
import random
import sys
from collections import Counter

from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "graphrag-main"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "common"))
from conditions import path_lines, trie_text_lines  # noqa: E402

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")


def radix_skipped(seqs):
    """Tokens of each sequence covered by a prefix seen earlier in the batch."""
    seen, skipped = set(), 0
    for s in seqs:
        hit = 0
        for i in range(1, len(s) + 1):
            if tuple(s[:i]) in seen:
                hit = i
            else:
                break
        skipped += hit
        for i in range(1, len(s) + 1):
            seen.add(tuple(s[:i]))
    return skipped


for ds in ["rmanluo-RoG-cwq", "rmanluo-RoG-webqsp"]:
    D = json.load(open(os.path.join(ROOT, "graphrag-main/precomputed_paths_v2", f"paths_{ds}.json")))
    qs = [q for q in D if D[q]]
    for cap in [8, 30]:
        flat, tt, nodes_q = {}, {}, {}
        for q in qs:
            P = [[tuple(t) for t in p] for p in D[q][:cap]]
            lines = path_lines(tok, P, cap)
            flat[q] = [t for l in lines for t in l]
            tt[q] = [t for s in trie_text_lines(P, cap) for t in tok.encode(s, add_special_tokens=False)]
            nodes_q[q] = {tuple(l[:i + 1]) for l in lines for i in range(len(l))}
        rnd = qs[:]
        random.Random(0).shuffle(rnd)
        top = sorted(qs, key=lambda q: Counter(p[0][0] for p in D[q]).most_common(1)[0][0])
        for oname, order in [("random", rnd), ("topical", top)]:
            for B in [64, 512]:
                F = Fr = Tt = Tr = Nn = 0
                for i in range(0, len(order), B):
                    b = order[i:i + B]
                    f = sum(len(flat[q]) for q in b)
                    F += f
                    Fr += radix_skipped([flat[q] for q in b])
                    Tt += f - sum(len(tt[q]) for q in b)
                    Tr += f - sum(len(tt[q]) for q in b) + radix_skipped([tt[q] for q in b])
                    Nn += f - len(set().union(*(nodes_q[q] for q in b)))
                print(f"{ds[-6:]} cap{cap:<2} {oname:7s} B={B:<3}  skipped: flat+radix {Fr / F:.3f} | "
                      f"T {Tt / F:.3f} | T+radix {Tr / F:.3f} | M2 nodes {Nn / F:.3f}", flush=True)
