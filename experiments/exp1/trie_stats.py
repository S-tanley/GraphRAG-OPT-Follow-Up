"""Exp1: token-level redundancy of retrieved paths (CPU only).

For each dataset x path cap x tokenizer, reports:
  - flat path tokens (each path line tokenized as in faithful_pfkv.build_prompt)
  - intra-query trie tokens, two variants:
      line : token-prefix trie over the unchanged per-line tokenization
      seg  : per-hop-unit tokenization concatenated (prefix-stable by construction),
             plus how often it differs from the per-line tokenization
  - prefill attention-cost ratio  sum(S + depth_t) / sum(S + i)
  - cross-query redundancy under random / topical batches: GraphRAG-Opt path-level
    vs trie-node-level (token weighted)
"""

import argparse
import csv
import json
import os
import random
import statistics as st
import sys
from collections import Counter

from transformers import AutoTokenizer

sys.path.insert(0, os.environ.get("GROPT_SRC", "graphrag-main"))
from faithful_pfkv import _ENCODE_PREFIX, path_to_str_rog  # noqa: E402

TOKENIZERS = {"llama2": "meta-llama/Llama-2-7b-hf", "llama3": "meta-llama/Meta-Llama-3-8B",
              "qwen3": "Qwen/Qwen3-8B"}


def units(p):
    out = [p[0][0]]
    for _h, r, t in p:
        out += [" -> " + r, " -> " + t]
    return out


class Tok:
    def __init__(self, name):
        self.t = AutoTokenizer.from_pretrained(name)
        self.memo = {}

    def enc(self, s):
        if s not in self.memo:
            self.memo[s] = tuple(self.t.encode(s, add_special_tokens=False))
        return self.memo[s]

    def line(self, p):
        return self.enc(path_to_str_rog(p) + "\n")

    def seg(self, p):
        ids = []
        for u in units(p):
            ids += self.enc(u)
        return tuple(ids + list(self.enc("\n")))


def trie_nodes(seqs):
    """Set of token-prefix nodes (prefix tuples) and depth list for the trie of seqs."""
    nodes = set()
    for s in seqs:
        for i in range(1, len(s) + 1):
            nodes.add(s[:i])
    return nodes


def query_stats(seqs, S):
    flat = sum(len(s) for s in seqs)
    nodes = trie_nodes(seqs)
    # flat causal: path tokens attend preamble + all earlier path tokens
    att_flat = sum(S + i for i in range(1, flat + 1))
    att_trie = sum(S + len(n) for n in nodes)          # node depth = its prefix length
    return flat, len(nodes), att_flat, att_trie, nodes


def batch_redundancy(per_query, order, B):
    """Token-weighted redundancy over batches: path-level (GraphRAG-Opt), intra-query trie,
    trie + cross-query node sharing."""
    res = []
    for i in range(0, len(order), B):
        batch = order[i:i + B]
        flat = sum(per_query[q]["flat"] for q in batch)
        if flat == 0:
            continue
        paths = {}
        for q in batch:
            for s in per_query[q]["seqs"]:
                paths[s] = len(s)
        nodes = set().union(*(per_query[q]["nodes"] for q in batch))
        intra = sum(per_query[q]["trie"] for q in batch)
        res.append((1 - sum(paths.values()) / flat, 1 - intra / flat, 1 - len(nodes) / flat))
    return [st.mean(x[j] for x in res) for j in range(3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths_dir", default="graphrag-main/precomputed_paths_v2")
    ap.add_argument("--out", default="outputs/exp1/trie_stats.csv")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    for ds in ["rmanluo-RoG-cwq", "rmanluo-RoG-webqsp"]:
        D = json.load(open(os.path.join(args.paths_dir, f"paths_{ds}.json")))
        qs = [q for q in D if D[q]]
        for tname, tpath in TOKENIZERS.items():
            tk = Tok(tpath)
            S = len(tk.t.encode(_ENCODE_PREFIX["rog"], add_special_tokens=True))
            for cap in [8, 30]:
                for variant in ["line", "seg"]:
                    per_q, mism, n_paths = {}, 0, 0
                    agg = Counter()
                    for q in qs:
                        P = [[tuple(t) for t in p] for p in D[q][:cap]]
                        seqs = [tk.line(p) if variant == "line" else tk.seg(p) for p in P]
                        if variant == "seg":
                            mism += sum(tk.seg(p) != tk.line(p) for p in P)
                        n_paths += len(P)
                        flat, trie, af, at, nodes = query_stats(seqs, S)
                        per_q[q] = dict(flat=flat, trie=trie, seqs=seqs, nodes=nodes)
                        agg["flat"] += flat; agg["trie"] += trie; agg["af"] += af; agg["at"] += at
                    sav_q = [1 - per_q[q]["trie"] / per_q[q]["flat"] for q in qs]
                    row = dict(dataset=ds, tokenizer=tname, cap=cap, variant=variant, preamble=S,
                               queries=len(qs), paths_per_q=round(n_paths / len(qs), 2),
                               flat_tok_per_q=round(agg["flat"] / len(qs), 1),
                               trie_tok_per_q=round(agg["trie"] / len(qs), 1),
                               intra_saving_tokenweighted=round(1 - agg["trie"] / agg["flat"], 4),
                               intra_saving_q_mean=round(st.mean(sav_q), 4),
                               intra_saving_q_p10=round(sorted(sav_q)[len(sav_q) // 10], 4),
                               attn_cost_ratio=round(agg["at"] / agg["af"], 4),
                               seg_line_mismatch=(round(mism / n_paths, 4) if variant == "seg" else ""))
                    rnd = qs[:]
                    random.Random(0).shuffle(rnd)
                    top = sorted(qs, key=lambda q: Counter(p[0][0] for p in D[q]).most_common(1)[0][0])
                    for oname, order in [("random", rnd), ("topical", top)]:
                        for B in [64, 512]:
                            pl, intra, node = batch_redundancy(per_q, order, B)
                            row[f"{oname}{B}_path"] = round(pl, 4)
                            row[f"{oname}{B}_node"] = round(node, 4)
                    rows.append(row)
                    print(row, flush=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
