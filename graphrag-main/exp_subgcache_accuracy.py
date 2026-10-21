#!/usr/bin/env python3
"""
exp_subgcache_accuracy.py — Reviewer-B B1 accuracy axis.

Hits@1 (leakage-free) for the four KV-reuse fidelity points, isolating the cost of
SubGCache's no-positional-handling design contract and our recovery of it. Accuracy
is a per-query property (batch-independent), so it answers the matched-batch
accuracy request directly.

  full     : full in-context prefill (upper bound; == GraphRAG-R).
  noalign  : verbatim KV reuse, realignment OFF = SubGCache's position-handling
             contract, reimplemented in our engine (GROPT_NO_REALIGN=1).
  realign  : + RoPE realignment (ours).
  realign_rc: + selective recompute (EPIC first-k), our full operating point.

Usage:
  python exp_subgcache_accuracy.py --model meta-llama/Llama-2-7b-hf \
     --datasets rmanluo/RoG-cwq rmanluo/RoG-webqsp --n 300 --out results/subgcache_accuracy.csv
"""
import argparse, csv, json, os, sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hf_reuse import HFPathCache, full_generate, generic_reuse_generate   # noqa
from faithful_pfkv import encode_prefix_ids                                # noqa
from metrics import evaluate_metrics                                       # noqa

PATHS_FILE = {
    "rmanluo/RoG-cwq": "precomputed_paths/paths_rmanluo-RoG-cwq.json",
    "rmanluo/RoG-webqsp": "precomputed_paths/paths_rmanluo-RoG-webqsp.json",
    "metaqa-3hop": "precomputed_paths/paths_metaqa-3hop.json",
}


def load_ds(name, n, seed=42):
    if name == "metaqa-3hop":
        from metaqa_adapter import load_metaqa
        ds = load_metaqa(hops=3, split="test", data_dir=str(HERE / "metaQA")).shuffle(seed=seed)
    else:
        from datasets import load_dataset
        ds = load_dataset(name, split="test").shuffle(seed=seed)
    return ds.select(range(min(n, len(ds))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--datasets", nargs="+", default=["rmanluo/RoG-cwq", "rmanluo/RoG-webqsp"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--max_paths", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--recompute_k", type=int, default=2)
    ap.add_argument("--out", default="results/subgcache_accuracy.csv")
    a = ap.parse_args()

    device = "cuda"
    template = "rog" if "rog" in a.model.lower() else "fewshot"
    tok_src = "meta-llama/Llama-2-7b-hf" if "rog" in a.model.lower() else a.model
    tok = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16).to(device).eval()
    enc = encode_prefix_ids(tok, template)

    rows = []
    for ds_name in a.datasets:
        ds = load_ds(ds_name, a.n)
        pf = json.load(open(HERE / PATHS_FILE[ds_name]))
        conds = {"full": [], "noalign": [], "realign": [], "realign_rc": []}
        gold = []
        # one cache per condition (realign vs noalign keep separate caches to be safe)
        pc_re = HFPathCache(model, tok, device, encode_prefix_ids=enc)
        pc_na = HFPathCache(model, tok, device, encode_prefix_ids=enc)
        for i in range(len(ds)):
            s = ds[i]; q = s["question"]; gold.append(s["answer"])
            paths = [[tuple(t) for t in p] for p in pf.get(q, [])][:a.max_paths]
            conds["full"].append(full_generate(model, tok, q, paths, template,
                                                a.max_paths, a.max_new_tokens, device))
            os.environ["GROPT_NO_REALIGN"] = "1"
            conds["noalign"].append(generic_reuse_generate(model, tok, pc_na, q, paths, template,
                                                           a.max_paths, a.max_new_tokens, device, recompute_k=0))
            os.environ["GROPT_NO_REALIGN"] = "0"
            conds["realign"].append(generic_reuse_generate(model, tok, pc_re, q, paths, template,
                                                           a.max_paths, a.max_new_tokens, device, recompute_k=0))
            conds["realign_rc"].append(generic_reuse_generate(model, tok, pc_re, q, paths, template,
                                                             a.max_paths, a.max_new_tokens, device, recompute_k=a.recompute_k))
            if (i + 1) % 100 == 0:
                print(f"[{ds_name}] {i+1}/{len(ds)}", flush=True)
        os.environ.pop("GROPT_NO_REALIGN", None)
        for c, preds in conds.items():
            m = evaluate_metrics(preds, gold)
            rows.append({"dataset": ds_name, "condition": c, "n": len(gold),
                         "hits_at_1": m["hits_at_1"], "f1": m["f1"], "substr": m["accuracy"]})
            print(f"[{ds_name}] {c:11} Hits@1={m['hits_at_1']:.4f} F1={m['f1']:.4f}", flush=True)
        del pc_re, pc_na; torch.cuda.empty_cache()

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
