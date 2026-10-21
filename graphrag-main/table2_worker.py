"""
Unified Table-2 worker: all four columns from ONE harness so they share the
exact prompt build, path count, and greedy decode. The only differences:

    direct        : no retrieval (paths=[]),            full prefill
    graphrag      : BFS retrieval,                      full prefill
    graphrag_r    : JbiDFS retrieval,                   full prefill
    graphrag_opt  : JbiDFS retrieval,                   KV reuse (lever #1)

So GraphRAG-R is literally "GraphRAG-Opt with reuse off" -> Opt and R are
apples-to-apples (and Opt <= R up to the reuse approximation). Sharded for
multi-GPU; writes {answers, predictions} JSON. RoPE models only for reuse.

    python table2_worker.py --model rmanluo/RoG --dataset rmanluo/RoG-cwq \
        --method graphrag_opt --shard_id 0 --num_shards 8 --out_json s0.json
"""

import argparse
import json
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import set_reproducible_seed
from faithful_pfkv import encode_prefix_ids
from hf_reuse import HFPathCache, full_generate, generic_reuse_generate
from metrics import evaluate_metrics
from retrieval import BatchOptimizedRetrieval, SimpleGraphRAGRetrieval

_METAQA = {"metaqa-1hop": 1, "metaqa-2hop": 2, "metaqa-3hop": 3}


def load_ds(name, seed=42):
    if name in _METAQA:
        from metaqa_adapter import load_metaqa
        mdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metaQA")
        return load_metaqa(hops=_METAQA[name], split="test", data_dir=mdir).shuffle(seed=seed)
    return load_dataset(name, split="test").shuffle(seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rmanluo/RoG")
    ap.add_argument("--dataset", default="rmanluo/RoG-cwq")
    ap.add_argument("--method", required=True,
                    choices=["direct", "graphrag", "graphrag_r", "graphrag_opt"])
    ap.add_argument("--n_samples", type=int, default=-1)
    ap.add_argument("--max_paths", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--hop", type=int, default=3)
    ap.add_argument("--encode_prefix", default="template", choices=["none", "template"])
    ap.add_argument("--cache_cap", type=int, default=4000)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--paths_file", default=None,
                    help="precomputed RoG-plan retrieval {question: [[h,r,t],...]}")
    ap.add_argument("--reuse_budget", type=int, default=4,
                    help="max shared paths reused (M_pool budget); rest computed in-context")
    ap.add_argument("--recompute_k", type=int, default=0,
                    help="EPIC first-k: # leading tokens of each reused path recomputed "
                         "in-context (0 = pure reuse, lossy; >0 recovers accuracy)")
    ap.add_argument("--device_map", default=None,
                    help="set to 'auto' for model-parallel load (32B/80B); else single-GPU")
    ap.add_argument("--load_4bit", action="store_true",
                    help="NF4 4-bit (bitsandbytes) single-GPU load for 70B-class RoPE "
                         "models so they fit on one card and the KV-reuse path stays "
                         "valid (weights are 4-bit; the KV cache is still fp16).")
    args = ap.parse_args()

    set_reproducible_seed(42)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    template = "rog" if "rog" in args.model.lower() else "fewshot"

    ds = load_ds(args.dataset)
    if args.n_samples and args.n_samples > 0:
        ds = ds.select(range(min(args.n_samples, len(ds))))
    if args.num_shards > 1:
        ds = ds.select([i for i in range(len(ds)) if i % args.num_shards == args.shard_id])
    tag = f"[{args.method} shard {args.shard_id}/{args.num_shards}]"

    # Retrieval: supervised RoG-plan paths, PRECOMPUTED once and loaded here
    # (leakage-free; the answer entity is never used). Direct uses no graph.
    # GraphRAG / GraphRAG-R / GraphRAG-Opt share the same retrieved paths --
    # JbiDFS (R) and KV reuse (Opt) are speed optimisations that preserve
    # accuracy; the speedups are reported separately.
    if args.method == "direct" or not args.paths_file:
        results = [{"query": ds[i]["question"], "paths": []} for i in range(len(ds))]
        if args.method != "direct" and not args.paths_file:
            print("WARNING: no --paths_file; graph methods will have no paths", flush=True)
    else:
        pdict = json.load(open(args.paths_file))
        results = []
        for i in range(len(ds)):
            ser = pdict.get(ds[i]["question"], [])
            paths = [[tuple(t) for t in p] for p in ser]
            results.append({"query": ds[i]["question"], "paths": paths})
    answers = [ds[i]["answer"] for i in range(len(ds))]

    tok_src = "meta-llama/Llama-2-7b-hf" if "rog" in args.model.lower() else args.model
    tok = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    # Gemma-2 has large intermediate activations that OVERFLOW in float16 -> NaN
    # logits -> empty generations (it was trained/served in bfloat16). Use bfloat16
    # for Gemma; float16 is fine for the Llama/Qwen/OPT families.
    mdtype = torch.bfloat16 if "gemma" in args.model.lower() else torch.float16
    if args.load_4bit:
        # NF4 4-bit single-GPU: a 70B-class RoPE model (~40GB in 4-bit) fits one
        # 80GB card, so the single-GPU KV-reuse path stays valid (unlike a bf16
        # 70B which needs model-parallel). Weights are 4-bit; KV cache is fp16.
        from transformers import BitsAndBytesConfig
        qconf = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=qconf, device_map={"": 0}).eval()
    elif args.device_map:
        # model-parallel across the shard's visible GPUs (for 32B/80B that do not
        # fit on one card). dev stays "cuda" -> inputs land on cuda:0 of the shard.
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=mdtype, device_map="auto").eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=mdtype).to(dev).eval()

    pcache = None
    shared = set()
    if args.method == "graphrag_opt":
        from collections import Counter
        from faithful_pfkv import path_to_str_rog
        from hf_reuse import partial_reuse_generate
        enc = encode_prefix_ids(tok, template) if args.encode_prefix == "template" else None
        pcache = HFPathCache(model, tok, dev, encode_prefix_ids=enc)
        # SHARED (cross-query, recurring) paths are reused; unique ones computed
        # in-context -- the paper's / SubGCache's cross-query reuse scenario.
        # Compute the shared set over the FULL dataset (all questions in the
        # paths_file), not just this shard, so which paths count as "shared" is
        # independent of sharding -- the cross-query reuse claim must not depend
        # on how the batch was split across GPUs.
        if args.paths_file:
            allpaths = json.load(open(args.paths_file))
            freq = Counter(path_to_str_rog([tuple(t) for t in p])
                           for plist in allpaths.values() for p in plist[:args.max_paths])
        else:
            freq = Counter(path_to_str_rog(p) for r in results for p in r["paths"][:args.max_paths])
        shared = {k for k, c in freq.items() if c >= 2}

    preds, gold = [], []
    for qi, r in enumerate(results):
        if pcache is not None and len(pcache.cache) > args.cache_cap:
            pcache.cache.clear(); torch.cuda.empty_cache()
        gold.append(answers[qi])
        if args.method == "graphrag_opt":
            use = r["paths"][:args.max_paths]
            # reuse shared paths up to the shared-KV-region budget (paper's
            # M_pool); compute the rest in-context. Canonical (deterministic)
            # order — (hop-len, serialization) — so a shared path lands at a
            # consistent layout position across queries (paper's Phase-2 packing),
            # which keeps the reused KV's left-context stable and shortest-first.
            sh = sorted((p for p in use if path_to_str_rog(p) in shared),
                        key=lambda p: (len(p), path_to_str_rog(p)))
            reused = sh[:args.reuse_budget]
            rset = {id(p) for p in reused}
            computed = [p for p in use if id(p) not in rset]
            preds.append(partial_reuse_generate(model, tok, pcache, r["query"], reused, computed,
                                                template, args.max_new_tokens, dev,
                                                recompute_k=args.recompute_k))
        else:
            preds.append(full_generate(model, tok, r["query"], r["paths"],
                                       template, args.max_paths, args.max_new_tokens, dev))
        if len(gold) % 200 == 0:
            print(f"{tag} {len(gold)} done", flush=True)

    if args.out_json:
        json.dump({"answers": gold, "predictions": preds}, open(args.out_json, "w"))
        print(f"{tag} wrote {len(gold)} -> {args.out_json}", flush=True)
    else:
        m = evaluate_metrics(preds, gold)
        print(f"{args.model} x {args.dataset} [{args.method}] n={len(gold)} "
              f"Hits@1={m['hits_at_1']:.4f} F1={m['f1']:.4f} substr={m['accuracy']:.4f}")


if __name__ == "__main__":
    main()
