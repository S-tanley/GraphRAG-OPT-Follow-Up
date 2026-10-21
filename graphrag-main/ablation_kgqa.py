"""
End-task ablation worker: KGQA accuracy vs. KV-reuse recompute fraction.

Maps to the paper's methods (generation stage, identical retrieval):
    full        = GraphRAG-R   : full in-context prefill, no KV reuse (reference)
    none        = GraphRAG-Opt : RoPE-realign reuse only (the method as written)
    epic-k      = GraphRAG-Opt + EPIC first-k recompute
    cacheblend  = GraphRAG-Opt + CacheBlend top-r% recompute

Reuse is faithful (each path once, as cached KV at its correct position; no
double-representation) and GLOBAL (a path's isolated KV is computed once and
reused by every query containing it). Llama-family models only.

Single-GPU table:
    python ablation_kgqa.py --model rmanluo/RoG --dataset rmanluo/RoG-cwq --n_samples 64
Sharded worker (driven by run_full_ablation.py):
    python ablation_kgqa.py ... --shard_id 0 --num_shards 8 --out_json shard0.json
"""

import argparse
import json

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import set_reproducible_seed
from faithful_pfkv import faithful_generate, encode_prefix_ids
from metrics import evaluate_metrics
from retrieval import BatchOptimizedRetrieval, SimpleGraphRAGRetrieval
from selective_recompute import LlamaManualPrefill

LABELS = {
    "full": "GraphRAG-R (no reuse)",
    "none": "GraphRAG-Opt (reuse)",
    "epic1": "  + EPIC first-1",
    "epic2": "  + EPIC first-2",
    "epic4": "  + EPIC first-4",
    "cacheblend2": "  + CacheBlend 2%",
    "cacheblend4": "  + CacheBlend 4%",
    "cacheblend15": "  + CacheBlend 15%",
    "cacheblend30": "  + CacheBlend 30%",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rmanluo/RoG")
    ap.add_argument("--dataset", default="rmanluo/RoG-cwq")
    ap.add_argument("--n_samples", type=int, default=64, help="-1 = full test set")
    ap.add_argument("--hop", type=int, default=3)
    ap.add_argument("--max_paths", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--policies", default="full,none,epic1,epic2,cacheblend15")
    ap.add_argument("--path_enum", action="store_true")
    ap.add_argument("--paths_file", default=None,
                    help="precomputed LEAK-FREE RoG-plan paths {question: [[h,r,t],...]}. "
                         "When set, retrieval is the question-only RoG-plan retriever "
                         "(same as the main Table 2) instead of the answer-aware "
                         "q_entity->a_entity enumeration in retrieval.py. Use this for "
                         "the reported ablation so reuse fidelity is measured on the "
                         "same leakage-free paths as the headline accuracy table.")
    ap.add_argument("--template", default=None, choices=[None, "rog", "fewshot"])
    ap.add_argument("--encode_prefix", default="none", choices=["none", "template"],
                    help="lever #1: condition each path's cached KV on a shared "
                         "query-independent prefix (removes the attention sink)")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--cache_cap", type=int, default=3000, help="max cached paths before flush")
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--out_json", default=None, help="write shard predictions here")
    args = ap.parse_args()

    set_reproducible_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    template = args.template or ("rog" if "rog" in args.model.lower() else "fewshot")

    ds = load_dataset(args.dataset, split="test").shuffle(seed=args.seed)
    if args.n_samples and args.n_samples > 0:
        ds = ds.select(range(min(args.n_samples, len(ds))))
    if args.num_shards > 1:
        ds = ds.select([i for i in range(len(ds)) if i % args.num_shards == args.shard_id])
    tag = f"[shard {args.shard_id}/{args.num_shards}]" if args.num_shards > 1 else ""
    _retr = "RoG-plan(leak-free)" if args.paths_file else ("JbiDFS" if args.path_enum else "BFS")
    print(f"{tag} {args.model} x {args.dataset}: {len(ds)} samples, "
          f"retrieval={_retr}", flush=True)

    if args.paths_file:
        # LEAK-FREE: load precomputed question-only RoG-plan paths (the answer
        # entity is never used in retrieval), keyed by question text -- identical
        # provenance to the main Table 2 (table2_worker.py). The answer is used
        # only for scoring below.
        pdict = json.load(open(args.paths_file))
        results = []
        for i in range(len(ds)):
            ser = pdict.get(ds[i]["question"], [])
            paths = [[tuple(t) for t in p] for p in ser]
            results.append({"query": ds[i]["question"], "paths": paths})
        retr_tag = "RoG-plan(leak-free)"
    elif args.path_enum:
        results, _ = BatchOptimizedRetrieval(ds, device=dev).retrieve_batch(args.hop, True, args.max_paths)
        retr_tag = "JbiDFS"
    else:
        results, _ = SimpleGraphRAGRetrieval(ds).retrieve_batch(args.hop, args.max_paths)
        retr_tag = "BFS"
    answers = [ds[i]["answer"] for i in range(len(ds))]

    tok_src = "meta-llama/Llama-2-7b-hf" if "rog" in args.model.lower() else args.model
    tok = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    dt = torch.float16 if args.dtype == "float16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dt).to(dev).eval()
    engine = LlamaManualPrefill(model)
    eos = [tok.eos_token_id] if tok.eos_token_id is not None else []
    enc_prefix = encode_prefix_ids(tok, template) if args.encode_prefix == "template" else None

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    preds = {p: [] for p in policies}
    rec_frac = {p: [] for p in policies}
    gold = []
    path_cache = {}   # GLOBAL cross-query path-KV reuse (per replica)

    for qi, r in enumerate(results):
        if not r["paths"]:
            continue
        if len(path_cache) > args.cache_cap:
            path_cache.clear()
            torch.cuda.empty_cache()
        gold.append(answers[qi])
        for p in policies:
            ans, rf = faithful_generate(engine, tok, r["query"], r["paths"], policy=p,
                                        max_paths=args.max_paths,
                                        max_new_tokens=args.max_new_tokens,
                                        eos_ids=eos, template=template, path_cache=path_cache,
                                        encode_prefix=enc_prefix)
            preds[p].append(ans)
            rec_frac[p].append(rf)
        if (len(gold) % 50) == 0:
            print(f"{tag} {len(gold)} done", flush=True)

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"preds": preds, "rec_frac": rec_frac, "gold": gold,
                       "policies": policies}, f)
        print(f"{tag} wrote {len(gold)} samples -> {args.out_json}", flush=True)
        return

    print(f"\nModel={args.model}  Dataset={args.dataset}  "
          f"retrieval={_retr}  samples_used={len(gold)}")
    print(f"{'method':<26}{'recompute%':>11}{'Hits@1':>9}{'F1':>9}{'substr':>9}")
    print("-" * 64)
    for p in policies:
        m = evaluate_metrics(preds[p], gold)
        rf = 100 * sum(rec_frac[p]) / max(1, len(rec_frac[p]))
        print(f"{LABELS.get(p, p):<26}{rf:>10.1f}%{m['hits_at_1']:>9.3f}{m['f1']:>9.3f}{m['accuracy']:>9.3f}")


if __name__ == "__main__":
    main()
