"""Fair A-1 baseline: run ALL reuse policies in ONE harness at the SAME scope.

For each model x dataset (N=500, leak-free RoG-plan paths, encode_prefix=template):
  - full                        : full in-context prefill (reference)
  - sel:none                    : deployed GraphRAG-Opt -- reuse ONLY recurring
                                  shared paths (cross-query count r>1), compute
                                  one-off paths in-context, 0% token recompute
  - sel:cacheblend15/30, epicX  : same selective scope + token recompute on the
                                  reused paths (CacheBlend / EPIC selection)
  - all:none/cacheblend/epic    : reuse-EVERYTHING regime (CacheBlend/EPIC native
                                  port: reuse all paths, recompute a token fraction)

The shared set = {path : global count >= 2} reproduces table2_worker's
`shared = {k for k,c in freq.items() if c>=2}` so sel:none should match the
deployed Table-2 GraphRAG-Opt accuracy.
"""
import argparse, json, collections, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import set_reproducible_seed
from faithful_pfkv import faithful_generate, encode_prefix_ids, build_prompt
from selective_recompute import LlamaManualPrefill
from metrics import evaluate_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="rmanluo/RoG")
ap.add_argument("--dataset", default="rmanluo/RoG-cwq")
ap.add_argument("--paths_file", required=True)
ap.add_argument("--n_samples", type=int, default=500)
ap.add_argument("--max_paths", type=int, default=8)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

set_reproducible_seed(args.seed)
dev = "cuda"
template = "rog" if "rog" in args.model.lower() else "fewshot"
ds = load_dataset(args.dataset, split="test").shuffle(seed=args.seed)
if args.n_samples > 0:
    ds = ds.select(range(min(args.n_samples, len(ds))))
pdict = json.load(open(args.paths_file))
results, answers = [], []
for i in range(len(ds)):
    ser = pdict.get(ds[i]["question"], [])
    results.append({"query": ds[i]["question"],
                    "paths": [[tuple(t) for t in p] for p in ser]})
    answers.append(ds[i]["answer"])

tok_src = "meta-llama/Llama-2-7b-hf" if "rog" in args.model.lower() else args.model
tok = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(dev).eval()
engine = LlamaManualPrefill(model)
eos = [tok.eos_token_id] if tok.eos_token_id is not None else []
enc_prefix = encode_prefix_ids(tok, template)

# --- global path frequency over the whole set -> shared (r>1) set, as in table2_worker ---
freq = collections.Counter()
for r in results:
    if not r["paths"]:
        continue
    _, _, _, pstrs = build_prompt(tok, r["query"], r["paths"], args.max_paths, template)
    freq.update(pstrs)
reuse_set = {k for k, c in freq.items() if c >= 2}
n_shared = len(reuse_set)
n_total = len(freq)
print(f"{args.model} x {args.dataset}: {n_shared}/{n_total} distinct paths recur (r>1) "
      f"= {100*n_shared/max(1,n_total):.1f}%", flush=True)

# policy_name -> (faithful policy, selective?)
RUN = [
    ("full",            "full",         False),
    ("sel:none",        "none",         True),
    ("sel:epic1",       "epic1",        True),
    ("sel:cacheblend15","cacheblend15", True),
    ("sel:cacheblend30","cacheblend30", True),
    ("all:none",        "none",         False),
    ("all:cacheblend15","cacheblend15", False),
    ("all:cacheblend30","cacheblend30", False),
]
preds = {name: [] for name, _, _ in RUN}
gold = []
# The isolated path-KV is policy-independent and deterministic, so reuse accuracy
# depends only on the reuse_set / recompute selection, not on whether the KV came
# from a persistent cross-query cache. We re-encode per query (path_cache=None) to
# keep VRAM bounded; the deployed Table-2 harness uses the persistent cache purely
# for the throughput measurement.
for qi, r in enumerate(results):
    if not r["paths"]:
        continue
    gold.append(answers[qi])
    for name, pol, sel in RUN:
        rs = reuse_set if sel else None
        ans, _ = faithful_generate(engine, tok, r["query"], r["paths"], policy=pol,
                                   max_paths=args.max_paths, max_new_tokens=16,
                                   eos_ids=eos, template=template,
                                   path_cache=None, encode_prefix=enc_prefix,
                                   reuse_set=rs)
        preds[name].append(ans)
        torch.cuda.empty_cache()
    if len(gold) % 100 == 0:
        print(f"  {len(gold)} done", flush=True)

print(f"\n=== {args.model} x {args.dataset}  (N={len(gold)}, encode_prefix=template) ===")
print(f"{'policy':<20}{'Hits@1':>9}{'F1':>9}")
for name, _, _ in RUN:
    m = evaluate_metrics(preds[name], gold)
    print(f"{name:<20}{m['hits_at_1']:>9.3f}{m['f1']:>9.3f}", flush=True)
