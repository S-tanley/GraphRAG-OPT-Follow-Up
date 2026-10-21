"""Exp0 helper: GraphRAG-Opt reuse-ALL through the HF engine (hf_reuse.generic_reuse_generate)
on the precomputed leak-free paths, so it can be compared with the manual-engine
ablation ('none' policy) and with/without GROPT_PATH_NEWLINE."""

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.environ.get("GROPT_SRC", "graphrag-main"))
from config import set_reproducible_seed                      # noqa: E402
from faithful_pfkv import encode_prefix_ids                   # noqa: E402
from hf_reuse import HFPathCache, generic_reuse_generate      # noqa: E402
from table2_worker import load_ds                             # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="rmanluo/RoG")
ap.add_argument("--dataset", default="rmanluo/RoG-cwq")
ap.add_argument("--paths_file", required=True)
ap.add_argument("--n_samples", type=int, default=500)
ap.add_argument("--max_paths", type=int, default=8)
ap.add_argument("--max_new_tokens", type=int, default=16)
ap.add_argument("--out_json", required=True)
args = ap.parse_args()

set_reproducible_seed(42)
ds = load_ds(args.dataset).select(range(args.n_samples))
pdict = json.load(open(args.paths_file))
template = "rog" if "rog" in args.model.lower() else "fewshot"
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf" if template == "rog" else args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to("cuda").eval()
pcache = HFPathCache(model, tok, "cuda", encode_prefix_ids=encode_prefix_ids(tok, template))

gold, preds = [], []
for i in range(len(ds)):
    paths = [[tuple(t) for t in p] for p in pdict.get(ds[i]["question"], [])]
    gold.append(ds[i]["answer"])
    preds.append(generic_reuse_generate(model, tok, pcache, ds[i]["question"], paths, template,
                                        args.max_paths, args.max_new_tokens, "cuda"))
json.dump({"answers": gold, "predictions": preds}, open(args.out_json, "w"))
print(f"wrote {len(preds)} -> {args.out_json}")
