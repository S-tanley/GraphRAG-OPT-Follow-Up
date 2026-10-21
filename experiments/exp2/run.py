"""Run attention conditions for one (model, dataset) and save predictions per condition.

  python experiments/exp2/run.py --model rmanluo/RoG --dataset rmanluo/RoG-cwq \
      --conds M0,M1-hyb,M2-par --out_dir outputs/exp2
"""

import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "graphrag-main"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "common"))
from config import set_reproducible_seed            # noqa: E402
from faithful_pfkv import extract_answer            # noqa: E402
from table2_worker import load_ds                   # noqa: E402
from conditions import build                        # noqa: E402
from masked_engine import MaskedEngine              # noqa: E402

PATHS = {"rmanluo/RoG-cwq": "paths_rmanluo-RoG-cwq.json",
         "rmanluo/RoG-webqsp": "paths_rmanluo-RoG-webqsp.json"}


def first_line_done(text):
    """True once extract_answer(text) is fixed: some COMPLETE line (followed by a newline)
    is the line extract_answer would return. Mirrors faithful_pfkv.extract_answer."""
    for line in text.split("\n")[:-1]:
        line = line.strip()
        if not line or line.startswith("###"):
            continue
        for pre in ("the answer is:", "the answer is", "answer:", "answer is"):
            if line.lower().startswith(pre):
                line = line[len(pre):].strip(" .,")
        if line:
            return True
    return False


def load_model(name, lora=None):
    template = "rog" if "rog" in name.lower() else "fewshot"
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf" if template == "rog" else name)
    dtype = torch.bfloat16 if "gemma" in name.lower() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to("cuda").eval()
    if lora:                                   # merge a LoRA adapter (Exp3 remedy 4)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora).merge_and_unload().eval()
    return tok, model, template


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rmanluo/RoG")
    ap.add_argument("--dataset", default="rmanluo/RoG-cwq")
    ap.add_argument("--paths_dir", default=os.path.join(ROOT, "graphrag-main", "precomputed_paths_v2"))
    ap.add_argument("--conds", required=True)
    ap.add_argument("--n_samples", type=int, default=500)
    ap.add_argument("--max_paths", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--lora", default=None, help="LoRA adapter dir; its basename is added to the tag")
    args = ap.parse_args()

    set_reproducible_seed(42)
    ds = load_ds(args.dataset)
    if args.n_samples > 0:
        ds = ds.select(range(min(args.n_samples, len(ds))))
    pdict = json.load(open(os.path.join(args.paths_dir, PATHS[args.dataset])))
    tok, model, template = load_model(args.model, args.lora)
    eng = MaskedEngine(model)
    L = len(eng.layers)
    tag = f"{args.model.split('/')[-1]}_{args.dataset.split('-')[-1]}_cap{args.max_paths}"
    mname = args.model
    if args.lora:
        tag += "_" + os.path.basename(args.lora.rstrip("/"))
        mname += "+" + os.path.basename(args.lora.rstrip("/"))
    os.makedirs(args.out_dir, exist_ok=True)

    for cond in args.conds.split(","):
        out = os.path.join(args.out_dir, f"{tag}_{cond}.json")
        if os.path.exists(out):
            print("skip", out, flush=True)
            continue
        t0 = time.time()
        preds, gold, ntok, nrec = [], [], [], []
        for i in range(len(ds)):
            q = ds[i]["question"]
            paths = [[tuple(t) for t in p] for p in pdict.get(q, [])]
            ctx, suf, s0, npath = build(cond, tok, q, paths, template, args.max_paths)
            kw = {}
            if ctx.get("full_layers_top"):
                kw = dict(full_layers=set(range(L - ctx["full_layers_top"], L)), full_mask=ctx["full_mask"])
            if ctx.get("ctx_rows") is not None:
                kw["ctx_rows"] = ctx["ctx_rows"]
            stop = lambda ids: first_line_done(tok.decode(ids, skip_special_tokens=True))  # noqa: E731
            gen = eng.generate(ctx, suf, s0, args.max_new_tokens, tok.eos_token_id, stop_fn=stop, **kw)
            preds.append(extract_answer(tok.decode(gen, skip_special_tokens=True)))
            gold.append(ds[i]["answer"])
            ntok.append(npath)
            nrec.append(ctx.get("n_recompute", 0))
        json.dump({"cond": cond, "model": mname, "dataset": args.dataset,
                   "max_paths": args.max_paths, "answers": gold, "predictions": preds,
                   "path_tokens": ntok, "recompute_tokens": nrec, "questions": [ds[i]["question"] for i in range(len(ds))]},
                  open(out, "w"))
        print(f"[{tag}] {cond}: {len(preds)} done in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
