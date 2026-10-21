"""
Multi-GPU driver for the GraphRAG-Opt (KV-reuse) column of Table 2, at full
dataset scale. Launches one hf_reuse.py worker per GPU on a dataset shard
(generic architecture-agnostic reuse, lever-#1 prefix-conditioning), aggregates
predictions, and scores. Writes results live.

    python run_graphrag_opt.py --models rmanluo/RoG google/gemma-2-9b Qwen/Qwen3-8B \
        --datasets rmanluo/RoG-cwq rmanluo/RoG-webqsp metaqa-3hop
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

from metrics import evaluate_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
DLABEL = {"rmanluo/RoG-cwq": "CWQ", "rmanluo/RoG-webqsp": "WEB", "metaqa-3hop": "MQA"}


def run_combo(model, ds, num_gpus, max_paths, max_new, hop, encode_prefix, outdir):
    safe = f"{model.replace('/','-')}_{DLABEL.get(ds, ds)}"
    procs = []
    for g in range(num_gpus):
        out_json = os.path.join(outdir, f"{safe}_shard{g}.json")
        cmd = [sys.executable, os.path.join(HERE, "hf_reuse.py"),
               "--model", model, "--dataset", ds, "--n_samples", "-1",
               "--max_paths", str(max_paths), "--max_new_tokens", str(max_new),
               "--hop", str(hop), "--encode_prefix", encode_prefix,
               "--shard_id", str(g), "--num_shards", str(num_gpus), "--out_json", out_json]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g),
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        log = open(os.path.join(outdir, f"{safe}_shard{g}.log"), "w")
        procs.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), out_json))
    for p, _ in procs:
        p.wait()
    ans, preds = [], []
    for _, oj in procs:
        if os.path.exists(oj):
            d = json.load(open(oj))
            ans.extend(d["answers"]); preds.extend(d["predictions"])
    if not preds:
        return None
    m = evaluate_metrics(preds, ans)
    return m["hits_at_1"], m["f1"], m["accuracy"], len(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["rmanluo/RoG", "meta-llama/Llama-2-7b-hf",
                             "google/gemma-2-9b", "Qwen/Qwen3-8B"])
    ap.add_argument("--datasets", nargs="+",
                    default=["rmanluo/RoG-cwq", "rmanluo/RoG-webqsp", "metaqa-3hop"])
    ap.add_argument("--num_gpus", type=int, default=8)
    ap.add_argument("--max_paths", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--hop", type=int, default=3)
    ap.add_argument("--encode_prefix", default="template", choices=["none", "template"])
    ap.add_argument("--results", default="/tmp/graphrag_opt.tsv")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    outdir = args.outdir or tempfile.mkdtemp(prefix="opt_")
    os.makedirs(outdir, exist_ok=True)
    if not os.path.exists(args.results):
        with open(args.results, "w") as f:
            f.write("model\tdataset\tmethod\tHits@1\tF1\tsubstr\tn\tsecs\n")

    for model in args.models:
        for ds in args.datasets:
            t0 = time.time()
            res = run_combo(model, ds, args.num_gpus, args.max_paths, args.max_new_tokens,
                            args.hop, args.encode_prefix, outdir)
            secs = time.time() - t0
            if res is None:
                line = f"{model}\t{DLABEL.get(ds, ds)}\tGraphRAG-Opt\tFAILED\t-\t-\t0\t{secs:.0f}"
            else:
                h, f1, sub, n = res
                line = (f"{model}\t{DLABEL.get(ds, ds)}\tGraphRAG-Opt\t"
                        f"{h:.4f}\t{f1:.4f}\t{sub:.4f}\t{n}\t{secs:.0f}")
            with open(args.results, "a") as f:
                f.write(line + "\n")
            print(line, flush=True)
    print(f"\nDone -> {args.results}", flush=True)


if __name__ == "__main__":
    main()
