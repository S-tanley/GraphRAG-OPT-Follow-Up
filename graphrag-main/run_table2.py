"""
Unified multi-GPU driver for the full Table 2. Runs all four columns through the
SAME worker (table2_worker.py) so every cell shares prompt build + path count +
greedy decode. Sharded across GPUs; aggregates per-shard predictions; scores
Hits@1/F1/substr; writes a TSV live.

    python run_table2.py --models rmanluo/RoG google/gemma-2-9b Qwen/Qwen3-8B \
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
# Table columns. graphrag_r = full-prefill retrieval number (== GraphRAG accuracy,
# JbiDFS is exact); opt = reuse-only (lossy); opt_rc = reuse + selective recompute
# (the operating point). graphrag_opt_rc is a pseudo-method that maps to the
# graphrag_opt worker method with recompute_k>0.
METHODS = ["direct", "graphrag_r", "graphrag_opt", "graphrag_opt_rc"]
LABEL = {"direct": "Direct", "graphrag": "GraphRAG",
         "graphrag_r": "GraphRAG-R", "graphrag_opt": "GraphRAG-Opt",
         "graphrag_opt_rc": "GraphRAG-Opt+RC"}
DLABEL = {"rmanluo/RoG-cwq": "CWQ", "rmanluo/RoG-webqsp": "WEB", "metaqa-3hop": "MQA"}


def run_cell(model, ds, method, num_gpus, max_paths, max_new, hop, outdir,
             gpus_per_shard=1, paths_dir=None, reuse_budget=4, recompute_k=2,
             n_samples=-1, load_4bit=False):
    safe = f"{model.replace('/','-')}_{DLABEL.get(ds, ds)}_{method}"
    # map pseudo-method -> worker method + recompute setting
    worker_method = "graphrag_opt" if method == "graphrag_opt_rc" else method
    rc = recompute_k if method == "graphrag_opt_rc" else 0
    paths_file = None
    if paths_dir:
        pf = os.path.join(paths_dir, f"paths_{ds.replace('/','-')}.json")
        if os.path.exists(pf):
            paths_file = pf
    n_shards = max(1, num_gpus // gpus_per_shard)
    procs = []
    for s in range(n_shards):
        oj = os.path.join(outdir, f"{safe}_s{s}.json")
        cmd = [sys.executable, os.path.join(HERE, "table2_worker.py"),
               "--model", model, "--dataset", ds, "--method", worker_method,
               "--n_samples", str(n_samples), "--max_paths", str(max_paths),
               "--max_new_tokens", str(max_new), "--hop", str(hop),
               "--reuse_budget", str(reuse_budget), "--recompute_k", str(rc),
               "--shard_id", str(s), "--num_shards", str(n_shards), "--out_json", oj]
        if paths_file:
            cmd += ["--paths_file", paths_file]
        if load_4bit:
            cmd += ["--load_4bit"]
        gpu_block = ",".join(str(s * gpus_per_shard + j) for j in range(gpus_per_shard))
        if gpus_per_shard > 1:
            cmd += ["--device_map", "auto"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_block,
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        log = open(os.path.join(outdir, f"{safe}_s{s}.log"), "w")
        procs.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), oj))
    for p, _ in procs:
        p.wait()
    ans, preds = [], []
    for _, oj in procs:
        if os.path.exists(oj):
            d = json.load(open(oj)); ans.extend(d["answers"]); preds.extend(d["predictions"])
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
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--num_gpus", type=int, default=8)
    ap.add_argument("--max_paths", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--hop", type=int, default=3)
    ap.add_argument("--paths_dir", default=None,
                    help="dir with precomputed paths_<dataset>.json (RoG-plan retrieval)")
    ap.add_argument("--gpus_per_shard", type=int, default=0,
                    help="GPUs per replica (0=auto by model size; >1 = model parallel)")
    ap.add_argument("--reuse_budget", type=int, default=4)
    ap.add_argument("--recompute_k", type=int, default=2,
                    help="EPIC first-k for the graphrag_opt_rc operating-point column")
    ap.add_argument("--n_samples", type=int, default=-1,
                    help="-1 = full test set; smaller for fast/subset runs")
    ap.add_argument("--results", default="/tmp/table2_unified.tsv")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--load_4bit", action="store_true",
                    help="NF4 4-bit single-GPU load (70B-class RoPE models) so reuse stays valid")
    args = ap.parse_args()

    outdir = args.outdir or tempfile.mkdtemp(prefix="t2u_")
    os.makedirs(outdir, exist_ok=True)
    if not os.path.exists(args.results):
        with open(args.results, "w") as f:
            f.write("model\tdataset\tmethod\tHits@1\tF1\tsubstr\tn\tsecs\n")

    def auto_gps(m):
        if args.gpus_per_shard > 0:
            return args.gpus_per_shard
        ml = m.lower()
        # 27B (54GB) and 32B (64GB) fit on one 80GB A100 -> single GPU. 70B/72B
        # (140GB bf16) and 80B need model parallel (device_map=auto). The KV-reuse
        # path is now DEVICE-AWARE (hf_reuse._layer_kv_device), so reuse works
        # model-parallel too -- a 72B GraphRAG-Opt number is obtainable this way.
        return 4 if any(s in ml for s in ("80b", "72b", "70b")) else 1

    for model in args.models:
        gps = auto_gps(model)
        for ds in args.datasets:
            for method in args.methods:
                t0 = time.time()
                res = run_cell(model, ds, method, args.num_gpus, args.max_paths,
                               args.max_new_tokens, args.hop, outdir,
                               gpus_per_shard=gps, paths_dir=args.paths_dir,
                               reuse_budget=args.reuse_budget, recompute_k=args.recompute_k,
                               n_samples=args.n_samples, load_4bit=args.load_4bit)
                secs = time.time() - t0
                if res is None:
                    line = f"{model}\t{DLABEL.get(ds, ds)}\t{LABEL[method]}\tFAILED\t-\t-\t0\t{secs:.0f}"
                else:
                    h, f1, sub, n = res
                    line = (f"{model}\t{DLABEL.get(ds, ds)}\t{LABEL[method]}\t"
                            f"{h:.4f}\t{f1:.4f}\t{sub:.4f}\t{n}\t{secs:.0f}")
                with open(args.results, "a") as f:
                    f.write(line + "\n")
                print(line, flush=True)
    print(f"\nDone -> {args.results}", flush=True)


if __name__ == "__main__":
    main()
