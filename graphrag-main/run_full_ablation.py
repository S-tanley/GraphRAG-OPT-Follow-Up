"""
Multi-GPU driver for the KGQA recompute ablation.

Splits the test set across GPUs (one ablation_kgqa.py worker per GPU, each with
its own model replica and GLOBAL per-replica path-KV cache), waits, then
aggregates predictions and prints the GraphRAG-R / GraphRAG-Opt / +recompute
table.

Example (full CWQ on 8 GPUs, RoG):
    conda activate graphrag
    python run_full_ablation.py --model rmanluo/RoG --dataset rmanluo/RoG-cwq \
        --n_samples -1 --num_gpus 8 --path_enum
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

from metrics import evaluate_metrics

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
    ap.add_argument("--n_samples", type=int, default=-1, help="-1 = full test set")
    ap.add_argument("--num_gpus", type=int, default=8)
    ap.add_argument("--hop", type=int, default=3)
    ap.add_argument("--max_paths", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--policies", default="full,none,epic1,epic2,cacheblend15")
    ap.add_argument("--path_enum", action="store_true")
    ap.add_argument("--paths_dir", default=None,
                    help="dir with precomputed leak-free paths_<dataset>.json; when set, "
                         "the ablation uses the same question-only RoG-plan retrieval as "
                         "Table 2 (no answer leakage). Overrides --path_enum.")
    ap.add_argument("--template", default=None)
    ap.add_argument("--encode_prefix", default="none", choices=["none", "template"])
    ap.add_argument("--cache_cap", type=int, default=1500)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    outdir = args.outdir or tempfile.mkdtemp(prefix="abl_")
    os.makedirs(outdir, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    safe = (args.model + "_" + args.dataset).replace("/", "-")
    print(f"Launching {args.num_gpus} shard(s): {args.model} x {args.dataset} "
          f"(n={args.n_samples}); outputs -> {outdir}", flush=True)

    procs = []
    for g in range(args.num_gpus):
        out_json = os.path.join(outdir, f"{safe}_shard{g}.json")
        cmd = [sys.executable, os.path.join(here, "ablation_kgqa.py"),
               "--model", args.model, "--dataset", args.dataset,
               "--n_samples", str(args.n_samples), "--hop", str(args.hop),
               "--max_paths", str(args.max_paths), "--max_new_tokens", str(args.max_new_tokens),
               "--policies", args.policies, "--dtype", args.dtype,
               "--encode_prefix", args.encode_prefix, "--cache_cap", str(args.cache_cap),
               "--shard_id", str(g), "--num_shards", str(args.num_gpus),
               "--out_json", out_json]
        paths_file = None
        if args.paths_dir:
            pf = os.path.join(args.paths_dir, f"paths_{args.dataset.replace('/','-')}.json")
            if os.path.exists(pf):
                paths_file = pf
        if paths_file:
            cmd += ["--paths_file", paths_file]
        elif args.path_enum:
            cmd.append("--path_enum")
        if args.template:
            cmd += ["--template", args.template]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g),
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        log = open(os.path.join(outdir, f"{safe}_shard{g}.log"), "w")
        procs.append((g, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), out_json))

    t0 = time.time()
    failed = []
    for g, p, _ in procs:
        if p.wait() != 0:
            failed.append(g)
    print(f"All shards finished in {time.time()-t0:.0f}s "
          f"({'OK' if not failed else 'FAILED shards: ' + str(failed)})", flush=True)

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    agg_preds = {p: [] for p in policies}
    agg_rec = {p: [] for p in policies}
    agg_gold = []
    for g, _, out_json in procs:
        if not os.path.exists(out_json):
            print(f"WARNING: shard {g} produced no output", flush=True)
            continue
        with open(out_json) as f:
            d = json.load(f)
        agg_gold.extend(d["gold"])
        for p in policies:
            agg_preds[p].extend(d["preds"].get(p, []))
            agg_rec[p].extend(d["rec_frac"].get(p, []))

    _retr = "RoG-plan(leak-free)" if args.paths_dir else ("JbiDFS" if args.path_enum else "BFS")
    print(f"\n===== {args.model}  x  {args.dataset} "
          f"({_retr})  samples_used={len(agg_gold)} =====")
    print(f"{'method':<26}{'recompute%':>11}{'Hits@1':>9}{'F1':>9}{'substr':>9}")
    print("-" * 64)
    for p in policies:
        m = evaluate_metrics(agg_preds[p], agg_gold)
        rf = 100 * sum(agg_rec[p]) / max(1, len(agg_rec[p]))
        print(f"{LABELS.get(p, p):<26}{rf:>10.1f}%{m['hits_at_1']:>9.3f}{m['f1']:>9.3f}{m['accuracy']:>9.3f}")


if __name__ == "__main__":
    main()
