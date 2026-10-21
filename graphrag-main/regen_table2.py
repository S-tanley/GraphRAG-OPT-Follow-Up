"""
Regenerate Table 2 (accuracy) at FULL dataset scale — multi-GPU via sharding.

For each model x dataset x method, launches one single-GPU *batched* main.py
worker per GPU on a dataset shard (fast true GPU batching), aggregates the
per-shard predictions, and scores Hits@1 / F1 / substr on the full set. Writes
results live to a TSV. Replaces the 64-sample subset numbers and the duplicated
cells flagged by the reviewer.

Columns (full-prefill; GraphRAG-Opt reuse reported separately):
    Direct LLM   : no retrieval
    GraphRAG     : BFS retrieval + full prefill
    GraphRAG-R   : JbiDFS retrieval + full prefill

    python regen_table2.py --models rmanluo/RoG --datasets rmanluo/RoG-cwq
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

METHODS = {
    "Direct":     [],
    "GraphRAG":   ["--retrieve"],
    "GraphRAG-R": ["--retrieve", "--path_enum"],
}
DLABEL = {"rmanluo/RoG-cwq": "CWQ", "rmanluo/RoG-webqsp": "WEB", "metaqa-3hop": "MQA"}


def run_combo(model, ds, method, flags, num_gpus, beam, max_new, hop, outdir,
              gpus_per_shard=1):
    safe = f"{model.replace('/','-')}_{DLABEL.get(ds, ds)}_{method}"
    # Paths make prompts long; batch 16 keeps KV (x beam) within memory.
    batch = "16" if flags else "48"
    n_shards = max(1, num_gpus // gpus_per_shard)
    procs = []
    for s in range(n_shards):
        gpu_block = ",".join(str(s * gpus_per_shard + j) for j in range(gpus_per_shard))
        out_preds = os.path.join(outdir, f"{safe}_shard{s}.json")
        cmd = [sys.executable, os.path.join(HERE, "main.py"),
               "--dataset", ds, "--llm_model", model,
               "--test_size", "1000000", "--batch_num", batch,
               "--num_beams", str(beam), "--max_new_tokens", str(max_new),
               "--hop_constraint", str(hop),
               "--shard_id", str(s), "--num_shards", str(n_shards),
               "--out_preds", out_preds, *flags]
        if gpus_per_shard > 1:
            cmd += ["--device_map", "auto"]   # model parallel within the GPU block
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_block,
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        log = open(os.path.join(outdir, f"{safe}_shard{s}.log"), "w")
        procs.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), out_preds))

    for p, _ in procs:
        p.wait()

    answers, preds = [], []
    for _, out_preds in procs:
        if os.path.exists(out_preds):
            d = json.load(open(out_preds))
            answers.extend(d["answers"])
            preds.extend(d["predictions"])
    if not preds:
        return None
    m = evaluate_metrics(preds, answers)
    return m["hits_at_1"], m["f1"], m["accuracy"], len(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["rmanluo/RoG", "meta-llama/Llama-2-7b-hf",
                             "google/gemma-2-9b", "Qwen/Qwen3-8B"])
    ap.add_argument("--datasets", nargs="+",
                    default=["rmanluo/RoG-cwq", "rmanluo/RoG-webqsp", "metaqa-3hop"])
    ap.add_argument("--methods", nargs="+", default=list(METHODS.keys()))
    ap.add_argument("--num_gpus", type=int, default=8)
    ap.add_argument("--num_beams", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=50)
    ap.add_argument("--hop", type=int, default=3)
    ap.add_argument("--gpus_per_shard", type=int, default=0,
                    help="GPUs per model replica (0=auto by model size). >1 enables "
                         "model parallelism for models too big for one GPU.")
    ap.add_argument("--results", default="/tmp/table2_regen.tsv")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    outdir = args.outdir or tempfile.mkdtemp(prefix="t2_")
    os.makedirs(outdir, exist_ok=True)
    if not os.path.exists(args.results):
        with open(args.results, "w") as f:
            f.write("model\tdataset\tmethod\tHits@1\tF1\tsubstr\tn\tsecs\n")

    # GPUs per replica: auto by model size (fp16 footprint vs 80GB), unless overridden.
    def auto_gps(m):
        if args.gpus_per_shard > 0:
            return args.gpus_per_shard
        ml = m.lower()
        if "80b" in ml:
            return 4
        if "32b" in ml or "30b" in ml:
            return 2
        return 1

    for model in args.models:
        gps = auto_gps(model)
        for ds in args.datasets:
            for method in args.methods:
                t0 = time.time()
                res = run_combo(model, ds, method, METHODS[method], args.num_gpus,
                                args.num_beams, args.max_new_tokens, args.hop, outdir,
                                gpus_per_shard=gps)
                secs = time.time() - t0
                if res is None:
                    line = f"{model}\t{DLABEL.get(ds, ds)}\t{method}\tFAILED\t-\t-\t0\t{secs:.0f}"
                else:
                    h, f1, sub, n = res
                    line = (f"{model}\t{DLABEL.get(ds, ds)}\t{method}\t"
                            f"{h:.4f}\t{f1:.4f}\t{sub:.4f}\t{n}\t{secs:.0f}")
                with open(args.results, "a") as f:
                    f.write(line + "\n")
                print(line, flush=True)

    print(f"\nDone -> {args.results}  (logs: {outdir})", flush=True)


if __name__ == "__main__":
    main()
