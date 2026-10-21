"""
Precompute supervised RoG-plan retrieval ONCE per dataset (the RoG model plans
relation paths; grounded with bfs_with_rule). Saves {question: [[h,r,t],...]} so
every generation backbone reuses the SAME literature-standard retrieval (no
answer used). 8-GPU sharded.

Orchestrator:  python precompute_retrieval.py --dataset rmanluo/RoG-cwq --out paths_cwq.json
Worker (auto): ... --shard_id i --num_shards 8 --out_json shard_i.json
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def worker(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from config import set_reproducible_seed
    from graph_utils import get_paths_rog_plan
    from table2_worker import load_ds
    set_reproducible_seed(42)
    ds = load_ds(args.dataset)
    idxs = [i for i in range(len(ds)) if i % args.num_shards == args.shard_id]
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True)
    planner = AutoModelForCausalLM.from_pretrained("rmanluo/RoG", dtype=torch.float16).to("cuda").eval()
    out = {}
    for k, i in enumerate(idxs):
        ex = ds[i]
        paths = get_paths_rog_plan(planner, tok, ex["q_entity"], ex["graph"], ex["question"],
                                   args.hop, args.max_paths, num_beam=args.num_beam)
        out[ex["question"]] = [[list(t) for t in p] for p in paths]
        if (k + 1) % 100 == 0:
            print(f"[shard {args.shard_id}] {k+1}/{len(idxs)}", flush=True)
    json.dump(out, open(args.out_json, "w"))
    print(f"[shard {args.shard_id}] wrote {len(out)} -> {args.out_json}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--num_gpus", type=int, default=8)
    ap.add_argument("--hop", type=int, default=3)
    ap.add_argument("--max_paths", type=int, default=15)
    ap.add_argument("--num_beam", type=int, default=4)
    # worker mode
    ap.add_argument("--shard_id", type=int, default=-1)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    if args.shard_id >= 0:
        worker(args)
        return

    out = args.out or f"/tmp/paths_{args.dataset.replace('/','-')}.json"
    tmp = tempfile.mkdtemp(prefix="precomp_")
    procs = []
    for g in range(args.num_gpus):
        oj = os.path.join(tmp, f"s{g}.json")
        cmd = [sys.executable, os.path.join(HERE, "precompute_retrieval.py"),
               "--dataset", args.dataset, "--hop", str(args.hop),
               "--max_paths", str(args.max_paths), "--num_beam", str(args.num_beam),
               "--shard_id", str(g), "--num_shards", str(args.num_gpus), "--out_json", oj]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g),
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        log = open(os.path.join(tmp, f"s{g}.log"), "w")
        procs.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), oj))
    merged = {}
    for p, oj in procs:
        p.wait()
        if os.path.exists(oj):
            merged.update(json.load(open(oj)))
    json.dump(merged, open(out, "w"))
    print(f"Precomputed retrieval for {args.dataset}: {len(merged)} questions -> {out}", flush=True)


if __name__ == "__main__":
    main()
