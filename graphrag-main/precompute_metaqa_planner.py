"""
Precompute MetaQA retrieval using the FINE-TUNED planner (RoG transfer recipe):
the MetaQA-adapted planner generates relation sequences; we ground them with
exact-relation BFS on the FULL WikiMovies KG from the topic entity (RoG's
bfs_with_rule). Leak-free (no answer used). 8-GPU sharded.

  python precompute_metaqa_planner.py --planner RoG_metaqa_planner --num_gpus 8 \
      --out precomputed_paths_v2/paths_metaqa-3hop.json
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse_kb(kb_path):
    adj = defaultdict(list)
    rels = set()
    for line in open(kb_path, encoding="utf-8"):
        p = line.strip().split("|")
        if len(p) == 3:
            h, r, t = p[0].strip(), p[1].strip(), p[2].strip()
            adj[h].append((r, t)); adj[t].append((r, h)); rels.add(r)
    return adj, rels


def worker(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from config import set_reproducible_seed
    from graph_utils import _greedy_parse_rule, _bfs_with_rule
    from table2_worker import load_ds
    set_reproducible_seed(42)

    adj, rels = parse_kb(HERE / "metaQA" / "kb.txt")
    rel_sorted = sorted(rels, key=len, reverse=True)
    ds = load_ds(args.dataset)
    idxs = [i for i in range(len(ds)) if i % args.num_shards == args.shard_id]
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True)
    planner = AutoModelForCausalLM.from_pretrained(args.planner, dtype=torch.float16).to("cuda").eval()

    INSTR = ("Please generate a valid relation path that can be helpful for "
             "answering the following question: ")
    out = {}
    for k, i in enumerate(idxs):
        ex = ds[i]
        ids = tok(f"<s>[INST] {INSTR}{ex['question']} [/INST]", return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen = planner.generate(**ids, max_new_tokens=40, num_beams=args.num_beam,
                                   num_return_sequences=args.num_beam, do_sample=False,
                                   pad_token_id=tok.eos_token_id)
        rules = []
        for j in range(gen.shape[0]):
            txt = tok.decode(gen[j, ids["input_ids"].shape[1]:], skip_special_tokens=True)
            rule = _greedy_parse_rule(txt, rel_sorted, args.hop)
            if rule:
                rules.append(rule)
        paths, seen = [], set()
        topics = [t for t in ex["q_entity"] if t in adj]
        for rule in rules:
            for s in topics:
                for p in _bfs_with_rule(adj, s, rule):
                    key = tuple(p)
                    if p and key not in seen:
                        seen.add(key); paths.append(p)
            if len(paths) >= args.max_paths:
                break
        out[ex["question"]] = [[list(t) for t in p] for p in paths[:args.max_paths]]
        if (k + 1) % 200 == 0:
            print(f"[shard {args.shard_id}] {k+1}/{len(idxs)}", flush=True)
    json.dump(out, open(args.out_json, "w"))
    print(f"[shard {args.shard_id}] wrote {len(out)} -> {args.out_json}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="metaqa-3hop")
    ap.add_argument("--planner", default="RoG_metaqa_planner")
    ap.add_argument("--out", default="precomputed_paths_v2/paths_metaqa-3hop.json")
    ap.add_argument("--num_gpus", type=int, default=8)
    ap.add_argument("--hop", type=int, default=3)
    ap.add_argument("--max_paths", type=int, default=30)
    ap.add_argument("--num_beam", type=int, default=4)
    ap.add_argument("--shard_id", type=int, default=-1)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    if args.shard_id >= 0:
        worker(args); return

    tmp = tempfile.mkdtemp(prefix="precomp_mqaft_")
    procs = []
    for g in range(args.num_gpus):
        oj = os.path.join(tmp, f"s{g}.json")
        cmd = [sys.executable, str(HERE / "precompute_metaqa_planner.py"),
               "--dataset", args.dataset, "--planner", args.planner, "--hop", str(args.hop),
               "--max_paths", str(args.max_paths), "--num_beam", str(args.num_beam),
               "--shard_id", str(g), "--num_shards", str(args.num_gpus), "--out_json", oj]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), HF_HUB_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        log = open(os.path.join(tmp, f"s{g}.log"), "w")
        procs.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), oj))
    merged = {}
    for p, oj in procs:
        p.wait()
        if os.path.exists(oj):
            merged.update(json.load(open(oj)))
    json.dump(merged, open(args.out, "w"))
    print(f"Precomputed MetaQA (fine-tuned planner): {len(merged)} questions -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
