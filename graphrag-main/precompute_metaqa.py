"""
MetaQA-appropriate retrieval precompute (leak-free, schema-aware).

The RoG planner is Freebase-tuned and cannot ground WikiMovies' 9-relation
schema, and the dataset adapter's 500-node subgraph cap truncates the hop-3
neighborhood where 3-hop answers live (verified: answer is within 3 hops of the
topic 100% of the time in the FULL KG, but only ~35% recall under the capped
subgraph). This retriever therefore works from the FULL WikiMovies KG:

  1. enumerate simple paths up to `hop` hops from the topic entity (full KG),
  2. rank them by MiniLM cosine similarity between the question and the path's
     verbalized relation sequence (no answer entity used -> leak-free),
  3. keep the top `max_paths`, depth-`hop` paths preferred on ties.

This is the subgraph-expansion / dense-ranking retrieval family (DALK, GrapeQA,
ATLANTIC) the paper already covers, instantiated for a non-Freebase graph.

8-GPU sharded over the MiniLM encoder.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import deque, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse_kb(kb_path):
    adj = defaultdict(list)
    for line in open(kb_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        p = line.split("|")
        if len(p) == 3:
            h, r, t = p[0].strip(), p[1].strip(), p[2].strip()
            adj[h].append((r, t))
            adj[t].append((r, h))
    return adj


def enum_paths(adj, start, max_hop, hard_cap=4000):
    out = []
    q = deque([(start, [], {start})])
    while q and len(out) < hard_cap:
        node, path, seen = q.popleft()
        if path:
            out.append(path)
        if len(path) < max_hop:
            for rel, nb in adj.get(node, ()):
                if nb not in seen:
                    q.append((nb, path + [[node, rel, nb]], seen | {nb}))
    return out


def rel_text(path):
    return " ".join(t[1].replace("_", " ") for t in path)


def worker(args):
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from config import set_reproducible_seed
    from table2_worker import load_ds
    set_reproducible_seed(42)

    adj = parse_kb(HERE / "metaQA" / "kb.txt")
    ds = load_ds(args.dataset)
    idxs = [i for i in range(len(ds)) if i % args.num_shards == args.shard_id]
    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")

    out = {}
    for k, i in enumerate(idxs):
        ex = ds[i]
        starts = [s for s in ex["q_entity"] if s in adj]
        cand = []
        for s in starts:
            cand.extend(enum_paths(adj, s, args.hop))
        if not cand:
            out[ex["question"]] = []
            continue
        qv = emb.encode([ex["question"]], normalize_embeddings=True)[0]
        pv = emb.encode([rel_text(p) for p in cand], normalize_embeddings=True, batch_size=512)
        sims = pv @ qv
        order = list(np.argsort(-sims))
        d3 = [cand[j] for j in order if len(cand[j]) == args.hop]
        dl = [cand[j] for j in order if len(cand[j]) < args.hop]
        ranked = (d3 + dl)[:args.max_paths]
        out[ex["question"]] = [[list(t) for t in p] for p in ranked]
        if (k + 1) % 100 == 0:
            print(f"[shard {args.shard_id}] {k+1}/{len(idxs)}", flush=True)
    json.dump(out, open(args.out_json, "w"))
    print(f"[shard {args.shard_id}] wrote {len(out)} -> {args.out_json}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="metaqa-3hop")
    ap.add_argument("--out", default=None)
    ap.add_argument("--num_gpus", type=int, default=8)
    ap.add_argument("--hop", type=int, default=3)
    ap.add_argument("--max_paths", type=int, default=30)
    ap.add_argument("--shard_id", type=int, default=-1)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    if args.shard_id >= 0:
        worker(args)
        return

    out = args.out or f"precomputed_paths/paths_{args.dataset.replace('/','-')}.json"
    tmp = tempfile.mkdtemp(prefix="precomp_mqa_")
    procs = []
    for g in range(args.num_gpus):
        oj = os.path.join(tmp, f"s{g}.json")
        cmd = [sys.executable, str(HERE / "precompute_metaqa.py"),
               "--dataset", args.dataset, "--hop", str(args.hop),
               "--max_paths", str(args.max_paths),
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
    print(f"Precomputed MetaQA retrieval: {len(merged)} questions -> {out}", flush=True)


if __name__ == "__main__":
    main()
