"""E5 prerequisite: how often does a retrieved path (or triple / relation chain) of a request already occur
in earlier requests (arrival order) or in Wtrain? That bounds the hit rate of a path-segment KV cache.
  .venv-mac/bin/python experiments/moe_edge/path_reuse.py
"""
import json, os
from datasets import load_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
train = [json.loads(l) for l in open(os.path.join(ROOT, "outputs/exp3/train_paths.jsonl"))]
tr_paths = {tuple(map(tuple, p)) for r in train for p in r["paths"][:8] if p}
tr_triples = {tuple(t) for r in train for p in r["paths"][:8] for t in p}
for name, f in [("cwq", "paths_rmanluo-RoG-cwq.json"), ("webqsp", "paths_rmanluo-RoG-webqsp.json")]:
    D = json.load(open(os.path.join(ROOT, "graphrag-main/precomputed_paths_v2", f)))
    ds = load_dataset(f"rmanluo/RoG-{name}", split="test").shuffle(seed=42)
    seen_p, seen_t = set(), set()
    n_p = hit_p = hit_pt = 0
    n_t = hit_t = hit_tt = 0
    for i in range(500):
        P = [tuple(map(tuple, p)) for p in D.get(ds[i]["question"], [])[:8] if p]
        for p in P:
            n_p += 1; hit_p += p in seen_p; hit_pt += (p in seen_p) or (p in tr_paths)
            for t in p:
                n_t += 1; hit_t += t in seen_t; hit_tt += (t in seen_t) or (t in tr_triples)
        seen_p.update(P); seen_t.update(t for p in P for t in p)
    print(f"{name}: {n_p} paths, seen in earlier test requests {hit_p / n_p:.1%}, earlier or Wtrain {hit_pt / n_p:.1%}; "
          f"{n_t} triples, seen earlier {hit_t / n_t:.1%}, earlier or Wtrain {hit_tt / n_t:.1%}")
