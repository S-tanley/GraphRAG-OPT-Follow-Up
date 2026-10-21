"""Check that the offload engine computes exactly what the in-memory model computes (greedy tokens and
logits), on a model that fits in memory. Usage:
  .venv-mac/bin/python experiments/moe_edge/validate_offload.py --model mlx-community/Qwen3-30B-A3B-4bit \
      --store ~/mlx_offload/Qwen3-30B-A3B-4bit --slots 48
"""
import argparse, time
import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from offload_engine import OffloadEngine

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--store", required=True)
ap.add_argument("--slots", type=int, default=48); ap.add_argument("--policy", default="lru")
ap.add_argument("--la1", action="store_true")
args = ap.parse_args()
prompts = ["Q: Which country is Paris the capital of?\nA:", "The knowledge graph says: Barack Obama -> spouse -> Michelle Obama.\nQ: Who is Obama's wife?\nA:",
           "Explain in two sentences why the sky is blue.", "List three rivers in Europe:"]

def run(model, tok, n=24):
    outs, logits0 = [], []
    for p in prompts:
        ids = tok.encode(p); c = make_prompt_cache(model)
        lg = model(mx.array([ids]), cache=c)[:, -1]; logits0.append(np.array(lg.astype(mx.float32))); gen = []
        for _ in range(n):
            t = int(mx.argmax(lg, -1).item()); gen.append(t)
            lg = model(mx.array([[t]]), cache=c)[:, -1]
        outs.append(gen)
    return outs, logits0

m1, tok = load(args.model)
t = time.time(); ref, l_ref = run(m1, tok); t_ref = time.time() - t
del m1; mx.clear_cache()
m2, _ = load(args.model, lazy=True)
eng = OffloadEngine.attach(m2, args.store, args.slots, policy=args.policy, la1=args.la1)
print(f"after attach: active memory {mx.get_active_memory() / 1e9:.2f} GB (pools {args.slots} slots x {eng.L} layers)")
t = time.time(); off, l_off = run(m2, tok); t_off = time.time() - t
same = [a == b for a, b in zip(ref, off)]
dl = max(float(np.abs(a - b).max()) for a, b in zip(l_ref, l_off))
print(f"greedy tokens identical: {sum(same)}/{len(same)}; max |prefill logit diff| {dl:.4f}")
print(f"time: in-memory {t_ref:.1f}s, offload {t_off:.1f}s; stats {eng.stats}")
