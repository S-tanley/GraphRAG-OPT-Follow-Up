"""Gate re-implementation check for collect_traces.GateBank.

Self-consistency: routing MoE layer L's OWN input with GateBank.route(L, .) must reproduce the
experts the model actually used at layer L (same top-k set), for every token. Also reports the
one-layer-ahead hit rate (|lookahead ∩ actual| / k), which is the Mixtral-offloading/HOBBIT signal.

  python experiments/moe/test_gates.py --model zai-org/GLM-4.5-Air --max_memory 0:130GiB,1:130GiB,cpu:1000GiB
"""

import argparse
import os
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_traces import GateBank, Recorder  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--max_memory", required=True)
args = ap.parse_args()
mm = {(int(k) if k.isdigit() else k): v for k, v in (kv.split(":") for kv in args.max_memory.split(","))}
tok = AutoTokenizer.from_pretrained(args.model)
cfg = AutoConfig.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto", device_map="auto", max_memory=mm).eval()
rec = Recorder(model, lambda ids: GateBank(args.model, cfg, ids))

# extra hooks: capture each MoE block's own input to test self-consistency
inputs = {}
name2mod = dict(model.named_modules())
for li, name in enumerate(rec.layers):
    def h(mod, a, kw, li=li):
        x = a[0] if a else kw.get("hidden_states")
        inputs[li] = x.reshape(-1, x.shape[-1]).detach()
    name2mod[name.rsplit(".", 1)[0]].register_forward_pre_hook(h, with_kwargs=True)

text = ("Q: What is the capital of France?\nA: Paris\n\nContext (knowledge graph paths):\n"
        "New York -> location.location.containedby -> Mid-Atlantic states\n"
        "Q: Where is the location that includes Washington D.C. and New York?\nA:")
ids = tok(text, return_tensors="pt").input_ids.to(next(model.parameters()).device)
rec.start()
with torch.no_grad():
    model(input_ids=ids)
routes, la = rec.take()
self_ok, la_hit, tot = [], [], 0
for li in range(len(rec.layers)):
    actual = routes[li][0].long()
    mine = rec.gates.route(rec.layer_ids[li], inputs[li]).cpu().long()
    a, m = actual.sort(-1).values, mine.sort(-1).values
    self_ok.append((a == m).all(-1).float().mean().item())
    if li + 1 < len(rec.layers) and la[li] is not None:
        nxt = routes[li + 1][0].long()
        hit = [len(set(x.tolist()) & set(y.tolist())) / nxt.shape[-1] for x, y in zip(la[li].long(), nxt)]
        la_hit.append(sum(hit) / len(hit))
print(f"{args.model}: tokens={ids.shape[1]} layers={len(rec.layers)}")
print(f"  self-consistency (exact top-k set): mean {sum(self_ok) / len(self_ok):.4f}, min {min(self_ok):.4f}")
print(f"  one-layer-ahead hit rate: mean {sum(la_hit) / max(1, len(la_hit)):.3f}  "
      f"(first 5 layers {[round(x, 2) for x in la_hit[:5]]}, last 5 {[round(x, 2) for x in la_hit[-5:]]})")
