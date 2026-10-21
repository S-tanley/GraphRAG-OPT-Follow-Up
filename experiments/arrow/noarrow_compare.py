"""A-noarrow (only the arrows use cached, context-free KV) vs M0."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "graphrag-main"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "exp2"))
from analyze import mcnemar, per_q_hit  # noqa: E402

O = os.path.join(ROOT, "outputs")
for m in ["RoG", "Llama-2-7b-hf"]:
    for d in ["cwq", "webqsp"]:
        r = json.load(open(f"{O}/exp2/{m}_{d}_cap8_M0.json"))
        x = json.load(open(f"{O}/arrow/{m}_{d}_cap8_A-noarrow.json"))
        rh, h = per_q_hit(r["predictions"], r["answers"]), per_q_hit(x["predictions"], x["answers"])
        up = sum(b > a for a, b in zip(rh, h))
        dn = sum(b < a for a, b in zip(rh, h))
        same = sum(a == b for a, b in zip(r["predictions"], x["predictions"]))
        cached = 1 - sum(x["recompute_tokens"]) / sum(x["path_tokens"])
        print(f"{m} {d}: M0 {sum(rh) / len(rh):.3f}  A-noarrow {sum(h) / len(h):.3f}  "
              f"delta {(sum(h) - sum(rh)) / len(h):+.3f} ({up}/{dn}, p={mcnemar(up, dn):.3f})  "
              f"identical {same}/{len(h)}  cached share of path tokens {cached:.3f}")
