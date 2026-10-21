"""Exp3 LoRA comparison: base vs LoRA-M0 vs LoRA-M2 under each inference condition."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "graphrag-main"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "exp2"))
from analyze import mcnemar, per_q_hit  # noqa: E402

O = os.path.join(ROOT, "outputs")
cap = int(sys.argv[1]) if len(sys.argv) > 1 else 8


def path(model, d, c):
    if model == "base":
        return f"{O}/exp{2 if cap == 8 else 4}/RoG_{d}_cap{cap}_{c}.json"
    return f"{O}/exp3/RoG_{d}_cap{cap}_lora_RoG_{model}_{c}.json"


for d in ["cwq", "webqsp"]:
    print(f"\n=== RoG {d} cap{cap}  (Δ vs same-model M0; paired McNemar) ===")
    for model in ["base", "M0", "M2-hyb"]:
        name = {"base": "base", "M0": "LoRA-M0", "M2-hyb": "LoRA-M2"}[model]
        if not os.path.exists(path(model, d, "M0")):
            continue
        ref = per_q_hit(**{k: v for k, v in zip(["preds", "gold"], [None, None])}) if False else None
        r = json.load(open(path(model, d, "M0")))
        rh = per_q_hit(r["predictions"], r["answers"])
        cells = [f"M0 {sum(rh) / len(rh):.3f}"]
        for c in ["T", "M2-hyb", "M2-par", "M1-hyb"]:
            f = path(model, d, c)
            if not os.path.exists(f):
                continue
            x = json.load(open(f))
            h = per_q_hit(x["predictions"], x["answers"])
            up = sum(b > a for a, b in zip(rh, h))
            dn = sum(b < a for a, b in zip(rh, h))
            cells.append(f"{c} {sum(h) / len(h):.3f} ({(sum(h) - sum(rh)) / len(h):+.3f}, p={mcnemar(up, dn):.2f})")
        print(f"  {name:8s} " + " | ".join(cells))
