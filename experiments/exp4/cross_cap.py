"""Exp4: compare cap=30 conditions against the cap=8 M0 baseline (same 500 questions)."""
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
        ref = json.load(open(f"{O}/exp2/{m}_{d}_cap8_M0.json"))
        rh = per_q_hit(ref["predictions"], ref["answers"])
        rt = sum(ref["path_tokens"]) / len(rh)
        print(f"\n{m} {d}: reference M0 cap8 Hit {sum(rh) / len(rh):.3f}, path tokens {rt:.0f}")
        for cap, c in [(8, "T"), (8, "M2-hyb"), (30, "M0"), (30, "T"), (30, "M2-hyb"), (30, "M2-par")]:
            f = f"{O}/exp{2 if cap == 8 else 4}/{m}_{d}_cap{cap}_{c}.json"
            x = json.load(open(f))
            h = per_q_hit(x["predictions"], x["answers"])
            up = sum(b > a for a, b in zip(rh, h))
            dn = sum(b < a for a, b in zip(rh, h))
            t = sum(x["path_tokens"]) / len(h)
            print(f"  cap{cap:<3}{c:8s} Hit {sum(h) / len(h):.3f}  vs M0-cap8 {(sum(h) - sum(rh)) / len(h):+.3f} "
                  f"({up}/{dn}, p={mcnemar(up, dn):.3f})  path tokens {t:.0f} ({t / rt:.2f}x)")
