"""Arrow experiment summary: Hit, ΔHit vs M0 (paired McNemar), and recompute fraction.

CacheBlend / EPIC reference numbers come from GraphRAG-Opt's own ablation
(results/ablation_a1_template.txt: same 500 questions, cap 8, template prefix, manual
Llama engine whose full-prefill Hit matches our M0 within one question).
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "graphrag-main"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "exp2"))
from analyze import mcnemar, per_q_hit  # noqa: E402

O = os.path.join(ROOT, "outputs")
CB = {  # (model, dataset) -> {label: (Hit, recompute%)} from ablation_a1_template.txt
    ("RoG", "cwq"): {"full (ablation)": (0.510, 100), "CacheBlend 15%": (0.452, 14.9), "CacheBlend 30%": (0.468, 29.9), "EPIC first-2": (0.442, 7.3)},
    ("RoG", "webqsp"): {"full (ablation)": (0.760, 100), "CacheBlend 15%": (0.708, 14.9), "CacheBlend 30%": (0.712, 29.9), "EPIC first-2": (0.708, 8.8)},
    ("Llama-2-7b-hf", "cwq"): {"full (ablation)": (0.408, 100), "CacheBlend 15%": (0.316, 14.9), "CacheBlend 30%": (0.328, 29.9), "EPIC first-2": (0.296, 7.3)},
    ("Llama-2-7b-hf", "webqsp"): {"full (ablation)": (0.678, 100), "CacheBlend 15%": (0.604, 14.9), "CacheBlend 30%": (0.618, 29.9), "EPIC first-2": (0.592, 8.8)},
}
CONDS = [("exp2", "M0"), ("exp2", "T"), ("exp2", "M1-hyb"), ("exp2", "M2-hyb"),
         ("arrow", "A-pieces"), ("arrow", "A-rand"), ("arrow", "A-arrow"), ("arrow", "A-arrowlast")]

for m in ["RoG", "Llama-2-7b-hf"]:
    for d in ["cwq", "webqsp"]:
        f0 = f"{O}/exp2/{m}_{d}_cap8_M0.json"
        if not os.path.exists(f0):
            continue
        r = json.load(open(f0))
        rh = per_q_hit(r["predictions"], r["answers"])
        print(f"\n### {m} × {d}（cap 8，n={len(rh)}）")
        print("| 条件 | Hit | ΔHit vs M0 | 变好/变差 | p | 重算比例（路径 token） |")
        print("|---|---|---|---|---|---|")
        for sub, c in CONDS:
            f = f"{O}/{sub}/{m}_{d}_cap8_{c}.json"
            if not os.path.exists(f):
                continue
            x = json.load(open(f))
            h = per_q_hit(x["predictions"], x["answers"])
            up = sum(b > a for a, b in zip(rh, h))
            dn = sum(b < a for a, b in zip(rh, h))
            rec = x.get("recompute_tokens")
            rf = (f"{100 * sum(rec) / max(1, sum(x['path_tokens'])):.1f}%" if rec and sum(rec) > 0
                  else ("100%" if c in ("M0", "T") else "0%"))
            print(f"| {c} | {sum(h) / len(h):.3f} | {(sum(h) - sum(rh)) / len(h):+.3f} | {up}/{dn} | "
                  f"{mcnemar(up, dn):.3f} | {rf} |")
        for lab, (hit, rf) in CB[(m, d)].items():
            print(f"| {lab}（GraphRAG-Opt ablation） | {hit:.3f} | {hit - sum(rh) / len(rh):+.3f} | – | – | {rf}% |")
