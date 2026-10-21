"""Score Exp0 prediction files and check exact-equality gates."""

import glob
import json
import os
import sys

sys.path.insert(0, os.environ.get("GROPT_SRC", "graphrag-main"))
from metrics import evaluate_metrics  # noqa: E402

outdir = sys.argv[1]
rows = {}
for f in sorted(glob.glob(os.path.join(outdir, "*.json"))):
    d = json.load(open(f))
    name = os.path.basename(f)[:-5]
    if "policies" in d:                       # ablation_kgqa format
        for p in d["policies"]:
            rows[f"{name}:{p}"] = (d["preds"][p], d["gold"])
    else:                                     # table2_worker / reuse_all_hf format
        rows[name] = (d["predictions"], d["answers"])

print(f"{'run':<40}{'n':>5}{'Hit':>8}{'F1':>8}")
for k, (p, g) in rows.items():
    m = evaluate_metrics(p, g)
    print(f"{k:<40}{len(p):>5}{m['hits_at_1']:>8.3f}{m['f1']:>8.3f}")


def agree(a, b):
    pa, pb = rows[a][0], rows[b][0]
    same = sum(x == y for x, y in zip(pa, pb))
    return f"{a} vs {b}: identical predictions {same}/{len(pa)}"


for a, b in [("t2_graphrag_r", "t2_opt_b0"), ("t2_opt_b4", "t2_opt_b4_nl"),
             ("t2_opt_b8", "t2_opt_b8_nl"), ("hf_reuseall", "hf_reuseall_nl")]:
    if a in rows and b in rows:
        print(agree(a, b))
