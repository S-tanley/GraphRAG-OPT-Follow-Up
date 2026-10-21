"""Score the R1 reuse-fidelity ablation json shards into TSV rows."""
import glob
import json
import os
import re
from metrics import evaluate_metrics

ORCH = "/tmp/orch"
CFG = {  # tag -> (label, budget, recompute_k)
    "ceiling": ("GraphRAG-R (full prefill)", 0, "-"),
    "b2_rc0": ("Opt reuse-only", 2, 0),
    "b4_rc0": ("Opt reuse-only", 4, 0),
    "b8_rc0": ("Opt reuse-only", 8, 0),
    "b15_rc0": ("Opt reuse-only", 15, 0),
    "b4_rc1": ("Opt + EPIC first-1", 4, 1),
    "b4_rc2": ("Opt + EPIC first-2", 4, 2),
    "b8_rc2": ("Opt + EPIC first-2", 8, 2),
}
order = ["ceiling", "b2_rc0", "b4_rc0", "b8_rc0", "b15_rc0", "b4_rc1", "b4_rc2", "b8_rc2"]
for tag in order:
    f = os.path.join(ORCH, f"abl_{tag}.json")
    if not os.path.exists(f):
        continue
    d = json.load(open(f))
    m = evaluate_metrics(d["predictions"], d["answers"])
    label, b, rc = CFG[tag]
    print(f"{label}\t{b}\t{rc}\t{m['hits_at_1']:.4f}\t{m['f1']:.4f}\t{m['accuracy']:.4f}\t{len(d['answers'])}")
