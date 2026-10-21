#!/usr/bin/env python3
"""Reviewer C robustness figure: (A) within-batch path-redundancy SPECTRUM across
datasets and batching policies (diverse 'random' vs entity-clustered 'topical'),
(B) GRACEFUL DEGRADATION of the generation-side benefit (redundant prefill work
eliminated = 1/(1-f)) vs redundancy f -> 1x as overlap->0 (never < 1x), with the
redundancy-INDEPENDENT JbiDFS retrieval speedup annotated as the floor."""
import csv
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RED = ROOT / "results/redundancy.csv"
RET = ROOT / "results/retrieval_latency.csv"

DS = {"CWQ": ("#1f77b4", "Freebase / CWQ"), "WEB": ("#2ca02c", "Freebase / WebQSP"),
      "MQA": ("#d62728", "WikiMovies / MetaQA")}

red = defaultdict(dict)   # (ds,policy) -> {B: f}
for r in csv.DictReader(open(RED)):
    red[(r["dataset"], r["batching"])][int(r["batch_size"])] = float(r["path_redundancy"])

# JbiDFS retrieval speedup vs BFS in the cost-dominating multi-hop regime (hop 3-4),
# which is redundancy-INDEPENDENT. Report the per-dataset range across hops 3-4.
ret = defaultdict(list)
for r in csv.DictReader(open(RET)):
    if int(r["hop"]) in (3, 4):
        ds = {"rmanluo/RoG-cwq": "CWQ", "rmanluo/RoG-webqsp": "WEB", "metaqa-3hop": "MQA"}.get(r["dataset"])
        if ds:
            ret[ds].append(float(r["speedup_vs_bfs"]))

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6))

# Panel A: redundancy spectrum
for ds, (c, lab) in DS.items():
    for pol, ls, mk in [("topical", "-", "o"), ("random", "--", "s")]:
        pts = red.get((ds, pol), {})
        xs = sorted(pts); ys = [pts[x] * 100 for x in xs]
        axA.plot(xs, ys, ls, color=c, marker=mk, ms=4, lw=1.7,
                 label=f"{ds} ({pol})")
axA.set_xscale("log", base=2)
axA.set_xticks([1, 4, 16, 64, 256]); axA.set_xticklabels([1, 4, 16, 64, 256])
axA.set_xlabel("batch size"); axA.set_ylabel("within-batch path redundancy (%)")
axA.set_title("(a) Redundancy spectrum: diverse vs. entity-clustered batches")
axA.legend(fontsize=7, ncol=2); axA.grid(True, alpha=0.25)

# Panel B: graceful degradation curve
import numpy as np
fs = np.linspace(0, 0.45, 100)
axB.plot(fs * 100, 1 / (1 - fs), "k-", lw=2, label="generation benefit = 1/(1-f)\n(redundant prefill eliminated)")
# operating points (max topical redundancy per dataset)
for ds, (c, lab) in DS.items():
    f = max(red.get((ds, "topical"), {0: 0}).values())
    axB.plot(f * 100, 1 / (1 - f), "o", color=c, ms=10, mec="black")
    axB.annotate(f"{ds}\n{f*100:.0f}% -> {1/(1-f):.2f}x", (f * 100, 1 / (1 - f)),
                 textcoords="offset points", xytext=(6, -2), fontsize=8, color=c)
axB.axhline(1.0, ls="--", c="gray", lw=1)
axB.text(1, 1.02, "no-op floor (never < 1x: reuse degrades gracefully to a no-op)", fontsize=7.5, color="gray")
axB.set_xlabel("within-batch path redundancy f (%)")
axB.set_ylabel("generation-side benefit (x)")
axB.set_title("(b) Graceful degradation + redundancy-independent retrieval floor")
flo = ", ".join(f"{ds} {min(ret[ds]):.1f}-{max(ret[ds]):.1f}x" for ds in ["CWQ", "WEB", "MQA"])
axB.text(0.5, 1.55, f"JbiDFS RETRIEVAL speedup (hop 3-4) is redundancy-INDEPENDENT\n"
                    f"and holds even at zero overlap:  {flo}",
         fontsize=7.8, color="darkgreen",
         bbox=dict(boxstyle="round", fc="honeydew", ec="green", alpha=0.7))
axB.legend(fontsize=8, loc="lower right"); axB.grid(True, alpha=0.25)
axB.set_ylim(0.95, 1.85)

fig.suptitle("Robustness to query diversity / low redundancy (Reviewer C)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = HERE / "results/fig_robustness.pdf"
fig.savefig(out); fig.savefig(str(out).replace(".pdf", ".png"), dpi=140)
print(f"Wrote {out}")

# numeric summary
print("\nRedundancy spectrum (path %, topical / random):")
for ds in DS:
    t = max(red.get((ds, "topical"), {0: 0}).values()) * 100
    rnd = max(red.get((ds, "random"), {0: 0}).values()) * 100
    rr = ret.get(ds, [0])
    print(f"  {ds}: topical max {t:.1f}%  random max {rnd:.1f}%  -> gen benefit {1/(1-t/100):.2f}x ; JbiDFS retrieval {min(rr):.1f}-{max(rr):.1f}x (overlap-independent)")
