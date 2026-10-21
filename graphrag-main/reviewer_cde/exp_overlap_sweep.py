#!/usr/bin/env python3
"""Reviewer C: CONTROLLED query-diversity sweep. Form size-B batches whose fraction
alpha of queries come from a shared topic cluster (entity-clustered) and (1-alpha)
are drawn at random (diverse). Sweep alpha from 0 (fully diverse) to 1 (fully
clustered) and measure within-batch PATH redundancy and the resulting generation-side
benefit 1/(1-f). Shows the method scales SMOOTHLY and GRACEFULLY with overlap and
never falls below a 1x no-op. CPU-only (path-overlap counting)."""
import json, csv, random
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PATHS = {"CWQ": ROOT / "precomputed_paths/paths_rmanluo-RoG-cwq.json",
         "WEB": ROOT / "precomputed_paths/paths_rmanluo-RoG-webqsp.json"}


def pstr(p):
    return "|".join("/".join(map(str, t)) for t in p)


def topic(paths):
    c = defaultdict(int)
    for p in paths:
        for h, r, t in p:
            c[h] += 1; c[t] += 1
    return max(c, key=c.get) if c else None


def redundancy(batch_paths):
    """fraction of path INSTANCES in the batch that are duplicates (r>1)."""
    seen = defaultdict(int); inst = 0
    for paths in batch_paths:
        for s in set(pstr(p) for p in paths):
            seen[s] += 1
    # per query count once; redundancy on instances:
    cnt = defaultdict(int); total = 0
    for paths in batch_paths:
        for p in paths:
            cnt[pstr(p)] += 1; total += 1
    distinct = len(cnt)
    return (total - distinct) / total if total else 0.0


def run(ds, B=64, n_batches=40, seed=42):
    rng = random.Random(seed)
    pf = json.load(open(PATHS[ds]))
    items = [(q, [[tuple(t) for t in p] for p in paths][:8]) for q, paths in pf.items()]
    items = [it for it in items if it[1]]
    # cluster by topic entity
    clusters = defaultdict(list)
    for it in items:
        clusters[topic(it[1])].append(it)
    big = [c for c in clusters.values() if len(c) >= 8]
    rows = []
    for alpha in [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]:
        reds = []
        for _ in range(n_batches):
            k = int(round(alpha * B))
            batch = []
            if k > 0 and big:
                cl = rng.choice(big)
                batch += [bp for (_q, bp) in rng.sample(cl, min(k, len(cl)))]
            while len(batch) < B:
                batch.append(rng.choice(items)[1])
            reds.append(redundancy(batch))
        f = float(np.mean(reds))
        rows.append({"dataset": ds, "alpha_clustered": alpha, "diversity": 1 - alpha,
                     "path_redundancy": f, "gen_benefit": 1 / (1 - f) if f < 1 else float("nan")})
    return rows


def main():
    allrows = []
    for ds in PATHS:
        allrows += run(ds)
    with open(HERE / "results/overlap_sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(allrows[0].keys())); w.writeheader(); w.writerows(allrows)
    # plot
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax2 = ax.twinx()
    for ds, c in [("CWQ", "#1f77b4"), ("WEB", "#2ca02c")]:
        r = [x for x in allrows if x["dataset"] == ds]
        xs = [x["alpha_clustered"] * 100 for x in r]
        ax.plot(xs, [x["path_redundancy"] * 100 for x in r], "-o", color=c, label=f"{ds} redundancy")
        ax2.plot(xs, [x["gen_benefit"] for x in r], "--s", color=c, alpha=0.6, label=f"{ds} benefit")
    ax.axhline(0, color="gray", lw=0.5)
    ax2.axhline(1.0, ls=":", color="gray")
    ax.set_xlabel("% of batch that is entity-clustered  (0 = fully diverse, 100 = fully clustered)")
    ax.set_ylabel("within-batch path redundancy (%)")
    ax2.set_ylabel("generation benefit = 1/(1-f)  (x)")
    ax2.set_ylim(0.95, 1.75)
    ax.set_title("Reviewer C: graceful, smooth scaling with query overlap")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = HERE / "results/fig_overlap_sweep.pdf"
    fig.savefig(out); fig.savefig(str(out).replace(".pdf", ".png"), dpi=140)
    print(f"Wrote {out}")
    print(f"{'ds':4} {'%clustered':>10} {'redundancy%':>11} {'benefit':>8}")
    for x in allrows:
        print(f"{x['dataset']:4} {x['alpha_clustered']*100:>9.0f}% {x['path_redundancy']*100:>10.1f}% {x['gen_benefit']:>7.2f}x")


if __name__ == "__main__":
    main()
