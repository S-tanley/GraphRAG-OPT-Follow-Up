#!/usr/bin/env python3
"""
plot_reviewer_b.py — figures for the Reviewer-B rebuttal.

Generates, from the result CSVs in results/:
  fig_nonrog_hop.pdf      : Q2  non-RoG (networkx) path-enum scaling vs hop depth
  fig_batch_sweep.pdf     : Q3  TTFT/throughput speedup vs batch size (decomposed)
  fig_subgcache.pdf       : Q1a SubGCache-contract vs ours (TTFT/accuracy)
Each plotter is guarded by file existence so the script runs as data lands.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
R = HERE / "results"
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
DSCOLOR = {"rmanluo/RoG-cwq": "#c0392b", "rmanluo/RoG-webqsp": "#2980b9",
           "metaqa-3hop": "#27ae60"}
DSLABEL = {"rmanluo/RoG-cwq": "CWQ (Freebase)", "rmanluo/RoG-webqsp": "WebQSP (Freebase)",
           "metaqa-3hop": "MetaQA (WikiMovies)"}


def plot_nonrog_hop():
    f = R / "nonrog_hop_scaling_summary.csv"
    if not f.exists():
        print("skip nonrog_hop (no data)"); return
    df = pd.read_csv(f)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for ds in df.dataset.unique():
        s = df[df.dataset == ds].sort_values("hop")
        c = DSCOLOR.get(ds, "#555")
        ax[0].plot(s.hop, s.nx_paths_mean, "o-", color=c, label=DSLABEL.get(ds, ds))
        ax[1].plot(s.hop, s.nx_ms_mean, "o-", color=c, label=DSLABEL.get(ds, ds))
    for a, ylab, title in ((ax[0], "mean # paths / query", "Path count"),
                           (ax[1], "mean retrieval latency (ms)", "Enumeration latency")):
        a.set_yscale("log"); a.set_xlabel("hop bound $k$"); a.set_ylabel(ylab)
        a.set_title(title); a.set_xticks([1, 2, 3, 4])
    ax[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Hop-depth pathology with a RoG-INDEPENDENT enumerator "
                 "(networkx all_simple_paths) on the same KG subgraphs", fontsize=10)
    fig.tight_layout()
    fig.savefig(R / "fig_nonrog_hop.pdf")
    print("wrote fig_nonrog_hop.pdf")


def _load_batch():
    fs = list(R.glob("batch_sweep_llama_*.csv"))
    if not fs:
        return None
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def plot_batch_sweep():
    df = _load_batch()
    if df is None:
        print("skip batch_sweep (no data)"); return
    pol = "topical" if "topical" in df.policy.unique() else df.policy.unique()[0]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for ds in df.dataset.unique():
        c = DSCOLOR.get(ds, "#555")
        sub = df[(df.dataset == ds) & (df.policy == pol)]
        piv = sub.pivot_table(index="batch", columns="condition", values="mean_ttft_ms")
        tp = sub.pivot_table(index="batch", columns="condition", values="throughput_tok_s")
        if "full" in piv and "reuse" in piv:
            ax[0].plot(piv.index, piv["full"] / piv["reuse"], "o-", color=c, label=DSLABEL.get(ds, ds))
            ax[1].plot(tp.index, tp["reuse"] / tp["full"], "o-", color=c, label=DSLABEL.get(ds, ds))
    for a, ylab, title in ((ax[0], "TTFT speedup (full / reuse)", "TTFT vs batch size"),
                           (ax[1], "throughput speedup (reuse / full)", "Throughput vs batch size")):
        a.set_xscale("log", base=2); a.set_xlabel("batch size"); a.set_ylabel(ylab)
        a.axhline(1.0, color="k", ls="--", lw=0.8); a.set_title(title)
    ax[0].legend(fontsize=8)
    fig.suptitle(f"GraphRAG-Opt KV-reuse speedup vs batch size ({pol} batching), Llama-2-7B", fontsize=10)
    fig.tight_layout(); fig.savefig(R / "fig_batch_sweep.pdf")
    print("wrote fig_batch_sweep.pdf")


def plot_retrieval_latency():
    f = R / "retrieval_latency.csv"
    if not f.exists():
        print("skip retrieval_latency (no data)"); return
    df = pd.read_csv(f)
    dss = [d for d in ["rmanluo/RoG-cwq", "rmanluo/RoG-webqsp", "metaqa-3hop"] if d in df.dataset.unique()]
    fig, ax = plt.subplots(1, len(dss), figsize=(3.4 * len(dss), 3.3), squeeze=False)
    for j, ds in enumerate(dss):
        a = ax[0][j]
        s = df[df.dataset == ds].sort_values("hop")
        x = s.hop.values
        w = 0.38
        a.bar(x - w / 2, s.bfs_ms, w, label="BFS (RoG/GraphRAG)", color="#bbb")
        a.bar(x + w / 2, s.jbidfs_ms, w, label="JbiDFS (ours)", color=DSCOLOR.get(ds, "#c0392b"))
        a.set_yscale("log"); a.set_xlabel("hop bound $k$"); a.set_xticks([1, 2, 3, 4])
        a.set_title(DSLABEL.get(ds, ds), fontsize=9)
        for _, r in s.iterrows():
            if r.jbidfs_ms > 0:
                a.text(r.hop, max(r.bfs_ms, r.jbidfs_ms) * 1.3, f"{r.speedup_vs_bfs:.1f}x",
                       ha="center", fontsize=7.5, color="#333")
    ax[0][0].set_ylabel("retrieval latency (ms)"); ax[0][0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle("JbiDFS vs. the standard KGQA enumerator (BFS), matched output", fontsize=10)
    fig.tight_layout(); fig.savefig(R / "fig_retrieval_latency.pdf")
    print("wrote fig_retrieval_latency.pdf")


def plot_subgcache():
    f = R / "subgcache_accuracy.csv"
    if not f.exists():
        print("skip subgcache (no data)"); return
    df = pd.read_csv(f)
    order = ["full", "noalign", "realign", "realign_rc"]
    labels = ["Full\nprefill", "SubGCache\ncontract\n(no realign)", "Ours\n(+realign)", "Ours\n(+realign\n+recompute)"]
    dss = list(df.dataset.unique())
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    x = range(len(order)); w = 0.8 / max(1, len(dss))
    for k, ds in enumerate(dss):
        s = df[df.dataset == ds].set_index("condition")
        vals = [s.loc[c, "hits_at_1"] if c in s.index else 0 for c in order]
        ax.bar([i + k * w for i in x], vals, w, label=DSLABEL.get(ds, ds), color=DSCOLOR.get(ds, "#888"))
    ax.set_xticks([i + w * (len(dss) - 1) / 2 for i in x]); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Hits@1 (leakage-free)"); ax.legend(fontsize=8)
    ax.set_title("Accuracy cost of SubGCache's no-positional-handling design, recovered by ours (Llama-2-7B)", fontsize=9)
    fig.tight_layout(); fig.savefig(R / "fig_subgcache.pdf")
    print("wrote fig_subgcache.pdf")


if __name__ == "__main__":
    plot_nonrog_hop()
    plot_retrieval_latency()
    plot_batch_sweep()
    plot_subgcache()
    print("done")
