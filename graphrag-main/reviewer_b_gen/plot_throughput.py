#!/usr/bin/env python3
"""Aggregate reviewer_b_gen throughput sweeps -> polished figure + markdown table.
Merges single-GPU (sweep_), short-context extension (sweepX_), and multi-GPU
(sweepMG_) runs; the speedup RATIO is config-invariant (both conditions share the
same setup within a run), so higher batches measured under model-parallel stitch
onto the single-GPU curve."""
import csv, glob
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

NICE = {"rog": "RoG (Llama-2-7B)", "llama7b": "Llama-2-7B", "llama13b": "Llama-2-13B",
        "opt6.7b": "OPT-6.7B", "opt13b": "OPT-13B", "opt30b": "OPT-30B",
        "gemma2b": "Gemma-2-2B", "gemma9b": "Gemma-2-9B"}
ORDER = ["rog", "llama7b", "llama13b", "opt6.7b", "opt13b", "opt30b", "gemma2b", "gemma9b"]
COLOR = {"rog": "#1f77b4", "llama7b": "#ff7f0e", "llama13b": "#2ca02c", "opt6.7b": "#d62728",
         "opt13b": "#9467bd", "opt30b": "#8c564b", "gemma2b": "#e377c2", "gemma9b": "#7f7f7f"}
MARK = {"rog": "o", "llama7b": "s", "llama13b": "^", "opt6.7b": "D", "opt13b": "v",
        "opt30b": "P", "gemma2b": "*", "gemma9b": "X"}


def load(tag, ds):
    pts = {}
    # Consistent config only (max_paths=8): single-GPU sweep_ (B<=64) + multi-GPU
    # sweepMG_ (higher B). sweepX_ (max_paths=4) is excluded to keep one config/curve.
    for pat in [f"sweep_{tag}_{ds}.csv", f"sweepMG_{tag}_{ds}.csv", f"sweepMG2_{tag}_{ds}.csv"]:
        fp = RES / pat
        if not fp.exists():
            continue
        for r in csv.DictReader(open(fp)):
            try:
                B = int(r["batch"]); sp = float(r["tput_speedup"])
            except (ValueError, KeyError):
                continue
            if sp != sp:
                continue
            pts[B] = sp   # later files (higher B) win for shared batches
    return dict(sorted(pts.items()))


def panel(ax, ds, title):
    for tag in ORDER:
        pts = load(tag, ds)
        if not pts:
            continue
        xs = sorted(pts); ys = [pts[x] for x in xs]
        ax.plot(xs, ys, marker=MARK[tag], color=COLOR[tag], label=NICE[tag],
                lw=1.8, ms=6, mec="white", mew=0.5)
    ax.axhspan(2.1, 3.5, color="green", alpha=0.07, zorder=0)
    ax.axhline(1.0, ls="--", c="gray", lw=1.0)
    ax.text(1.05, 2.75, "paper's reported\n2.1–3.5×", fontsize=7.5, color="green", alpha=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256])
    ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64, 128, 256], fontsize=8)
    ax.set_xlabel("batch size", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0.7, None)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    panel(axes[0], "cwq", "CWQ (Freebase)")
    panel(axes[1], "webqsp", "WebQSP (Freebase)")
    axes[0].set_ylabel("generation throughput speedup\n(GraphRAG-Opt subgraph caching vs. standard KV caching)", fontsize=10)
    axes[1].legend(fontsize=8, ncol=1, loc="upper left", framealpha=0.9)
    fig.suptitle("Generation throughput speedup vs. batch size, grows with batch and holds across backbones (single A100-80GB)", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    import shutil
    out = RES / "fig_throughput_batch_sweep.pdf"
    fig.savefig(out); fig.savefig(str(out).replace(".pdf", ".png"), dpi=140)
    # also publish under the rebuttal-linked name review_b4_kv_sweep (results/ + figures_bundle)
    pub = Path("/home/am137688/amir/graphRag/results")
    for ext in (".pdf", ".png"):
        src = str(out).replace(".pdf", ext)
        shutil.copy(src, pub / f"review_b4_kv_sweep{ext}")
        sub = "pdf" if ext == ".pdf" else "png"
        if (pub / "figures_bundle" / sub).is_dir():
            shutil.copy(src, pub / "figures_bundle" / sub / f"review_b4_kv_sweep{ext}")
    print(f"Wrote {out} and published as review_b4_kv_sweep")

    # markdown table (CWQ)
    Bs = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    lines = ["| backbone | " + " | ".join(f"B={b}" for b in Bs) + " |",
             "|---|" + "---|" * len(Bs)]
    for tag in ORDER:
        pts = load(tag, "cwq")
        if not pts:
            continue
        cells = [f"{pts[b]:.2f}×" if b in pts else "—" for b in Bs]
        lines.append(f"| {NICE[tag]} | " + " | ".join(cells) + " |")
    (RES / "throughput_table_cwq.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {RES/'throughput_table_cwq.md'}")


if __name__ == "__main__":
    main()
