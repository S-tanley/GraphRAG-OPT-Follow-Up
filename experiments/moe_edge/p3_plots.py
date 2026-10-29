"""Multi-user plan 3.2.2 / 3.3.3: throughput vs p99 TPOT.

  fig 1 (GPU resident, p32_resident.json): change vs fcfs at the same B, for P-prev / P-anchor / true routes, every
        B x wait bound k; the H5 box (p99 <= +20%, throughput >= +10%) shaded
  fig 2 (offload, N=128; p33e_*.json (300 s) at cache 50%, p33d_*.json (30 s) otherwise): absolute throughput vs p99 over B 8..64 for sync+LFU, + substitution only,
        full stack (fcfs) and full stack (P-anchor batching, k=2)

  python experiments/moe_edge/p3_plots.py outputs/moe/mu developer-log/report/figs
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager, ticker  # noqa: E402

for name in ("PingFang SC", "Hiragino Sans GB", "Heiti SC", "Arial Unicode MS", "Noto Sans CJK SC"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False

BLUE, ORANGE, GREEN, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID, SURF = "#1f1f1e", "#6b6a63", "#e6e5e0", "#fcfcfb"
PF = "lfu/ah8:AVG:m8:p0.3:g0.5:kl0.15"


def style(ax):
    ax.set_facecolor(SURF)
    ax.grid(color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def fig_resident(rows, out):
    sel = [("prev", "P-prev（上一 token 路由）", BLUE, "o"), ("anchor", "P-anchor", ORANGE, "s"),
           ("oracle", "真实路由（上限）", GREEN, "^")]
    Ns = sorted({r["n"] for r in rows})
    fig, axes = plt.subplots(1, len(Ns), figsize=(3.3 * len(Ns), 3.9), sharex=True, sharey=True, facecolor=SURF)
    for ax, n in zip(axes, Ns):
        style(ax)
        ax.add_patch(plt.Rectangle((-5, 10), 25, 30, color=GREEN, alpha=0.08, lw=0, zorder=0))
        ax.axhline(10, color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.axvline(20, color=MUTED, lw=1, ls=(0, (4, 3)))
        for s, label, col, mk in sel:
            for b in sorted({r["B"] for r in rows if r["B"] < n}):      # B = N: nothing to choose
                ref = next(r for r in rows if r["n"] == n and r["B"] == b and r["select"] == "fcfs")
                pts = sorted((r for r in rows if r["n"] == n and r["B"] == b and r["select"] == s),
                             key=lambda r: r["klim"])
                xs = [100 * (r["tpot_p99_ms"] / ref["tpot_p99_ms"] - 1) for r in pts]
                ys = [100 * (r["throughput_tok_s"] / ref["throughput_tok_s"] - 1) for r in pts]
                ax.plot(xs, ys, color=col, lw=1.2, marker=mk, ms=7, mec=SURF, mew=1.5,
                        label=label if b == 8 else None, zorder=3)
        ax.set_title(f"N = {n}", fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("p99 TPOT 相对轮转（%）", fontsize=9, color=INK)
    axes[0].set_ylabel("吞吐相对轮转（%）", fontsize=9, color=INK)
    xmax = max(max(line.get_xdata()) for ax in axes for line in ax.get_lines() if len(line.get_xdata()) > 1)
    axes[0].set_xlim(-5, xmax + 8)
    axes[0].set_ylim(-2, 20)
    axes[0].text(1, 18.8, "H5 区域", fontsize=8, color=MUTED, va="top")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.005, 0.93), ncol=3, frameon=False, fontsize=9,
               labelcolor=INK)
    fig.text(0.01, 0.985, "GPU 全驻留：相似性组 batch 的吞吐 vs p99（GLM-4.5-Air CoT · 同一 B 下相对轮转 · "
             "每条线 = 一个 B，点 = 等待上限 k 1/2/4）", fontsize=10, color=INK, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(out, dpi=150, facecolor=SURF)
    print("wrote", out)


def fig_offload(rows, out, n=128):
    series = [("同步 + LFU", lambda r: r["policy"] == "lfu/none" and r["subst"] == "none" and r["rho"] == 0
               and not r["protect"] and r["select"] == "fcfs", MUTED, "o"),
              ("+ 只做顶替 ≤ 10%（H6 参照）", lambda r: r["policy"] == "lfu/none" and r["subst"] == "rate:0.1"
               and r["rho"] == 0 and not r["protect"] and r["select"] == "fcfs", BLUE, "o"),
              ("全栈，不组 batch（轮转）", lambda r: r["policy"] == PF and r["subst"] == "rate:0.1" and r["rho"] == 0.5
               and r["protect"] and r["select"] == "fcfs", GREEN, "s"),
              ("全栈（P-anchor 组 batch，k=2）", lambda r: r["policy"] == PF and r["subst"] == "rate:0.1"
               and r["rho"] == 0.5 and r["protect"] and r["select"] == "anchor" and r["klim"] == 2, ORANGE, "^")]
    panels = [("h200_pcie", 0.5, "H200 PCIe"), ("mac_ssd", 0.5, "Mac SSD"), ("h200_pcie", 0.75, "H200 PCIe"),
              ("mac_ssd", 0.75, "Mac SSD")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.1), facecolor=SURF)
    for ax, (hw, f, hwname) in zip(axes, panels):
        style(ax)
        for label, ok, col, mk in series:
            pts = sorted((r for r in rows if r["hw"] == hw and r["cache_frac"] == f and r["n"] == n and ok(r)),
                         key=lambda r: r["B"])
            xs = [r["tpot_p99_ms"] / 1e3 for r in pts]
            ys = [r["throughput_tok_s"] for r in pts]
            ax.plot(xs, ys, color=col, lw=1.6, marker=mk, ms=7, mec=SURF, mew=1.5, label=label, zorder=3)
            if label.startswith("+ 只做"):
                for r, x, y in zip(pts, xs, ys):
                    ax.annotate(f"B={r['B']}", (x, y), xytext=(5, -9), textcoords="offset points", fontsize=7,
                                color=MUTED)
        ax.set_xscale("log")
        ax.set_yscale("log")
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            cand = [m * 10 ** e for e in range(-1, 4) for m in (1, 1.5, 2, 3, 4, 5, 7)]
            ticks = [v for v in cand if lim[0] <= v <= lim[1]]
            while len(ticks) > 8:
                ticks = ticks[::2]
            axis.set_major_locator(ticker.FixedLocator(ticks))
            axis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_formatter(ticker.NullFormatter())
        if f >= 0.75:
            ax.text(0.03, 0.05, "蓝线（只做顶替）与绿线重合：\n顶替吸收了全部缺失，不再加载", transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=7.5, color=MUTED)
        ax.set_title(f"{hwname} · 缓存 {f:.0%} · N = {n}", fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("p99 TPOT（秒，对数）", fontsize=9, color=INK)
    axes[0].set_ylabel("吞吐（token/s，对数）", fontsize=9, color=INK)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.005, 0.93), ncol=4, frameon=False, fontsize=9,
               labelcolor=INK)
    fig.text(0.01, 0.985, "Offload 多用户：吞吐 vs p99（GLM-4.5-Air CoT · 线上的点 = B 8 / 16 / 32 / 64 · "
             "左上更好 · 缓存按另外 72 个请求的频率预填 · 50% 模拟 300 s，75% 模拟 30 s）", fontsize=10, color=INK, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(out, dpi=150, facecolor=SURF)
    print("wrote", out)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    fig_resident(json.load(open(os.path.join(src, "p32", "p32_resident.json"))),
                 os.path.join(dst, "p32_resident_tput_vs_p99.png"))
    load = lambda tag: sum((json.load(open(os.path.join(src, "p33", f"{tag}_{x}.json"))) for x in ("fcfs", "anchor")), [])
    rows = load("p33e") + [r for r in load("p33d") if r["cache_frac"] != 0.5]    # cache 50%: the 300 s runs
    fig_offload(rows, os.path.join(dst, "p33_offload_tput_vs_p99.png"))


if __name__ == "__main__":
    main()
