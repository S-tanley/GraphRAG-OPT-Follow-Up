"""Multi-user plan 1.1.3: decode stall vs resident share rho, per cache size and fill (calibrated-GATE prefetch).

  python experiments/moe_edge/p11_plot.py outputs/moe/mu/p11 developer-log/report/figs/p11_stall_vs_rho.png
"""
import glob
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

for name in ("PingFang SC", "Hiragino Sans GB", "Heiti SC", "Arial Unicode MS", "Noto Sans CJK SC"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False

SERIES = [("glob", "Wtrain 固定", "#2a78d6", "o"), ("work", "同类负载固定", "#eb6834", "s"),
          ("pred:8", "请求级预测", "#1baf7a", "^"), ("ema:16:64:0.05", "在线 EMA", "#eda100", "D")]
INK, MUTED, GRID, SURF = "#1f1f1e", "#6b6a63", "#e6e5e0", "#fcfcfb"
RHOS = (0.0, 0.25, 0.5, 0.75, 0.9)
YLO, YHI = -12, 30


def main():
    rows = sum((json.load(open(f)) for f in sorted(glob.glob(sys.argv[1] + "/*.json"))), [])
    hws = [("mac_ssd", "Mac SSD"), ("h200_pcie", "H200 PCIe")]
    fs = (0.25, 0.5, 0.75, 0.9)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2), sharex=True, sharey=True, facecolor=SURF)
    for i, (hw, hwname) in enumerate(hws):
        for j, f in enumerate(fs):
            ax = axes[i][j]
            ax.set_facecolor(SURF)
            g = [r for r in rows if r["hw"] == hw and r["cache_frac"] == f and ":GATEs:" in r["policy"]]
            base = next(r for r in g if r["rho"] == 0)["stall_ms_per_tok"]
            ax.axhline(0, color=MUTED, lw=1)
            ax.axhline(-10, color=MUTED, lw=1, ls=(0, (4, 3)))
            off = []
            for fill, label, col, mk in SERIES:
                ys = [0.0] + [100 * (next(r for r in g if r["rho"] == rho and r["fill"] == fill)["stall_ms_per_tok"] / base - 1)
                              for rho in RHOS[1:]]
                yc = [min(y, YHI - 1) for y in ys]
                ax.plot(RHOS, yc, color=col, lw=2, marker=mk, ms=8, mec=SURF, mew=2, label=label, zorder=3)
                off += [f"ρ={x:g} {label} +{y:.0f}%" for x, y in zip(RHOS, ys) if y > YHI]
                if i == 0 and j == 1:                          # direct labels once, at the line ends
                    ax.annotate(label, (RHOS[-1], min(ys[-1], YHI - 1)), xytext=(7, -3 if fill == "glob" else 0),
                                textcoords="offset points", va="center", fontsize=8.5, color=INK)
            if off:                                            # values above the axis, listed instead of drawn
                ax.text(0.02, 0.97, "超出纵轴：\n" + "\n".join(off), transform=ax.transAxes, va="top", fontsize=7.5,
                        color=MUTED, zorder=5, bbox=dict(facecolor=SURF, edgecolor="none", alpha=0.92, pad=1.5))
            best = min(next(r for r in g if r["rho"] == rho and r["fill"] == fl)["stall_ms_per_tok"]
                       for rho in RHOS[1:] for fl, *_ in SERIES)
            ax.set_title(f"{hwname} · 缓存 {f:.0%}\nρ=0：{base:.1f} ms/token · 最好 {100 * (best / base - 1):+.1f}%".replace("-", "-"),
                         fontsize=9, color=INK, loc="left")
            ax.set_ylim(YLO, YHI)
            ax.set_xticks(RHOS)
            ax.set_xlim(-0.05, 1.25)
            ax.grid(axis="y", color=GRID, lw=0.8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(GRID)
            ax.tick_params(colors=MUTED, labelsize=8.5)
            if j == 0:
                ax.set_ylabel("decode 停顿相对 ρ=0 的变化（%）", fontsize=9, color=INK)
            if i == 1:
                ax.set_xlabel("常驻区占缓存的比例 ρ", fontsize=9, color=INK)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.005, 0.955), ncol=4, frameon=False, fontsize=9, labelcolor=INK)
    fig.text(0.01, 0.985, "常驻 + 流动分区：停顿 vs ρ（单用户 · GLM-4.5-Air CoT 50 · 预取 = 校准 GATE + 阈值 · 虚线 = 判据 -10%）",
             fontsize=10.5, color=INK, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(sys.argv[2], dpi=150, facecolor=SURF)
    print("wrote", sys.argv[2])


if __name__ == "__main__":
    main()
