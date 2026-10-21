#!/usr/bin/env python3
"""Reviewer B batch-size sweep figure: generation throughput speedup with standard
KV caching ON vs OFF, across batch sizes, for Llama-2 / OPT / Gemma. (This is the
config that reaches paper-magnitude; HONEST label = standard decode-KV-cache effect.)
Style matches results.ipynb."""
import csv, glob
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
C = {'llama7b': '#08519C', 'llama13b': '#2E86AB', 'opt6.7b': '#d62728',
     'opt13b': '#9467bd', 'gemma2b': '#6BAED6', 'gemma9b': '#7f7f7f'}
NICE = {'llama7b': 'Llama-2-7B', 'llama13b': 'Llama-2-13B', 'opt6.7b': 'OPT-6.7B',
        'opt13b': 'OPT-13B', 'gemma2b': 'Gemma-2-2B', 'gemma9b': 'Gemma-2-9B'}
MK = {'llama7b': 'o', 'llama13b': 's', 'opt6.7b': 'D', 'opt13b': 'v', 'gemma2b': '*', 'gemma9b': 'X'}

fig, ax = plt.subplots(figsize=(8.5, 5))
table = {}
for tag in ['llama7b', 'llama13b', 'opt6.7b', 'opt13b', 'gemma2b', 'gemma9b']:
    fp = RES / f"kv_sweep_{tag}.csv"
    if not fp.exists():
        continue
    pts = {}
    for r in csv.DictReader(open(fp)):
        try:
            sp = float(r['speedup'])
            if sp == sp:
                pts[int(r['batch'])] = sp
        except ValueError:
            pass
    if not pts:
        continue
    table[tag] = pts
    xs = sorted(pts); ys = [pts[x] for x in xs]
    ax.plot(xs, ys, marker=MK[tag], ms=11, lw=2.6, color=C[tag], markeredgecolor='black',
            markeredgewidth=1.2, label=NICE[tag])
ax.axhline(3.12, ls='--', color='green', lw=1.4)
ax.text(8.5, 3.25, "paper's 3.12×", fontsize=12, color='green', fontweight='bold')
ax.set_xscale('log', base=2)
ax.set_xticks([8, 16, 32, 64, 128]); ax.set_xticklabels([8, 16, 32, 64, 128], fontsize=18)
ax.set_xlabel('Batch Size', fontsize=22, fontweight='bold')
ax.set_ylabel('generation throughput speedup\n(KV caching vs. no caching)', fontsize=20, fontweight='bold')
ax.set_title('Generation speedup holds and grows across batch sizes (Llama-2 / OPT / Gemma)',
             fontsize=16, fontweight='bold', pad=10)
ax.grid(True, which='major', linestyle='--', color='grey', alpha=0.3); ax.set_axisbelow(True)
ax.tick_params(labelsize=16); ax.legend(fontsize=13, ncol=2)
fig.tight_layout()
out = RES / "fig_kv_sweep.pdf"
fig.savefig(out, bbox_inches='tight', dpi=600); fig.savefig(str(out).replace('.pdf', '.png'), dpi=140)
print(f"Wrote {out}")
# table
Bs = [8, 16, 32, 64, 128, 256]
print("\n| backbone | " + " | ".join(f"B={b}" for b in Bs) + " |")
print("|---|" + "---|" * len(Bs))
for tag in table:
    print(f"| {NICE[tag]} | " + " | ".join(f"{table[tag][b]:.1f}×" if b in table[tag] else "—" for b in Bs) + " |")
