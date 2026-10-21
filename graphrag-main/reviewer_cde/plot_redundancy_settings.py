#!/usr/bin/env python3
"""Reviewer C: cross-query redundancy / prefix-reuse across settings — our KGQA
datasets (measured) vs. published production LLM-serving & RAG traces (50%+ is the
norm). Primary-source numbers verified by the research agent. Style matches results.ipynb."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_LIGHT, C_MID, C_DARK, C_SKY = '#D2E1F2', '#2E86AB', '#08519C', '#6BAED6'

# (label, value%, group)  group: 'ours' | 'rag' | 'serving' | 'conv'
DATA = [
    ("MetaQA\n(ours)", 6.5, 'ours'),
    ("WebQSP\n(ours)", 15.0, 'ours'),
    ("CWQ\n(ours)", 41.7, 'ours'),
    ("SQA conv.\ncoref", 46.0, 'conv'),
    ("RAGCache\ntop-3%→MMLU", 60.0, 'rag'),
    ("Cache-Craft\ntop-5% chunks", 60.0, 'rag'),
    ("Mooncake\ntool&agent", 59.0, 'serving'),
    ("Mooncake\nsynthetic", 66.0, 'serving'),
    ("SGLang\nLLaVA-34B", 52.4, 'serving'),
    ("SGLang\nVicuna-33B", 74.1, 'serving'),
]
gcol = {'ours': C_DARK, 'conv': C_MID, 'rag': C_SKY, 'serving': C_LIGHT}
glab = {'ours': 'our KGQA datasets (measured)', 'conv': 'conversational KGQA',
        'rag': 'RAG document reuse (published)', 'serving': 'production LLM-serving traces (published)'}

fig, ax = plt.subplots(figsize=(13, 5.2))
x = range(len(DATA))
bars = ax.bar(x, [d[1] for d in DATA], color=[gcol[d[2]] for d in DATA],
              edgecolor='black', width=0.7)
for i, d in enumerate(DATA):
    ax.text(i, d[1] + 1, f'{d[1]:.0f}%' if d[1] == int(d[1]) else f'{d[1]:.1f}%',
            ha='center', va='bottom', fontsize=16, fontweight='bold')
ax.axhspan(50, 100, color='green', alpha=0.06)
ax.text(0.15, 52, '50%+ cross-query redundancy is the norm in deployed serving',
        fontsize=13, color='darkgreen', fontweight='bold')
ax.set_xticks(list(x)); ax.set_xticklabels([d[0] for d in DATA], fontsize=13)
ax.set_ylabel('cross-query redundancy /\nprefix-cache hit rate (%)', fontsize=20, fontweight='bold')
ax.set_title('High cross-query overlap is realistic — our datasets span a 6–42% spectrum; '
             'real deployments reach 50–74%', fontsize=17, fontweight='bold', pad=10)
ax.yaxis.grid(True, linestyle='--', color='grey', alpha=0.3); ax.set_axisbelow(True)
ax.tick_params(labelsize=14); ax.set_ylim(0, 82)
handles = [plt.Rectangle((0, 0), 1, 1, fc=gcol[g], ec='black') for g in ['ours', 'conv', 'rag', 'serving']]
ax.legend(handles, [glab[g] for g in ['ours', 'conv', 'rag', 'serving']], fontsize=12, loc='upper left')
fig.tight_layout()
out = Path(__file__).resolve().parent / "results/fig_redundancy_settings.pdf"
fig.savefig(out, bbox_inches='tight', dpi=600); fig.savefig(str(out).replace(".pdf", ".png"), dpi=140)
print(f"Wrote {out}")
