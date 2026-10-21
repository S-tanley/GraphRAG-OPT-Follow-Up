#!/usr/bin/env python3
"""Build /home/am137688/amir/graphRag/results/review.ipynb in the style of
results.ipynb (blue palette, bold 22-26pt fonts, black-edged bars, dashed grey grid,
dpi 600). Each figure is followed by a plain-English explanation markdown cell.
Data is embedded inline (self-contained, like results.ipynb)."""
import json
from pathlib import Path

DATA = json.load(open(Path(__file__).resolve().parent / "results/_review_data.json"))

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}

cells = []

# ---- title ----
cells.append(md("""# GraphRAG-Opt — Rebuttal Figures (Reviewers 1, B, C, D, E)

Self-contained notebook with every figure and number used in the rebuttal, in the
same visual style as `results.ipynb`. Each figure is followed by a short explanation.

**Palette:** `#D2E1F2` (light blue), `#2E86AB` (mid blue), `#08519C` (dark blue),
`#6BAED6` (sky). Bold 22–26 pt fonts, black-edged bars, dashed grey grid, dpi 600.
"""))

# ---- shared setup + DATA ----
cells.append(code(f"""import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ---- shared style (matches results.ipynb) ----
C_LIGHT, C_MID, C_DARK, C_SKY = '#D2E1F2', '#2E86AB', '#08519C', '#6BAED6'
plt.rcParams.update({{'font.size': 22, 'axes.titlesize': 24, 'axes.labelsize': 24}})

def style_ax(ax, xlabel=None, ylabel=None, title=None):
    if xlabel: ax.set_xlabel(xlabel, fontsize=24, fontweight='bold')
    if ylabel: ax.set_ylabel(ylabel, fontsize=24, fontweight='bold')
    if title:  ax.set_title(title, fontsize=24, fontweight='bold', pad=10)
    ax.grid(True, which='major', linestyle='--', color='grey', alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=22)

# ---- embedded rebuttal data ----
DATA = {json.dumps(DATA)}
"""))

# ===== B2(i) retrieval latency =====
cells.append(md("## Reviewer B — B2(i): JbiDFS retrieval vs. BFS (path enumeration latency)"))
cells.append(code("""ret = DATA['retrieval']
ds_list = ['CWQ', 'WEB', 'MQA']
hops = [2, 3, 4]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
for j, ds in enumerate(ds_list):
    ax = axes[j]
    bfs = [ret[ds][str(h)]['bfs'] for h in hops]
    jbi = [ret[ds][str(h)]['jbidfs'] for h in hops]
    x = np.arange(len(hops)); w = 0.38
    ax.bar(x - w/2, bfs, w, color=C_LIGHT, edgecolor='black', label='BFS (baseline)')
    ax.bar(x + w/2, jbi, w, color=C_MID, edgecolor='black', label='JbiDFS (ours)')
    for i, h in enumerate(hops):
        sp = ret[ds][str(h)]['sp']
        ax.text(x[i] + w/2, jbi[i] + max(bfs) * 0.02, f'{sp:.1f}×', ha='center',
                va='bottom', fontsize=18, fontweight='bold', color=C_DARK)
    ax.set_xticks(x); ax.set_xticklabels([f'{h}-hop' for h in hops])
    style_ax(ax, ylabel=('retrieval ms' if j == 0 else None), title=ds)
axes[0].legend(fontsize=16, loc='upper left')
fig.suptitle('JbiDFS path enumeration is 3–37× faster than BFS (multi-hop regime)',
             fontsize=22, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig('review_b2_retrieval.pdf', bbox_inches='tight', dpi=600)
plt.show()"""))
cells.append(md("""**What this shows.** Path enumeration is the GraphRAG retrieval bottleneck, and it
explodes at deep hops. Our **JbiDFS** (join-based bidirectional DFS) returns essentially
the same path set as RoG's BFS but **3.0–37.5× faster** in the 2–4-hop regime that
dominates retrieval cost (e.g. CWQ 3-hop 88→12 ms = 7.6×; MetaQA 4-hop 75→2 ms = 37×).
This **reproduces and exceeds** the paper's 2.64× retrieval claim, and the speedup is
**batch-independent** (a per-query algorithmic win). `retrieval_latency.csv`, N=200/dataset."""))

# ===== B3 non-RoG hop scaling =====
cells.append(md("## Reviewer B — B3: hop-depth pathology reproduced on a non-RoG (networkx) enumerator"))
cells.append(code("""nr = DATA['nonrog']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4.8))
hops = [1, 2, 3, 4]
mk = {'CWQ': 'o', 'WEB': 's', 'MQA': '^'}
col = {'CWQ': C_DARK, 'WEB': C_MID, 'MQA': C_SKY}
for ds in ['CWQ', 'WEB', 'MQA']:
    paths = [nr[ds][str(h)]['paths'] for h in hops]
    ms = [nr[ds][str(h)]['ms'] for h in hops]
    ax1.plot(hops, paths, marker=mk[ds], ms=12, lw=3, color=col[ds],
             markeredgecolor='black', markeredgewidth=1.5, label=ds)
    ax2.plot(hops, ms, marker=mk[ds], ms=12, lw=3, color=col[ds],
             markeredgecolor='black', markeredgewidth=1.5, label=ds)
for ax, yl, ti in [(ax1, 'paths / query', 'Path count vs. hop depth'),
                   (ax2, 'enumeration ms', 'Latency vs. hop depth')]:
    ax.set_yscale('log'); ax.set_xticks(hops)
    style_ax(ax, xlabel='hop bound k', ylabel=yl, title=ti)
ax1.legend(fontsize=18)
fig.suptitle('networkx all_simple_paths (zero RoG code) reproduces the superlinear hop-depth growth',
             fontsize=22, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig('review_b3_nonrog_hop.pdf', bbox_inches='tight', dpi=600)
plt.show()"""))
cells.append(md("""**What this shows.** Reviewer B asked whether the 1→4-hop cost explosion is a property
of RoG's BFS or of path enumeration itself. Re-running on the textbook `networkx`
`all_simple_paths` enumerator (zero RoG code), on the same KG subgraphs, reproduces the
**same superlinear shape**: CWQ paths 1→2,100 (2,818×) and latency 0.5→991 ms (1,887×).
So the pathology is **structural to hop-constrained path enumeration**, not a RoG quirk.
`nonrog_hop_scaling_summary.csv`."""))

# ===== B1 SubGCache accuracy =====
cells.append(md("## Reviewer B — B1: SubGCache fidelity contract (realignment matters)"))
cells.append(code("""sg = DATA['subg']
conds = ['full', 'noalign', 'realign', 'realign_rc']
labels = ['Full prefill', 'no realign\\n(SubGCache)', '+ realign\\n(ours)', '+ realign\\n+ recompute']
cols = [C_DARK, C_LIGHT, C_MID, C_SKY]
panels = [('rog', 'CWQ', 'RoG · CWQ'), ('rog', 'WEB', 'RoG · WebQSP'),
          ('llama', 'CWQ', 'Llama-2-7B · CWQ'), ('llama', 'WEB', 'Llama-2-7B · WebQSP')]
fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
for ax, (tag, ds, ti) in zip(axes, panels):
    vals = [sg[tag][ds][c] for c in conds]
    bars = ax.bar(range(4), vals, color=cols, edgecolor='black')
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=15, fontweight='bold')
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=13)
    style_ax(ax, ylabel=('Hits@1' if ti.startswith('RoG · CWQ') else None), title=ti)
    ax.set_ylim(0, max(vals) * 1.2)
fig.suptitle('Reusing ALL paths verbatim without realignment collapses accuracy; RoPE realignment recovers +19-28 pts',
             fontsize=18, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig('review_b1_subgcache.pdf', bbox_inches='tight', dpi=600)
plt.show()"""))
cells.append(md("""**What this shows.** SubGCache's design reuses a shared subgraph **verbatim with no
positional realignment**. In our position-varying multi-hop setting, reusing *all* paths
that way **collapses accuracy** (e.g. RoG·CWQ 0.53→0.18, −35 pts). Our **RoPE realignment**
recovers most of it (+19-28 pts), and selective recompute closes the rest, but verbatim
reuse of everything still sits below full prefill, which is exactly **why we deploy
shared-path reuse only** (where GraphRAG-Opt ≈ GraphRAG-R, non-inferior). This is the
stress regime; the deployed regime loses only 5–10 pts without realignment.
`subgcache_accuracy{,_rog}.csv`, N=300."""))

# ===== B2(ii) PathRAG selection =====
cells.append(md("## Reviewer B — B2(ii): JbiDFS (enumeration) vs. PathRAG-flow (selection) are complementary"))
cells.append(code("""pr = DATA['pathrag']
ds_list = ['CWQ', 'WEB']
metrics = [('paths', 'paths / query'), ('tok', 'context tokens / query'), ('hits', 'Hits@1')]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
x = np.arange(len(ds_list)); w = 0.38
for ax, (key, yl) in zip(axes, metrics):
    rog = [pr[ds]['rog-sel'][key] for ds in ds_list]
    flow = [pr[ds]['pathrag-flow'][key] for ds in ds_list]
    ax.bar(x - w/2, rog, w, color=C_LIGHT, edgecolor='black', label='RoG selection')
    ax.bar(x + w/2, flow, w, color=C_MID, edgecolor='black', label='PathRAG-flow (ported)')
    ax.set_xticks(x); ax.set_xticklabels(ds_list)
    style_ax(ax, ylabel=yl)
axes[0].legend(fontsize=15, loc='upper left')
fig.suptitle('PathRAG-flow (selection) over JbiDFS (enumeration): complementary stages, not competitors',
             fontsize=19, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig('review_b2_pathrag.pdf', bbox_inches='tight', dpi=600)
plt.show()"""))
cells.append(md("""**What this shows.** PathRAG's contribution is a **flow-based selection rule**, applied
*after* path enumeration; JbiDFS is the **enumerator**. Ported faithfully (α=0.8,
threshold=0.3, 15-edge cap) and run over the same CWQ/WebQSP subgraphs into the same reader,
PathRAG-flow admits *more* paths (larger context: 228→379 tokens on CWQ) for a small
Hits@1 change (+1.6–5.0 pts). The point is they **compose** — JbiDFS makes the enumeration
PathRAG's inner loop needs faster — not that one beats the other. `pathrag_compare.csv`, N=300."""))

# ===== C redundancy spectrum =====
cells.append(md("## Reviewer C — redundancy spectrum & graceful degradation (robustness to low overlap)"))
cells.append(code("""red = DATA['red']
fig, (axA, axB) = plt.subplots(1, 2, figsize=(16, 5))
col = {'CWQ': C_DARK, 'WEB': C_MID, 'MQA': C_SKY}
Bs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
for ds in ['CWQ', 'WEB', 'MQA']:
    top = [red[ds]['topical'][str(b)] * 100 for b in Bs]
    rnd = [red[ds]['random'][str(b)] * 100 for b in Bs]
    axA.plot(Bs, top, '-o', color=col[ds], lw=3, ms=9, markeredgecolor='black',
             markeredgewidth=1.2, label=f'{ds} (entity-clustered)')
    axA.plot(Bs, rnd, '--s', color=col[ds], lw=2, ms=7, markeredgecolor='black',
             markeredgewidth=1.0, alpha=0.7, label=f'{ds} (diverse)')
axA.set_xscale('log', base=2); axA.set_xticks([1, 4, 16, 64, 256])
axA.set_xticklabels([1, 4, 16, 64, 256])
style_ax(axA, xlabel='batch size', ylabel='within-batch path redundancy (%)',
         title='(a) Redundancy spectrum')
axA.legend(fontsize=12, ncol=2)
# panel B: graceful degradation 1/(1-f)
f = np.linspace(0, 0.45, 100)
axB.plot(f * 100, 1 / (1 - f), color=C_DARK, lw=3, label='generation benefit = 1/(1-f)')
for ds in ['CWQ', 'WEB', 'MQA']:
    fmax = max(red[ds]['topical'].values())
    axB.plot(fmax * 100, 1 / (1 - fmax), 'o', ms=14, color=col[ds], markeredgecolor='black')
    axB.annotate(f'{ds} {fmax*100:.0f}%', (fmax * 100, 1 / (1 - fmax)),
                 textcoords='offset points', xytext=(8, -4), fontsize=14, fontweight='bold', color=col[ds])
axB.axhline(1.0, ls='--', color='grey')
style_ax(axB, xlabel='within-batch redundancy f (%)', ylabel='generation benefit (×)',
         title='(b) Graceful degradation (never < 1×)')
axB.set_ylim(0.95, 1.75); axB.legend(fontsize=14, loc='upper left')
fig.tight_layout()
fig.savefig('review_c_robustness.pdf', bbox_inches='tight', dpi=600)
plt.show()"""))
cells.append(md("""**What this shows.** Reviewer C worried the method depends on high overlap. **(a)** We
already span a wide **6–42% redundancy spectrum** — diverse ("random") batches sit at the
low end, entity-clustered batches at the high end, and MetaQA (~6%) is a genuinely
low-overlap point. **(b)** The generation-side benefit (redundant prefill work
eliminated, `1/(1−f)`) **degrades gracefully to a no-op as overlap → 0 and never falls
below 1×** — reuse is upside-only (matching SGLang's "no overhead even without cache hits").
The **retrieval** speedup (3–37×) is redundancy-independent, so even at zero overlap the
paper's dominant win is retained. `redundancy.csv`."""))

# ===== C overlap sweep =====
cells.append(md("## Reviewer C — controlled query-diversity sweep"))
cells.append(code("""ov = DATA['overlap']
fig, ax = plt.subplots(figsize=(7.2, 4.8))
col = {'CWQ': C_DARK, 'WEB': C_MID}
for ds in ['CWQ', 'WEB']:
    pts = sorted(ov[ds]); xs = [p[0] * 100 for p in pts]; ys = [p[1] * 100 for p in pts]
    ax.plot(xs, ys, '-o', color=col[ds], lw=3, ms=10, markeredgecolor='black',
            markeredgewidth=1.2, label=ds)
style_ax(ax, xlabel='% of batch entity-clustered\\n(0 = fully diverse → 100 = clustered)',
         ylabel='within-batch redundancy (%)', title='Smooth, graceful scaling with overlap')
ax.legend(fontsize=18)
fig.tight_layout()
fig.savefig('review_c_overlap_sweep.pdf', bbox_inches='tight', dpi=600)
plt.show()"""))
cells.append(md("""**What this shows.** Holding the question pool fixed, we dial the fraction of a batch
that is entity-clustered from 0% (fully diverse) to 100%. Redundancy — and therefore the
reuse benefit — rises **smoothly and monotonically** and collapses to a near-no-op at full
diversity. There is no cliff and no regime where reuse inverts. `exp_overlap_sweep.py`."""))

# ===== C redundancy across settings =====
cells.append(md("## Reviewer C — high cross-query overlap is realistic (production-trace evidence)"))
cells.append(code("""# (label, value%, group); production numbers primary-source verified
SET = [("MetaQA\\n(ours)", 6.5, 'ours'), ("WebQSP\\n(ours)", 15.0, 'ours'), ("CWQ\\n(ours)", 41.7, 'ours'),
       ("SQA conv.\\ncoref", 46.0, 'conv'), ("RAGCache\\ntop-3%", 60.0, 'rag'), ("Cache-Craft\\ntop-5%", 60.0, 'rag'),
       ("Mooncake\\ntool&agent", 59.0, 'serving'), ("Mooncake\\nsynthetic", 66.0, 'serving'),
       ("SGLang\\nLLaVA-34B", 52.4, 'serving'), ("SGLang\\nVicuna-33B", 74.1, 'serving')]
gcol = {'ours': C_DARK, 'conv': C_MID, 'rag': C_SKY, 'serving': C_LIGHT}
glab = {'ours': 'our KGQA datasets (measured)', 'conv': 'conversational KGQA',
        'rag': 'RAG document reuse (published)', 'serving': 'production serving traces (published)'}
fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(range(len(SET)), [d[1] for d in SET], color=[gcol[d[2]] for d in SET], edgecolor='black', width=0.7)
for i, d in enumerate(SET):
    ax.text(i, d[1] + 1, f'{d[1]:.0f}%' if d[1] == int(d[1]) else f'{d[1]:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
ax.axhspan(50, 100, color='green', alpha=0.06)
ax.text(0.15, 52, '50%+ is the norm in deployed serving', fontsize=13, color='darkgreen', fontweight='bold')
ax.set_xticks(range(len(SET))); ax.set_xticklabels([d[0] for d in SET], fontsize=12)
style_ax(ax, ylabel='cross-query redundancy /\\nhit rate (%)')
ax.set_ylim(0, 82)
handles = [Patch(fc=gcol[g], ec='black') for g in ['ours', 'conv', 'rag', 'serving']]
ax.legend(handles, [glab[g] for g in ['ours', 'conv', 'rag', 'serving']], fontsize=11, loc='upper left')
fig.tight_layout(); fig.savefig('review_c_settings.pdf', bbox_inches='tight', dpi=600); plt.show()"""))
cells.append(md("""**What this shows.** Our KGQA datasets span a **6–42%** cross-query redundancy spectrum.
Published systems report **measured** cross-query reuse far higher on real traffic: SGLang
**52–74%** (a month of live Chatbot Arena), Mooncake **59–66%** (real Kimi traces),
Cache-Craft / RAGCache **60%** (top chunks/docs serve most requests), and conversational KGQA
(SQA) **46%** of turns coreference the prior answer. So 50%+ cross-query overlap is the *norm*
in deployment — the high-overlap regime our method targets is realistic, not cherry-picked,
while it still degrades gracefully when overlap is low."""))

# ===== Generation W_red =====
cells.append(md("## Generation: redundant prefill work `W_red` eliminated (honest, bounded by redundancy)"))
cells.append(code("""red = DATA['red']
fig, ax = plt.subplots(figsize=(7.6, 4.8))
col = {'CWQ': C_DARK, 'WEB': C_MID, 'MQA': C_SKY}
Bs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
for ds in ['CWQ', 'WEB', 'MQA']:
    ys = [1 / (1 - red[ds]['topical'][str(b)]) for b in Bs]
    ax.plot(Bs, ys, '-o', color=col[ds], lw=3, ms=9, markeredgecolor='black',
            markeredgewidth=1.2, label=f"{ds} (max {max(red[ds]['topical'].values())*100:.0f}%)")
ax.axhline(1.0, ls='--', color='grey')
ax.set_xscale('log', base=2); ax.set_xticks([1, 4, 16, 64, 256, 512])
ax.set_xticklabels([1, 4, 16, 64, 256, 512])
style_ax(ax, xlabel='batch size', ylabel='prefill work eliminated (×)',
         title='$W_{red}$ eliminated vs. batch (accuracy-preserving)')
ax.legend(fontsize=15)
fig.tight_layout()
fig.savefig('review_generation_wred.pdf', bbox_inches='tight', dpi=600)
plt.show()"""))
cells.append(md("""**What this shows (honest generation framing).** The cross-query subgraph KV reuse
eliminates the **redundant prefill work `W_red`** (a shared path prefilled once instead of
`r(π)` times). This **grows with batch but is bounded by the data's path redundancy**:
≤~1.6× on CWQ, ~1.2× on WebQSP, ~1.07× on MetaQA. It is **accuracy-preserving** (Opt ≈ R).
In a stock-HuggingFace reference engine this work-reduction is ~parity in wall-clock
(non-contiguous KV); a **contiguous shared-KV-region kernel** converts it to latency (cf.
CacheBlend 2.8–5×, SubGCache 6.68× TTFT). *This replaces the earlier wall-clock multiplier;
the throughput/TTFT figures in `results.ipynb` correspond to the standard-KV-cache /
batch-scaling comparison and are being reconciled with this framing.*"""))

# ===== Reviewer B batch-size sweep (KV on/off config) =====
cells.append(md("## Reviewer B — batch-size sweep (generation throughput, KV caching vs. no caching)"))
cells.append(code("""SWEEP = {'Llama-2-7B': {8:6.2,16:10.6,32:11.7,64:12.6,128:12.7},
         'OPT-6.7B': {8:7.4,16:10.2,32:11.5,64:12.7,128:12.4},
         'OPT-13B': {8:7.7,16:10.8,32:11.8,64:12.5,128:12.8},
         'Gemma-2-2B': {8:3.3,16:6.8,32:9.0,64:10.7,128:13.0},
         'Gemma-2-9B': {8:4.8,16:8.9,32:10.4,64:11.2}}
col = {'Llama-2-7B':C_DARK,'OPT-6.7B':'#d62728','OPT-13B':'#9467bd','Gemma-2-2B':C_SKY,'Gemma-2-9B':'#7f7f7f'}
mk = {'Llama-2-7B':'o','OPT-6.7B':'D','OPT-13B':'v','Gemma-2-2B':'*','Gemma-2-9B':'X'}
fig, ax = plt.subplots(figsize=(8.5, 5))
for m, d in SWEEP.items():
    xs = sorted(d); ax.plot(xs, [d[x] for x in xs], marker=mk[m], ms=11, lw=2.6, color=col[m],
                            markeredgecolor='black', markeredgewidth=1.2, label=m)
ax.axhline(3.12, ls='--', color='green', lw=1.4)
ax.text(8.5, 3.3, "paper's 3.12×", fontsize=12, color='green', fontweight='bold')
ax.set_xscale('log', base=2); ax.set_xticks([8,16,32,64,128]); ax.set_xticklabels([8,16,32,64,128])
style_ax(ax, xlabel='Batch Size', ylabel='generation throughput speedup\\n(KV caching vs. no caching)')
ax.legend(fontsize=13, ncol=2)
fig.tight_layout(); fig.savefig('review_b4_kv_onoff_control_DEPRECATED.pdf', bbox_inches='tight', dpi=600); plt.show()"""))  # DEPRECATED: standard-KV on/off control, not our method; published review_b4_kv_sweep comes from reviewer_b_gen/plot_throughput.py
cells.append(md("""**What this shows.** The reviewer asked for batch-size sensitivity at 16/32/64/128 on
Llama-2 and OPT. With the standard decode KV cache **enabled vs. disabled**, the generation
throughput speedup **holds and grows with batch** across all backbones (Llama-2-7B 6.2→12.7×,
OPT-6.7/13B 7.4→12.8×, Gemma-2 up to 13×), all comfortably above the paper's 3.12×.
*Honest label:* this is the **standard decode-KV-cache (`use_cache`) effect** — the configuration
that reproduces the paper-magnitude number — **not** the cross-query subgraph path reuse (whose
contribution is the accuracy-preserving `W_red` work-reduction in the previous figure). Both are
shown so the batch-size question is answered with real measured numbers."""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.10"}},
      "nbformat": 4, "nbformat_minor": 5}

out = Path("/home/am137688/amir/graphRag/results/review.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print(f"Wrote {out} with {len(cells)} cells")
