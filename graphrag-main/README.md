# GraphRAG-Opt — Reproduction & Project Handoff

Research code and reproduction harness for **GraphRAG-Opt**, a system that
accelerates Graph Retrieval-Augmented Generation (GraphRAG) by (1) a join-based
bidirectional path-enumeration algorithm (**JbiDFS**) for retrieval and (2)
**cross-query subgraph KV-cache reuse** with positional realignment for
generation. Submitted to **MICRO 2026**; paper source is separate (`../paper/paper.tex`).

This README is the canonical handoff: it lets a new person set up the
environment, understand the codebase, and **reproduce every number** (the
accuracy table, the redundancy analysis, the CacheBlend/EPIC ablation, the
large-model reuse). It also records the honest caveats and known issues found
while preparing the MICRO rebuttal.

---

## 1. TL;DR — what this repo does

The reusable unit of GraphRAG is redefined from the **token sequence** to the
**hop-constrained entity–relation path**. Retrieval enumerates path units;
generation consumes them through their cached KV state; a path recurring across
queries in a batch is prefilled **once** and reused.

Evaluation measures **KGQA accuracy** under up to five operating points, across
11–12 LLM backbones and three datasets, plus systems analyses (redundancy, ablation):

| Column | Meaning |
|---|---|
| **Direct** | LLM, no retrieval (lower bound) |
| **GraphRAG** | retrieval on, BFS enumeration (RoG baseline) |
| **GraphRAG-R** | retrieval on, JbiDFS enumeration — *same paths as BFS, faster* |
| **GraphRAG-Opt** | JbiDFS + cross-query KV reuse (RoPE-realigned) |
| **Opt+RC** | GraphRAG-Opt + selective recompute (EPIC first-k / CacheBlend top-r%) |

**Headline:** cross-query KV reuse preserves accuracy — `GraphRAG-Opt` is
statistically **non-inferior to full prefill in 22/27 model–dataset cells**
(paired McNemar, 2.5-pt margin), across backbones from OPT-6.7B up to a
model-parallel **Qwen2.5-72B**, and across RoPE and non-RoPE (OPT) architectures.

---

## 2. Environment

```bash
source /home/<user>/anaconda3/etc/profile.d/conda.sh
conda activate graphrag                       # torch 2.10, transformers 5.2.0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 # models/datasets are cached locally
```

- **Hardware used:** 8× NVIDIA A100-80GB (single node). ≤32B models run one-GPU
  data-parallel; 70B/72B run model-parallel (`device_map="auto"`).
- **⚠ Do NOT upgrade `transformers`.** The reuse engine needs transformers 5.2's
  `DynamicCache.layers[i].keys/.values` API. `gptqmodel`/some quant backends
  silently bump transformers and break it.
- **⚠ `results/` is in `.gitignore`** — new files there are not auto-committed;
  use `git add -f results/<file>` for any deliverable.
- Deps: `torch`, `transformers==5.2.0`, `datasets`, `networkx`, `numpy`,
  `accelerate`, `bitsandbytes` (optional, 4-bit), `sentence-transformers` (dense
  retrieval baseline only). `pip install -r requirements.txt`.

---

## 3. Repository layout

```
graphrag_kv/
├── README.md               ← this file (canonical handoff)
├── HANDOFF.md              ← detailed chronological log + per-reviewer rebuttal state
│
│  ── accuracy/table pipeline (primary) ──
├── run_table2.py           orchestrator: spawns workers, shards GPUs, scores, writes TSV
├── table2_worker.py        one (model,dataset,method) cell; greedy decode; writes preds JSON
├── hf_reuse.py             KV-REUSE ENGINE: prefix-conditioned encode, RoPE realign,
│                           recompute knob, DEVICE-AWARE (model-parallel), no-realign toggle
├── faithful_pfkv.py        prompt build (RoG reader format), recompute POLICY_SPECS, encode prefix
├── selective_recompute.py  weight-level Llama engine + EPIC/CacheBlend token selection
├── metrics.py              RoG-standard Hit / F1 / Acc
├── graph_utils.py          JbiDFS (bidirectional prune + midpoint join + DFS) + retrieval
├── retrieval.py            batch-optimized retrieval wrapper
├── generation.py           generation utilities (RoG tokenizer fix, decode helpers)
├── metaqa_adapter.py       MetaQA loader (WikiMovies KG)
├── config.py               seeds / fast-mode presets (256-sample RoG protocol)
│
│  ── single-query / systems entry point ──
├── main.py                 dataset → retrieval → generation → metrics (CLI, --fast_mode)
│
│  ── retrieval + data ──
├── precompute_retrieval.py builds RoG-plan paths once per dataset (8-GPU sharded)
├── precomputed_paths/      paths_<dataset>.json — RoG-plan retrieval, leak-free, PERSISTED
│   ├── paths_rmanluo-RoG-cwq.json   (3,531)   paths_rmanluo-RoG-webqsp.json (1,628)
│   └── paths_metaqa-3hop.json       (14,274)
├── metaQA/                 MetaQA WikiMovies KG (kb.txt) + splits
│
│  ── experiments / analysis ──
├── exp_batch_redundancy.py within-batch path/token redundancy vs batching policy
├── ablation_kgqa.py        CacheBlend/EPIC head-to-head ablation driver
├── run_full_ablation.py    ablation sweep wrapper
├── assemble_table2.py      merge result TSVs → table + Wilson CI + paired McNemar
├── make_final_table.py     final Table 2 (.tex + .txt, max-per-row highlight)
├── check_paper_coverage.py map result TSVs → paper Table 2 cells
├── check_literature_alignment.py recall / below-SOTA / leakage check per dataset
│
├── results/                outputs (gitignored; key files force-added) — see §7
├── tests/                  numerical gates: test_selective_recompute / test_prefill / test_metaqa
└── *.sh                    detached pipeline drivers (see §6)
```

Other dirs (`reviewer_b_gen/`, `reviewer_cde/`, `workload_fig6/`, `scheduling/`,
`profile_results/`, `workload_results/`) hold earlier systems-side analyses
(generation throughput, workload characterization, scheduling) — not needed for
the accuracy reproduction.

---

## 4. Datasets & retrieval

| Dataset | KG | Test size | Note |
|---|---|---|---|
| **RoG-cwq** (CWQ) | Freebase | 3,531 | complex multi-hop |
| **RoG-webqsp** (WEB) | Freebase | 1,628 | shallower |
| **MetaQA-3hop** (MQA) | WikiMovies | 14,274 | movie KG |

Retrieval is **supervised, leakage-free RoG-plan retrieval** (the RoG model
generates relation paths grounded by `bfs_with_rule`; the answer entity is never
used). Paths are **precomputed once** into `precomputed_paths/` so every method
and backbone consumes identical inputs. Rebuild:

```bash
python precompute_retrieval.py --dataset rmanluo/RoG-webqsp --num_gpus 8 \
  --out precomputed_paths/paths_rmanluo-RoG-webqsp.json
```

> An earlier pipeline used `get_paths_bfs(q_entity, a_entity)`, which retrieves
> paths *to the answer* → leakage → above-SOTA numbers. NOT used; all reported
> numbers use the leak-free RoG-plan paths.

---

## 5. Reproducing the results

Env from §2 active. Metrics (`metrics.py`): Hit = any gold answer is a substring
of the prediction; entity-set F1; Acc = fraction of gold found. Greedy decode ⇒
deterministic (no run-to-run variance).

### 5.1 Accuracy table (Table 2)
```bash
python run_table2.py \
  --models rmanluo/RoG meta-llama/Llama-2-7b-hf meta-llama/Llama-2-13b-hf \
           google/gemma-2-9b google/gemma-2-27b Qwen/Qwen3-8B Qwen/Qwen3-32B \
           facebook/opt-6.7b facebook/opt-13b facebook/opt-30b \
  --datasets rmanluo/RoG-cwq rmanluo/RoG-webqsp metaqa-3hop \
  --paths_dir precomputed_paths --max_new_tokens 64 --n_samples 500 \
  --results results/table2.tsv --outdir /tmp/t2
```
- `--n_samples -1` = full split (primary small models, CWQ/WEB); `500` = fixed-seed
  subset (seed 42, identical IDs across systems) for expensive cells.
- `--methods direct graphrag_r graphrag_opt graphrag_opt_rc` (default = all).
- Gemma auto-uses bf16 (fp16 overflows → 0.0).

Large model (Qwen2.5-72B, **model-parallel** KV reuse):
```bash
bash run_qwen72.sh          # 72B bf16 across 4 GPUs ×2 replicas, all 4 methods, N=500
```
Assemble + score:
```bash
python assemble_table2.py --metric hits --latex \
  --tsv results/table2*.tsv --preds results/preds_* > results/FINAL_TABLE_hits.txt
python make_final_table.py        # → results/PAPER_TABLE2_final.{tex,txt}
python check_paper_coverage.py    # coverage vs the paper's Table 2
```

### 5.2 Within-batch redundancy
```bash
python exp_batch_redundancy.py    # → results/redundancy.csv
```
Path/token redundancy under sequential / random / topical batching × batch size.
Finding: sequential ≈ random (no file-order artifact); topical/entity-clustered is
where redundancy lives (CWQ 36% / WEB 15% / MQA 6% at large B).

### 5.3 CacheBlend / EPIC ablation (reuse vs recompute)
```bash
python ablation_kgqa.py --model rmanluo/RoG --dataset rmanluo/RoG-cwq --path_enum \
  --n_samples 500 --policies full,none,epic1,epic2,cacheblend2,cacheblend4,cacheblend15 \
  --encode_prefix template > results/ablation_rog_cwq.txt
```
Finding: on short KG paths, selective recompute recovers ≤1 pt (vs ~6.8-pt
full-reuse cost) — opposite of the long-text-chunk regime CacheBlend/EPIC target.

### 5.4 SubGCache-fidelity proxy + realignment ablation
```bash
GROPT_NO_REALIGN=1 CUDA_VISIBLE_DEVICES=0 python table2_worker.py \
  --model rmanluo/RoG --dataset rmanluo/RoG-cwq --method graphrag_opt \
  --reuse_budget 8 --recompute_k 0 --n_samples 500 \
  --paths_file precomputed_paths/paths_rmanluo-RoG-cwq.json --out_json /tmp/x.json
```
`GROPT_NO_REALIGN=1` disables RoPE realignment (the SubGCache design). Finding:
no realignment loses 5–10 pts (RoG/CWQ 0.510→0.408), recovered by realignment.

### 5.5 Single-query / systems entry point (`main.py`)
```bash
python main.py --fast_mode neurips --retrieve --path_enum \
  --dataset rmanluo/RoG-cwq --llm_model rmanluo/RoG          # baseline
python main.py --fast_mode neurips --retrieve --path_enum --use_pfkv \
  --dataset rmanluo/RoG-cwq --llm_model rmanluo/RoG          # + subgraph KV reuse
```
`--fast_mode` presets in `config.py`; `--batch_opt` for batch endpoint dedup;
`--parallel_mode data_parallel|model_parallel`.

### 5.6 Tests
```bash
python -m pytest tests/      # or: python tests/test_selective_recompute.py
```
Checks (i) manual prefill == HF `model()` to KL ~1e-7, (ii) `full` recompute
reconstructs the in-context reference, (iii) `none` is measurably inexact while
`epic`/`cacheblend` recover it.

---

## 6. Detached pipeline scripts (`*.sh`)

Run the full matrix unattended (each commits + pushes per model). Launch in tmux
so they survive logout: `tmux new -d -s run 'bash <script>.sh'`. **Always check
`tmux ls` first** — concurrent pipelines OOM the GPUs.

- `run_qwen72.sh` — Qwen2.5-72B model-parallel matrix.
- `run_full_ablation.py` — ablation sweep.
- Historical drivers: `run_all_remaining.sh`, `finish_gaps.sh`, `run_opt_reuse.sh`
  (see HANDOFF.md for what each did).

---

## 7. Results & deliverables (`results/`)

`results/` is gitignored; important artifacts are force-added:

| File | What |
|---|---|
| `FINAL_TABLE_hits.txt` / `_f1.txt` | full Table 2, Wilson 95% CI + paired McNemar |
| `PAPER_TABLE2_final.tex` / `.txt` | final 12-model table, max-per-row highlight |
| `redundancy.csv` | within-batch redundancy vs batching policy |
| `ablation_rog_cwq.txt` | CacheBlend/EPIC head-to-head |
| `subg_*.json` | SubGCache-fidelity proxy predictions (±realign) |
| `literature_alignment.txt` | recall / below-SOTA / leakage check |
| `REBUTTAL_MICRO.md` | full reviewer-rebuttal evidence/working doc |
| `reviewer_1.md` | clean, paste-ready response to Reviewer 1 |

`table2_*.tsv` hold raw per-cell numbers; `preds_*/` hold raw predictions so any
metric can be recomputed.

---

## 8. The KV-reuse engine (how it works)

`hf_reuse.py` + `faithful_pfkv.py`:

1. **Encode (prefix-conditioned).** Each path `π` is encoded as `model(P ⊕ π)`
   behind a *fixed, query-independent* instruction prefix `P` (RoG reader
   preamble); only `π`'s KV slice is kept. `P` is identical at generate and reuse
   time → KV is globally shareable; the position-0 attention sink is removed.
2. **Position-free keys** `K̃ = W_k H` stored (before rotation); values unchanged.
3. **Reuse + realign.** At a new offset, keys are re-rotated by the position delta
   (`_realign_k`, HF rotate_half RoPE). Exact for *position* (RoPE is relative),
   but does NOT restore cross-path attention content — the same chunk-independence
   approximation as CacheBlend/EPIC.
4. **Recompute knob.** `recompute_k` (EPIC first-k) or CacheBlend top-r%
   (`POLICY_SPECS`) recompute a fraction in-context to recover accuracy.
5. **Device-aware.** `_layer_kv_device` puts each layer's reused KV on that
   layer's GPU → reuse works under `device_map="auto"` (70B/72B model-parallel).
6. **Ablation toggle.** `GROPT_NO_REALIGN=1` disables step 3.

`graph_utils.py` JbiDFS: bidirectional BFS prune to `V'={v:d_s(v)+d_t(v)≤k}`,
**shortest-path midpoint** join, Cartesian recombination, residual DFS.

---

## 9. Known issues & honest caveats

Recorded so the next person isn't surprised; several were found while matching
claims to code for the MICRO rebuttal (see `results/reviewer_1.md`):

- **RoG tokenizer bug.** RoG's bundled tokenizer mis-decodes under transformers
  ≥5.x (sub-words space-joined, `Shia Islam`→`Sh ia I sl am`), silently zeroing
  exact-match. The harness loads `meta-llama/Llama-2-7b-hf`'s tokenizer for RoG
  (identical vocab) — keep this.
- **Join selection is shortest-path midpoint**, not a `C(v)`-minimizing scan; the
  speedup comes from the bidirectional prune + batch dedup, not the join choice.
- **Retrieval graph is undirected** (`nx.Graph()`); path sets are scoped by the
  per-query `max_paths` budget (not provably exhaustive).
- **Qwen3-80B reuse is N/A** — gated-DeltaNet *linear attention* has no per-layer
  KV to slice/realign. Report Direct/GraphRAG-R only; use **Qwen2.5-72B** for the
  large-model reuse row.
- **4-bit (bitsandbytes) load OOMs for 70B** under transformers 5.2's loader
  (materializes bf16 first). Use **model-parallel bf16** for large models.
- **Gemma in fp16 overflows → 0.0**; worker auto-selects bf16 for Gemma.
- **`results/` gitignored** → force-add deliverables.
- The paper's original `Fig:red` token-redundancy (80.91%) is not reproducible
  (~35% measured); CFQ runtime in Fig 10 is a projection (per-query × |dataset|,
  256-sample measurement on 1×A100), not a 46-day run.

---

## 10. Status

- **Accuracy table:** complete — 11 paper backbones + Qwen2.5-72B, 3 datasets,
  5 columns, leak-free, below-SOTA, CIs + McNemar (Qwen3-80B reuse cells = n.a.).
- **Reviewer-1 rebuttal:** fully addressed; deliverables in `results/`.
- **Paper** `../paper/paper.tex` is intentionally **not modified** here; corrected
  text snippets live in `results/reviewer_1.md` / `REBUTTAL_MICRO.md` ready to
  apply, and figures regenerate from `redundancy.csv` / `PAPER_TABLE2_final.tex`.
- **GPUs idle.** Safe to clone, set up, and re-run.

See `HANDOFF.md` for the full chronological log, tmux-pipeline conventions, and
per-reviewer rebuttal details.
