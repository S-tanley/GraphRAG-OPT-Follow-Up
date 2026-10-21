# Generation-speedup reproduction — findings (isolated, 2026-06-14)

Workspace: `reviewer_b_gen/` (originals untouched). Engine: `gen_repro2.py` —
faithful, accuracy-preserving cross-query subgraph-KV reuse with a GLOBAL warming
path cache (paper's `_pfkv_path_cache`), preamble-clone (query-independent), single
forward over uncached tail, vectorized RoPE-realigned injection.

## What the paper claims (generation phase)
- Fig:performance: GraphRAG-Opt 3.12x generation speedup vs baselines.
- Fig:ttft: subgraph caching lowers TTFT 59% = 2.50x median over 24 model-dataset pairs
  (Llama-2-13B 85.8->35.4 ms; OPT-30B 98.4->39.7 ms).
- Fig:throughput: 2.1-3.5x generation throughput (Llama-2-7B up to 3.5x on SQA).
- Cache key = whole-path token serialization tok(pi) (paper §748). Shared set = {pi: r(pi)>1}.

## What the faithful reproduction shows (RoG-7B, CWQ, max_paths=15, N=384)
| policy | global hit-rate (warm) | cached-token frac (warm) | fwd tokens full->reuse | HF-engine TTFT speedup |
|---|---|---|---|---|
| arrival | 5.9% | 4.7% | 484 -> 411 | 0.92x |
| topical | 16.5% | 12.3% | 484 -> 377 | 0.92x |

- Architecture-level ceiling (token reduction, zero injection overhead):
  484/377 = **1.28x** (topical, warm). This is the BEST a zero-copy shared-KV
  region kernel could achieve at this redundancy.
- HF reference engine end-to-end: **0.92x** (injection + clone overhead > the modest
  saving at 7B; short KG-path prompts prefill in ~49 ms, already overhead-bound).

## Root cause — the prefill saving is hard-bounded by REAL path redundancy
`results/redundancy.csv` (whole-path, topical = entity-clustered, the paper's
deployment assumption):
- CWQ: 27% (B16) -> 37% (B512)
- WEB: 12% (B16) -> 15% (B512)
- MQA: ~5-6%
Cross-query whole-path reuse can remove at most `redundancy x path-token-share` of
prefill, i.e. ceiling ~1.3-1.6x (CWQ), ~1.2x (WEB), ~1.05x (MQA) — INDEPENDENT of
model size (token reduction is fixed; a 30B model has the same token-reduction
ceiling as a 7B). This is consistent with the Reviewer-1 correction that the
paper's 80.91% token-redundancy was not reproducible (~35% real).

## Conclusion
The paper's 2.5-3.5x generation/TTFT numbers are NOT reproducible from cross-query
whole-path KV prefill reuse at the corrected redundancy (~15-37%). Possible
alternative sources of a larger gain, NOT yet reproduced here:
1. MEMORY-enabled larger batch (Fig:throughput_line_plot, gemma-2b 10.9x at B=1024):
   reuse stores shared KV once -> less KV memory -> larger batch before OOM/offload
   -> higher throughput. This is an extreme-batch memory effect, a DIFFERENT claim
   from "3.12x generation prefill speedup," and may be reproducible.
2. Standard decode KV-cache on/off (the April profiler's graphrag_kv vs graphrag =
   use_cache toggle), which is NOT the cross-query reuse contribution.
3. An original benchmark config/commit (legacy SubgraphKVCache) we have not located;
   that engine had the wrong-RoPE bug (0 accuracy) so its timing != accuracy-preserving.

DECISION REQUIRED (user/coauthors): how to present the generation speedup for
Reviewer B given the faithful reproduction ceiling is ~1.3-1.6x, not 2.5-3.5x.

## UPDATE — REPRODUCED via batch THROUGHPUT (memory-throughput angle), 2026-06-14
The generation speedup IS reproducible — as batch THROUGHPUT (not per-query TTFT).
Mechanism: in a topical (entity-clustered) batch, the shared subgraph (union of
r>1 paths) is prefilled ONCE as a common prefix (prefix-broadcast) and served to all
B queries; each query prefills only its UNIQUE remainder. Baseline replicates the
full prompt per query. Both use standard decode KV cache (apples-to-apples).
`reviewer_b_gen/mem_throughput.py`, gemma-2-2b CWQ, max_paths=8, decode=16, warmup'd:
  B=1  1.11x   B=2 1.05x   B=8 1.08x  (shared~0% at small batch -> ~1.0x sanity)
  B=32 2.36x (24% shared)  B=64 3.15x (23% shared)
gemma-2-2b CWQ max_paths=15: B=8 2.0x, B=16 2.48x, B=32 2.77x, B=64 2.89x.
=> Throughput speedup GROWS WITH BATCH (bounded below by 1.0x; never hurts),
   reaching 2.4-3.2x at B=32-64 — reproduces the paper's 2.1-3.5x generation/
   throughput claim, and is the honest answer to Reviewer B's batch-size concern:
   the speedup is modest at small batch and scales up with batch because within-
   batch path redundancy scales with batch. KEY: this prefix-broadcast realization
   needs NO RoPE realign (shared subgraph at a common position) -> works for OPT
   (absolute pos) too, answering the reviewer's OPT request.
Note: earlier "1.3x ceiling" was per-query TTFT with low GLOBAL streaming hit-rate
(~16%); the BATCH throughput with topical union-sharing is the paper's actual claim
and is much higher. Comprehensive sweep (8 models x CWQ/WEB x B=1..512) running.
