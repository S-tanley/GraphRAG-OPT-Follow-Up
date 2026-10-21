# Methodology grounding — how our setup maps to the standard implementations

Written after auditing the canonical repos/papers (RoG, GNN-RAG, SubGCache,
CacheBlend, EPIC/LegoLink) against our code. Every non-obvious choice below is
justified by a primary source. This doc backs the R1 rebuttal (CacheBlend/EPIC
comparison the reviewer asked for) and records why our numbers are the standard
protocol, not an ad-hoc one.

================================================================================
## 1. Retrieval (CWQ, WebQSP) — matches RoG
================================================================================
RoG (Luo et al., ICLR'24) protocol, verified from RManLuo/reasoning-on-graphs:
- Plan relation paths with beam search: `num_beams == num_return_sequences`,
  top-K=3, greedy (`do_sample=False`), `max_new_tokens≈100`. Instruction:
  "Please generate a valid relation path that can be helpful for answering the
  following question: ".
- Ground each relation sequence by **exact-relation constrained BFS** on an
  **undirected** graph (`bfs_with_rule`); the count cap (`max_p=10`) is disabled
  in their code, so all instantiations are kept; prompt trimmed to a **4096-token
  budget** (shuffle then greedily fill), NOT a fixed path count.

OURS (`graph_utils.get_paths_rog_plan`, `precompute_retrieval.py`):
- RoG model plans paths; we use beam=4 (≥ RoG's 3 — strictly more coverage,
  still beam search). Grounded with `_bfs_with_rule` = RoG's `bfs_with_rule`
  (exact relation match), bidirectional adjacency = undirected (matches RoG).
- We cap at `max_paths=30` (≈450 tokens) instead of RoG's 4096-token budget.
  This is the one deliberate deviation: fewer paths than RoG. Consequence: our
  answer-recall ceiling is 0.644 (CWQ) — already **above** RoG's reported Hit
  (0.626) — so the cap is not what bounds us; reader extraction is. Defensible.
- NO answer entity is ever used in retrieval (assert this). RoG is leak-free too.

================================================================================
## 2. Evaluation metric — byte-identical to RoG/GNN-RAG
================================================================================
RoG/GNN-RAG `evaluate_results.py`:
  normalize = lowercase → strip punctuation → strip articles (a|an|the) →
              strip <pad> → collapse whitespace
  match(pred, gold) = normalize(gold) in normalize(pred)        # substring
  Hit = 1 if ANY gold answer matches (this is RoG's headline "Hits@1" column).
  F1  = over predicted-answer list; P=matched/|pred|, R=matched/|gold|.

OURS (`metrics.py`): identical normalize + substring `_match` + Hit (any-gold) +
list-F1. So our 0.58 (CWQ) / 0.835 (WEB) are directly comparable to RoG's
**Hit** numbers 0.626 / 0.857 — NOT to strict top-1 (GNN-RAG reports that
separately as H@1). Anchor bands: WebQSP Hit ~80–91, CWQ Hit ~57–69.

================================================================================
## 3. Reader prompt — RoG's exact Llama-2-chat template
================================================================================
RoG `prompts/llama2_predict.txt` + `SAQ_RULE_INSTRUCTION`:
  [INST] <<SYS>>\n<</SYS>>\nBased on the reasoning paths, please answer the given
  question. Please keep the answer as simple as possible and return all the
  possible answers as a list.\n\nReasoning Paths:\n{paths}\n\nQuestion:\n{q} [/INST]

OURS (`faithful_pfkv.build_prompt`, `hf_reuse.build_preamble_suffix`,
`_ENCODE_PREFIX`): now byte-for-byte this template (was previously missing the
`<<SYS>>` block and added a non-standard "Answer:"). All three builders share it
so the full-prefill (GraphRAG-R) and reuse (GraphRAG-Opt) paths see ONE prompt.
Non-RoG backbones use a generic few-shot template (they are not KGQA-tuned).

### The bug this fixed (important)
The reuse path used a DIFFERENT (Alpaca `### Instruction/### Response`) prompt
than the full-prefill path. Because the RoG reader is fine-tuned on `[INST]`,
this mismatch — not reuse — caused an apparent 14–19pt "reuse collapse". After
unifying the prompt, the measured reuse cost is 0.5–1pt (≈0 with recompute),
consistent with the literature below. Lesson: always sanity-check reuse with
`reuse_budget=0`, which must equal GraphRAG-R exactly (it does: 0.595==0.595).

================================================================================
## 4. KV-cache reuse — our method vs SubGCache / CacheBlend / EPIC
================================================================================
Shared root cause (CacheBlend states it cleanly): "the KV cache of a non-prefix
text chunk ignores the cross-attention between the chunk and the preceding
text." RoPE realignment fixes POSITION, not this context-dependence — exactly R1.

| method      | shared unit                     | tokens recomputed            | acc gap vs full prefill |
|-------------|---------------------------------|------------------------------|-------------------------|
| SubGCache   | whole representative subgraph (prefix block) | 0% (reuse as-is) | −1% to +9% (small graphs) |
| CacheBlend  | every non-prefix chunk          | ~15% (HKVD deviation-select) | within 0.02 F1          |
| EPIC/LegoLink | every non-prefix chunk        | first k=2 tokens/chunk (sink)| 0–7%                    |
| **OURS**    | cross-query SHARED paths (r>1), budgeted | EPIC first-k (k=2 default) on reused paths | **0.5–1pt (≈0 @ k=2)** |

Key points for the rebuttal:
- SubGCache (the paper's own cited prior work, zhu2025subgcache) reuses the
  WHOLE shared subgraph with ZERO recompute and stays accurate BECAUSE its unit
  is a stable prefix on tiny graphs (22–1071 nodes; token length unreported).
  It does NOT detail positional handling — our RoPE realignment is the
  contribution that lets the shared unit sit at VARYING (non-prefix) positions.
- For non-prefix reuse (our case: a path recurs at different offsets across
  queries), CacheBlend and EPIC both show a small selective recompute is
  REQUIRED to stay near full-prefill accuracy. We adopt EPIC's first-k (k=2,
  LegoLink-2) on the reused path units. This is the head-to-head R1 asked for.
- Honest framing (replaces the paper's "exact / preserves correctness"): reuse
  is approximate; the residual cross-context error is bounded by recomputing a
  few tokens per reused path. Our operating point (reuse_budget=4 shared paths,
  recompute_k=2) recovers GraphRAG-R accuracy.

================================================================================
## 5. MetaQA — why it is NOT in the accuracy table
================================================================================
Verified: the strong MetaQA numbers in the literature ALL use in-domain
supervision. RoG's MetaQA-3hop (84.81 scratch / 88.98 transfer) BOTH fine-tune
on 1000 MetaQA training samples; in-domain models (NSM/UniKGQA/GNN-RAG) are
saturated at 98–99% because MetaQA has only 9 relations + templated questions.
There is NO canonical unsupervised/zero-shot MetaQA retriever.

Our unsupervised dense retriever (MiniLM over relation paths, no MetaQA training)
gives recall 0.60 → Hit ~0.45 — far below 84.8 because it is off-protocol, not
because the pipeline is wrong. Putting 0.45 next to a literature of 84.8–99 would
be flagged. The paper's own text already says accuracy is compared on "only two
datasets (CWQ, WEB)". Decision: Table 2 = CWQ + WEB; MetaQA stays in the
workload/throughput characterization (where retrieval recall does not gate the
systems metrics). To put MetaQA in the accuracy table the standard way one must
fine-tune the planner on ~1k MetaQA samples (RoG's transfer recipe) — a bounded
follow-up if desired.

================================================================================
## 6. Model roster — drop OPT (no RoPE)
================================================================================
OPT uses learned ABSOLUTE position embeddings, so position-free-key + RoPE
realignment (the GraphRAG-Opt reuse mechanism) is undefined for it; any OPT Opt
cell would be unjustifiable. Final roster = 8 RoPE models: RoG, Llama-2-7b,
Llama-2-13b, Gemma-2-9b, Gemma-2-27b, Qwen3-8B, Qwen3-32B, Qwen3-Next-80B.
(The paper currently also miscounts "three families" then lists four.)

================================================================================
## Sources
================================================================================
RoG: github.com/RManLuo/reasoning-on-graphs, arxiv 2310.01061
GNN-RAG: github.com/cmavro/GNN-RAG, arxiv 2405.20139
SubGCache: arxiv 2505.10951 (AAAI 2026)
CacheBlend: github.com/YaoJiayi/CacheBlend, arxiv 2405.16444
EPIC/LegoLink: github.com/DerekHJH/epic, arxiv 2410.15332
