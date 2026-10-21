# PLAN — Legit, literature-aligned Table 2 + reuse story (MICRO 2026 rebuttal)

Execution plan for the next session (Opus). Read `HANDOFF.md` FIRST — it has the
environment, the two reviewer comments, all prior findings, and key file map.
This plan resolves the open decision in HANDOFF §5 as **A+ (hybrid of A and C)**:
keep the honest aligned RoG-plan retrieval (A), and make GraphRAG-Opt's reuse
*defensibly cheap in accuracy* by implementing reuse the way the paper actually
specifies it (shared-only, budgeted, canonically packed) plus selective
recompute (C's spirit, without a multi-day SubGCache reimplementation).

The deliverable: a full-scale Table 2 where every number is (i) produced by one
unified harness, (ii) free of answer leakage, (iii) plausibly close to but NOT
above published literature, (iv) with no duplicated cells and no /64 fractions,
and a reuse-fidelity ablation that answers R1 head-on.

================================================================================
## 0. GROUND RULES (apply to every phase — violations make numbers flaggable)
================================================================================

LITERATURE ANCHORS (RoG ICLR'24 + KGQA SOTA brackets; Hit metric):

| dataset       | RoG published | SOTA bracket | our Direct (RoG model) | our GraphRAG-R (current) |
|---------------|---------------|--------------|------------------------|--------------------------|
| CWQ           | 62.6          | 62–76        | 36.9 (full 3531)       | 50.9 (full 3531)         |
| WebQSP        | 85.7          | 85–92        | TBD                    | TBD                      |
| MetaQA-3hop   | ~84.8 (transfer; in-domain SOTA 98–99) | — | TBD | TBD |

HARD CONSTRAINTS:
1. **No leakage.** Never call `get_paths_bfs` / `get_paths_rog` (they take
   `a_entity` — gold answers — as traversal targets). Only `get_paths_rog_plan`
   (or improvements to it that still never see `a_entity`). Add an assert in the
   retrieval entry point that gold answers are not passed in.
2. **No above-SOTA cells.** If any (model, dataset) cell for a retrieval-augmented
   column exceeds the SOTA bracket (e.g. RoG/CWQ > ~0.63), treat it as a bug
   (leakage/eval artifact), not a win. Investigate before keeping.
3. **Full-scale only.** Final cells come from full test sets: CWQ 3531,
   WebQSP 1628, MetaQA-3hop 14274 (MQA may be a *seeded* fixed subsample of
   3000 if GPU-time-bound — disclose N in the table caption; never 64).
4. **One harness, one protocol.** All columns of a row from `table2_worker.py`
   (one decode config: greedy, fixed max_new_tokens, fixed seed 42, RoG
   reasoning prompt, RoG-standard Hit metric from `metrics.py`).
5. **No cherry-picking.** No per-cell seed/knob tuning. The Opt operating point
   (Phase 2) is chosen ONCE on CWQ+WEB dev slices, then frozen for the table.
6. **Stale numbers are radioactive.** Do not reuse anything from
   `/tmp/table2_final.tsv`, the paper's current Table 2 (85.94% etc.), or the
   REBUTTAL.md ablation tables (N=50/61, computed under *leaky* oracle paths —
   they must be regenerated in Phase 4). Grep the paper for old numbers at the end.
7. Environment: see HANDOFF §0 (conda env `graphrag`, `HF_HUB_OFFLINE=1`,
   8×A100, precomputed paths in `precomputed_paths/`).

================================================================================
## PHASE 1 — Close the retrieval gap legitimately (target: RoG/CWQ Hit 0.55–0.62)
================================================================================

Current aligned number (RoG-plan retrieval + RoG reasoning prompt): CWQ full
0.509 vs published 0.626. A ~5–12 pt reimplementation gap is defensible; 0.509
is borderline. Close it with literature-standard levers only. Iterate on a
FIXED dev slice (CWQ, first 300 by seeded shuffle), then re-precompute.

1a. **Diagnose recall vs generation.** Compute *answer recall* of the retrieved
    paths: fraction of questions where some normalized gold answer appears as a
    terminal/intermediate entity of a retrieved path. RoG's planning retrieval
    achieves ~roughly 0.7–0.8 recall on CWQ in the literature.
    - If recall ≈ Hit ceiling is the binder → fix retrieval (1b).
    - If recall ≫ Hit → fix generation side (1c).

1b. **Retrieval fidelity vs RoG official code** (github RoG `qa_prediction/` +
    `requirements`: plan generation uses beam search, top-3 relation paths,
    grounding keeps ALL instantiated paths). Check ours:
    - `precompute_retrieval.py --num_beam` default is 4; RoG uses 3. Match RoG (3).
    - `max_paths=8` cap: RoG does not cap at 8; it feeds all grounded paths
      (often dozens). Sweep `max_paths ∈ {8, 15, 30, all}` on the dev slice —
      more paths usually = higher Hit (and incidentally a *stronger* reuse story).
    - `_greedy_parse_rule`: compare parsed rules against RoG's own parser on ~50
      generations; fix systematic misparses (doubled domain prefixes already
      handled; check separators/multi-rule outputs).
    - Grounding `_bfs_with_rule`: confirm bidirectional adjacency matches RoG's
      `bfs_with_rule` semantics; confirm fallback chain only fires when primary
      grounding is empty.
1c. **Generation side parity with RoG eval:**
    - `max_new_tokens`: RoG generates full sentence answers; Hit is substring.
      Sweep {32, 64, 128} once on the dev slice; use the smallest value where
      Hit saturates (likely 64–128). Same value for ALL table cells.
    - Confirm `metrics.py` Hit matches RoG's `eval_hit` (normalize + one-direction
      substring over the whole prediction string).
1d. **Acceptance gate:** dev-slice GraphRAG-R Hit ≥ 0.55 on CWQ (stretch 0.58+),
    and spot-check WEB dev slice lands 0.75–0.86. If after 1b/1c CWQ plateaus at
    0.52–0.55, accept it and move on — report as "our reimplementation of RoG
    retrieval"; the paper's contribution is systems, not accuracy SOTA. Do NOT
    reach for gold entities to close the rest of the gap.
1e. **Re-precompute paths for all 3 datasets** with the final retriever config
    (8-GPU, `precompute_retrieval.py`), overwrite `precomputed_paths/*.json`,
    commit them (they're the dataset of record for everything downstream).

================================================================================
## PHASE 2 — GraphRAG-Opt operating point (reuse with ≤2–3 pt honest cost)
================================================================================

Key prior finding (HANDOFF §4): reusing ALL paths collapses accuracy (−10–15
pts); reusing FEW/SHORT shared paths costs ~2 pts. The paper's own spec
(Sec. 6: only r(π)>1 paths pooled, M_pool budget, packing + ordering) is the
*few/short* regime — implement exactly that and the honest number is small.

Mechanisms (mostly built, see `hf_reuse.py`):
  M1. **Shared-only reuse + budget** (`partial_reuse_generate`, `--reuse_budget`):
      reuse only paths shared across ≥2 queries in the batch, admitted by
      priority r(π)−1 under M_pool, everything else computed in-context.
  M2. **Lever-#1 prefix-conditioned encoding** (`encode_prefix_ids`, already
      default `template`): encode each cached path after the shared instruction
      preamble → kills the position-0 attention-sink and bakes in
      path→preamble attention. Amortized, no per-query cost.
  M3. **Prompt canonicalization (NEW, small code change):** order the reused
      shared paths FIRST, immediately after the shared preamble, in a canonical
      (e.g. lexicographic) order; unique paths follow. For query groups whose
      shared set matches the encode-time layout, reused KV is *exact by
      construction* (same prefix, same positions). This is the paper's Phase-2
      "packing/grouping by source entity" realized in prompt space — fully
      defensible, and it directly shrinks the context-dependence error R1 is
      about. Implement in `faithful_pfkv.build_prompt` + `partial_reuse_generate`.
  M4. **Selective recompute** (EPIC first-k / CacheBlend top-r%) on reused
      tokens (`generic_reuse_generate` recompute path + `selective_recompute.py`).

Sweep ONCE on dev slices (CWQ 300 + WEB 300, RoG model), grid:
  reuse_budget ∈ {2, 4, 8} × recompute ∈ {0, epic-2, cb-15%} × M3 ∈ {off, on}.
Record Hit + reused-token fraction (the speedup driver) for each cell.

**Operating point selection rule (frozen before Phase 3):** the config with the
highest reused-token fraction subject to Hit drop ≤ 2–3 pts vs GraphRAG-R on
BOTH dev slices. Expected sweet spot: budget 4–8 + M3 + cb-15% (or epic-2 if
cb's gain is marginal). Sanity: also run reuse-ALL at the operating point's
recompute setting to keep the "collapse" data point for the ablation narrative.

================================================================================
## PHASE 3 — Full-scale Table 2
================================================================================

3a. **Model roster decision (recommended): DROP OPT models.** OPT uses learned
    absolute position embeddings, not RoPE — the paper's position-free-key +
    RoPE-realignment mechanism is undefined for it, and a reviewer who knows OPT
    will flag any OPT GraphRAG-Opt cell. Replace the paper's OPT rows; final
    roster (8 models, all RoPE, all cached locally):
      RoG, Llama-2-7b, Llama-2-13b, Qwen3-8B, Qwen3-32B, Qwen3-Next-80B,
      Gemma-2-9b, Gemma-2-27b.
    Update the paper's "LLM Backbone Models" paragraph accordingly (it also
    currently miscounts: says "three families" then lists four). If the user
    insists on keeping OPT, OPT rows get Direct/GraphRAG columns only with a
    table footnote ("reuse requires RoPE"), never an Opt column.

3b. **Column semantics — fix the R2 trap honestly.** In the unified harness,
    GraphRAG and GraphRAG-R consume identical path sets (JbiDFS is exact), so
    with greedy decoding their *accuracy* is identical BY CONSTRUCTION — two
    byte-identical columns would re-trigger R2. Recommended layout:
      Direct LLM | GraphRAG (†) | GraphRAG-Opt (reuse-only) | GraphRAG-Opt (+recompute)
    with caption footnote †: "GraphRAG-R retrieves the same path set as
    GraphRAG (JbiDFS is exact, Sec. 5) and therefore has identical accuracy;
    the two differ only in retrieval runtime (Fig. 9)." The two Opt columns
    show the honest reuse cost and its recovery — this IS the R1 answer,
    embedded in the main table.

3c. **Run it.** `run_table2.py` over the roster × {CWQ, WEB, MQA} × 4 methods,
    `--paths_dir precomputed_paths`, operating point from Phase 2 baked into
    worker defaults, 8-GPU sharded, model-parallel auto for 32B/80B. Budget
    estimate: ~19k questions × 4 methods × 8 models — schedule MQA possibly as
    seeded N=3000. Persist all TSVs + per-question JSONs under `results_final/`
    (commit the TSVs).

3d. **Automated integrity gate** (write `check_table_integrity.py`):
    - no two cells byte-identical across datasets within a row (R2);
    - no cell above the SOTA bracket (anchors in §0);
    - Direct < GraphRAG for every row;
    - every cell's denominator == full N (or disclosed 3000 for MQA);
    - GraphRAG-Opt(+recompute) within 3 pts of GraphRAG on every row;
    - values not expressible as k/64.
    The table ships only when this script passes. Run it in CI fashion after
    `assemble_table2.py`.

================================================================================
## PHASE 4 — R1 reuse-fidelity ablation, regenerated honestly
================================================================================

The tables currently in `paper/REBUTTAL.md` §3 were computed under the LEAKY
oracle retrieval (0.98 ceilings) — they cannot ship. Regenerate on aligned
RoG-plan retrieval, RoG model, CWQ + WEB, N ≥ 1000 (seeded):
  rows = GraphRAG-R (full prefill) | Opt reuse-only | +EPIC-1 | +EPIC-2 |
         +CacheBlend-15% | reuse-ALL (the collapse point) ;
  cols = Hit, F1, recompute %, reused-token %.
Narrative: reuse error scales with how much context-dependent state is reused;
the paper's budgeted shared-path regime + small recompute recovers GraphRAG-R
accuracy at a fraction of prefill cost — same qualitative law as CacheBlend
Fig. 2. Update REBUTTAL.md §3 + §5 and `docs/REVIEWER_RESPONSE.md` with these
numbers and drop the N=50 tables.

================================================================================
## PHASE 5 — Performance numbers at the SAME operating point
================================================================================

Integrity requires speed and accuracy to describe one config. Re-measure on the
Phase-2 operating point (recompute cost included):
  - generation throughput (Fig. throughput) and TTFT (Fig. ttft) for ≥4 models;
  - retrieval speedup JbiDFS vs BFS (unchanged by this plan, re-run to confirm);
  - end-to-end speedup → update the 2.64×/3.12×/2.71× claims in abstract,
    intro, and evaluation to the measured values (whatever they are — if reuse
    +15% recompute gives 2.4× instead of 3.12×, the paper says 2.4×).

================================================================================
## PHASE 6 — Paper + rebuttal edits (paper/paper.tex)
================================================================================

1. Replace Table 2 (`Table:acc`) with the assembled LaTeX from Phase 3; metric
   named explicitly "Hit (RoG-standard)"; caption discloses full N, footnote †.
2. Fix Table 1 (`Table:Datasets`): sample counts are wrong/swapped (CWQ test =
   3531, WebQSP test = 1628 — paper says 1639/3531) and citations are swapped
   (CWQ is Talmor & Berant 2018 = `talmor2018web`; WebQSP is Yih et al. 2016 =
   `yih2016value`).
3. Apply REBUTTAL.md §4 reframing verbatim (abstract, contribution 3, Sec. 6
   positional-realignment paragraph, related work CacheBlend/EPIC paragraph).
4. Rewrite the "Model Accuracy" paragraph around the new table: drop "up to
   2.83% reduction" and the 36.28% claim; recompute averages from the real
   table; describe the reuse-cost/recovery curve and point to the Phase 4
   ablation.
5. Add 2–3 sentences in Sec. 6 specifying encode-time context (prefix-
   conditioned, canonical packing) — this is the "detailed reuse scheme" R1
   asked for, and Phase 2's M2/M3 make it true.
6. Update speedup numbers per Phase 5. Update model roster text per 3a.
7. Grep the .tex for every stale number: `85.94|93.36|86.94|2.83|36.28|76.56`
   and each old Table-2 value; nothing stale survives.

================================================================================
## PHASE 7 — Final adversarial audit
================================================================================

Fresh-context pass (subagent) that, given only `results_final/` + the .tex:
re-derives every quantitative claim in abstract/intro/eval from the result
files; checks Table 2 against `check_table_integrity.py`; re-runs 20 random
CWQ questions end-to-end and confirms the per-question JSON matches; confirms
no code path touches `a_entity` during retrieval. Fix anything it finds, then
update HANDOFF.md and commit/push.

================================================================================
## RISKS / FALLBACKS
================================================================================
- Phase 1 plateaus below 0.55 CWQ → ship 0.52–0.55 with the "reimplementation"
  framing; the integrity gate cares about above-SOTA and duplication, not about
  being a few points under RoG.
- Phase 2 can't get within 3 pts at meaningful reuse fraction → fall back to
  HANDOFF option C semantics fully: cap the shared pool at ≤1024 tokens
  (SubGCache's regime, cite it as the precedent) — small reuse, near-zero cost,
  honest smaller speedup.
- GPU time tight → priority order: CWQ full > WEB full > MQA(3000); RoG +
  7–9B models first, 32B/80B last (they fill remaining time).
