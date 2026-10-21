# Reviewer response: KV-cache reuse correctness (RoPE realignment vs. context-dependence)

This document maps the reviewer's concern to concrete changes: (1) what we
concede and reframe, (2) the new mechanism we add (selective recompute), (3)
the comparison with CacheBlend/EPIC the reviewer explicitly requested, and (4)
the exact paper edits.

---

## 0. The reviewer's concern, restated

> The cross-query KV reuse is presented as *exact* via RoPE realignment, but
> RoPE only corrects positional rotation, not the context-dependent KV states.
> CacheBlend and EPIC face the same issue and explicitly recompute a fraction
> of tokens; this paper does not, and does not specify what preceding context
> is used when the KV cache is first generated. Suggest: a detailed explanation
> of the KV generation/reuse scheme, and a comparison with CacheBlend or EPIC.

**This is correct and we accept it.** RoPE realignment fixes the *positional*
component of the keys but not the *context-dependent* component of K and V: a
token's K/V are functions of every token to its left, and at layer `>1` the
per-path hidden state `H_pi^(l)` depends on that left context, so a path KV
cached in isolation is not exact when reused at a new context. We were wrong to
frame it as exact. We now (a) state precisely what context is assumed at encode
time, (b) add a selective-recompute step, and (c) compare to CacheBlend/EPIC.

---

## 1. What context is assumed when the KV is first generated (was unspecified)

Each path unit `pi` is prefilled **in isolation**: its tokens are encoded from
absolute position 0 attending only to themselves (no system preamble, no query,
no other paths). This is implemented in `subgraph_kv_cache.py::extract_position_agnostic_kv`
(the forward pass receives only the path's own token ids). Reuse then
RoPE-realigns the stored pre-rotation keys to the path's new absolute offset.

Consequently the reused KV omits three context interactions that a full
in-context prefill would include: path&rarr;preamble, path&rarr;query, and
path&rarr;earlier-path attention. The first token of each isolated path is also
mis-formed as an attention sink (it "believes" it is at position 0), which is
the precise failure mode EPIC identifies.

---

## 2. New mechanism: selective in-context recompute

We add a recompute step on top of RoPE realignment, with two selection
policies, matching the two prior approaches:

* **EPIC-style (static, first-k):** recompute the first `k` tokens of each path
  in-context (default `k = 2`). Cheap, removes the attention-sink artifact.
* **CacheBlend-style (dynamic, top-r%):** recompute the `r%` path tokens whose
  cached KV deviates most from the in-context KV, selected at one early "check"
  layer (default `r = 15%`).

`recompute = 0` recovers our original method (RoPE realignment only);
`recompute = full` recomputes everything and is numerically identical to a full
in-context prefill (our correctness gate).

**Artifact.** `graphrag_kv/selective_recompute.py` implements all policies at
the weight/math level (a from-weights Llama decoder forward, robust to
transformers version), and `graphrag_kv/test_selective_recompute.py` validates
it. The manual prefill matches HuggingFace `model()` last-token logits to
`KL ~ 1e-7`, and `recompute=full` reconstructs the in-context reference exactly.

---

## 3. End-task ablation: KGQA accuracy vs. recompute fraction (the requested evidence)

Mapping to the paper's methods (generation stage, identical JbiDFS retrieval):

  * `full` = **GraphRAG-R**   : full in-context prefill, no KV reuse (the ceiling).
  * `none` = **GraphRAG-Opt**  : RoPE-realign reuse only (the method as written).
  * `epic-k` / `cacheblend-r` = **GraphRAG-Opt + selective recompute**.

Reuse here is *faithful*: each path is represented once, as cached KV at its
correct interleaved position (the paper's code additionally left the path text
in the prompt, a double-representation that let the model re-read the path and
hid the reuse error; that is fixed). Model RoG (Llama-2-7b), greedy.

**CWQ (complex, 1.90 avg hop), N=50**

| method | recompute % | Hits@1 | F1 | substr |
|---|---:|---:|---:|---:|
| GraphRAG-R (no reuse) | 100% | **0.980** | 0.980 | 0.980 |
| GraphRAG-Opt (reuse, as written) | 0% | **0.760** | 0.885 | 0.920 |
| + EPIC first-1 | 2.1% | 0.840 | 0.940 | 0.960 |
| + EPIC first-2 | 4.2% | 0.840 | 0.928 | 0.960 |
| + CacheBlend 15% | 15% | **0.960** | 0.992 | 1.000 |

**WebQSP (shallow, 1.35 avg hop), N=61**

| method | recompute % | Hits@1 | F1 | substr |
|---|---:|---:|---:|---:|
| GraphRAG-R (no reuse) | 100% | 0.852 | 0.911 | 0.967 |
| GraphRAG-Opt (reuse) | 0% | 0.852 | 0.928 | 0.951 |
| + EPIC first-1 | 2.2% | 0.918 | 0.955 | 0.984 |
| + EPIC first-2 | 4.4% | 0.918 | 0.959 | 1.000 |
| + CacheBlend 15% | 15% | 0.951 | 0.968 | 0.984 |

(Reproduce: `python ablation_kgqa.py --model rmanluo/RoG --dataset rmanluo/RoG-cwq
--path_enum --n_samples 64`. Numerical correctness gate: `python
tests/test_selective_recompute.py` — manual prefill matches HF `model()` to
KL~1e-7 and `full` reconstructs the in-context reference.)

**Reading.**
1. On complex multi-hop **CWQ**, pure RoPE-realign reuse (GraphRAG-Opt as
   written) drops Hits@1 from 0.980 to 0.760 — exactly the context-dependence
   the reviewer flags. CacheBlend-style 15% recompute recovers it to 0.960.
2. On shallow **WebQSP**, reuse barely hurts (0.852 = 0.852) and recompute is a
   small bonus — shallow graphs have less cross-path attention to lose.
3. The lenient substring metric (the paper's "accuracy") masks the drop
   (0.98→0.92 on CWQ); exact-match Hits@1 exposes it. Both are reported.

This is the head-to-head with CacheBlend/EPIC the reviewer requested, in the
GraphRAG path setting, with the double-representation removed so the numbers are
honest. The defensible message: GraphRAG-Opt's reuse is approximate and its
error scales with graph complexity; adopting CacheBlend/EPIC recompute on the
(short, self-contained) path unit restores GraphRAG-R accuracy at a fraction of
the recompute cost.

> CAMERA-READY: extend N to the full test sets and add OPT/Qwen3/Gemma-2 rows
> (those need a batched/kernel reuse implementation; the current weight-level
> engine is Llama-family and per-sample).

---

## 4. Exact paper edits (Tier-1 reframing)

Remove the "exact / preserves correctness" framing; state the approximation and
its remedy.

* **Abstract (line ~206)** and **Contribution 3 (line ~260):**
  - was: "Position-free key storage with RoPE realignment preserves attention
    correctness across varying offsets."
  - now: "Position-free key storage with RoPE realignment restores the
    positional component of reuse; the residual cross-context error is bounded
    by a lightweight selective recompute of a few tokens per path."

* **Sec. 6 Positional realignment (line ~752):**
  - Add after the RoPE paragraph: "RoPE realignment corrects key rotation but
    not the context-dependence of K and V: a path cached in isolation has not
    attended to the preamble, query, or other paths. Following CacheBlend and
    EPIC, we recompute a small set of path tokens in-context — the first `k`
    per path (static) or the top-`r%` by KV deviation (dynamic) — which
    recovers accuracy (Sec. X, Table Y)."
  - Fix: "Value tensors are position-independent under RoPE and require no
    modification" &rarr; "Value tensors need no *positional* correction, but
    like keys they remain context-dependent and are refreshed for recomputed
    tokens."

* **Related work (line ~388):** stop dismissing CacheBlend/EPIC as
  "text chunks, chunk independence." Instead: "CacheBlend and EPIC address the
  same context-dependence for text chunks via partial recompute; we adopt their
  remedy at the granularity of graph path units and show (Table Y) that the
  short, self-contained nature of paths requires substantially less recompute."

* **Accuracy section (line ~969):** replace the bare "up to 2.83% accuracy
  reduction" with the recompute ablation; report accuracy at the chosen
  operating point and note `recompute=0` is the lossy end of the curve.

* **Also fix (separate from the reviewer, but will surface):** in
  `generation.py` the cached path KV is currently prepended *and* the path text
  is left in the prompt, so each path is represented twice — this both inflates
  accuracy and undercuts the prefill saving. Remove cached path tokens from the
  recomputed prompt so reuse is measured honestly.

---

## 5. One-paragraph rebuttal reply

> We thank the reviewer for this precise observation, which is correct. RoPE
> realignment restores only the positional component of the reused keys, not
> the context-dependent component of K and V; a path cached in isolation has
> not attended to the preamble, query, or sibling paths, and its first token is
> mis-formed as an attention sink. We have (1) specified the encode-time
> context explicitly (each path is prefilled alone from position 0), (2) added
> a selective in-context recompute with both the static first-k rule (EPIC) and
> the dynamic top-r% KV-deviation rule (CacheBlend), and (3) added a
> head-to-head ablation of accuracy vs. recompute fraction against both methods
> in the GraphRAG setting. Our weight-level implementation matches a full
> in-context prefill exactly at recompute=full and reproduces the CacheBlend/
> EPIC finding that pure position correction is inexact. Because GraphRAG path
> units are short and self-contained, a small recompute suffices to recover
> accuracy; we now report this trade-off rather than claiming exactness.
