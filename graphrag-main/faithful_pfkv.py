"""
Faithful subgraph KV reuse for the generation stage (shared by the main pipeline
and the ablation harness).

"Faithful" = each retrieved path is represented exactly once, as cached KV at its
correct interleaved position in the prompt (NOT also as prompt text — that
double-representation lets the model re-read the path and hides the reuse error).
Reused path tokens are served from the isolated cache (RoPE-realigned); a
configurable subset is recomputed in-context (EPIC first-k / CacheBlend top-r%).

Llama-family models only (uses the weight-level engine in selective_recompute.py).
"""

import re

import torch

from selective_recompute import select_recompute_tokens

CHECK_LAYER = 1   # CacheBlend selects high-deviation tokens from one early layer

POLICY_SPECS = {
    "none": dict(policy="none"),
    "epic1": dict(policy="epic", k=1),
    "epic2": dict(policy="epic", k=2),
    "epic4": dict(policy="epic", k=4),
    "cacheblend2": dict(policy="cacheblend", ratio=0.02),
    "cacheblend4": dict(policy="cacheblend", ratio=0.04),
    "cacheblend15": dict(policy="cacheblend", ratio=0.15),
    "cacheblend30": dict(policy="cacheblend", ratio=0.30),
    "full": dict(policy="full"),
}

_FEWSHOT = (
    "You are a concise, factual assistant. Based on the following context, answer "
    "with the most relevant short phrase only. Do not add explanation.\n\n"
    "Q: What is the capital of France?\nA: Paris\n\n"
    "Q: Who was the first president of the United States?\nA: George Washington\n\n"
    "Context (knowledge graph paths):\n"
)

# Query-independent prefixes used to *condition* a path's cached KV (lever #1):
# encoding [prefix + path] and keeping the path slice gives the path KV attention
# to a representative instruction context and removes the position-0 attention
# sink, while staying shared across all queries (so global reuse is preserved).
# MUST be the same preamble the path actually sits behind at reuse time (see
# hf_reuse.build_preamble_suffix / build_prompt), so a path's isolated-encode
# left-context matches its in-prompt left-context as closely as possible.
_ENCODE_PREFIX = {
    "rog": ("[INST] <<SYS>>\n<</SYS>>\nBased on the reasoning paths, please answer "
            "the given question. Please keep the answer as simple as possible and "
            "return all the possible answers as a list.\n\nReasoning Paths:\n"),
    "fewshot": _FEWSHOT,
}


def encode_prefix_ids(tok, template):
    """Token ids of the query-independent encode prefix for a template."""
    return tok.encode(_ENCODE_PREFIX.get(template, _ENCODE_PREFIX["rog"]),
                      add_special_tokens=True)


def path_to_str_rog(path):
    if isinstance(path, str):
        return path
    if not path:
        return ""
    parts = [path[0][0]]
    for _h, r, t in path:
        parts.append(r)
        parts.append(t)
    return " -> ".join(parts)


def build_prompt(tok, query, paths, max_paths, template="rog"):
    """Return (full_ids, per-path spans, index of last token).
    Path lines are tokenized independently so a path's tokens in the full
    sequence are identical to its isolated-cache tokens.

    template='rog'     : RoG Alpaca instruction format (KGQA-tuned models).
    template='fewshot' : generic few-shot format with exact-answer demos.
    """
    if template == "fewshot":
        pre = _FEWSHOT
        suffix_text = f"\nQ: {query}\nA:"
    else:
        # RoG's EXACT reader prompt (prompts/llama2_predict.txt + SAQ_RULE_INSTRUCTION
        # from build_qa_input.py): Llama-2-chat [INST]<<SYS>> wrapper, question AFTER
        # the reasoning paths, no trailing "Answer:". Matches the fine-tuned RoG model.
        pre = ("[INST] <<SYS>>\n<</SYS>>\nBased on the reasoning paths, please answer "
               "the given question. Please keep the answer as simple as possible and "
               "return all the possible answers as a list.\n\nReasoning Paths:\n")
        suffix_text = f"\nQuestion:\n{query} [/INST]"
    full = tok.encode(pre, add_special_tokens=True)
    spans = []
    path_strs = []
    for p in paths[:max_paths]:
        s = path_to_str_rog(p)
        ids = tok.encode(s + "\n", add_special_tokens=False)
        if not ids:
            continue
        spans.append((len(full), len(full) + len(ids)))
        path_strs.append(s)
        full.extend(ids)
    full.extend(tok.encode(suffix_text, add_special_tokens=False))
    return full, spans, len(full) - 1, path_strs


def extract_answer(text):
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("###"):
            continue
        for pre in ("the answer is:", "the answer is", "answer:", "answer is"):
            if line.lower().startswith(pre):
                line = line[len(pre):].strip(" .,")
        if line:
            return line
    return re.split(r"\n|Answer:|A:|Q:", text)[0].strip()


def _carried_reuse_split(full_ids, spans, qlast, recompute_per_path, path_kvs):
    reuse, reuse_index = {}, {}
    carried = set(range(0, spans[0][0]))
    carried |= set(range(spans[-1][1], qlast + 1))
    for pi, (s, e) in enumerate(spans):
        rec = set(recompute_per_path[pi])
        for j in range(e - s):
            ap = s + j
            if j in rec:
                carried.add(ap)
            else:
                reuse[ap] = path_kvs[pi]
                reuse_index[ap] = j
    carried = sorted(carried)
    return [full_ids[p] for p in carried], carried, reuse, reuse_index


@torch.no_grad()
def faithful_generate(engine, tok, query, paths, policy="none", max_paths=8,
                      max_new_tokens=16, eos_ids=(), template="rog", path_cache=None,
                      encode_prefix=None, reuse_set=None):
    """Generate an answer for one query using faithful subgraph KV reuse.

    path_cache    : optional dict {path_str -> PathKV} giving GLOBAL cross-query
                    reuse — a path's KV is computed once and reused by every
                    query that contains it (the paper's shared KV region).
    encode_prefix : optional token-id list (lever #1). If given, each path's KV
                    is encoded with this query-independent prefix as left-context
                    (removes the position-0 attention sink and bakes in path->
                    preamble attention), staying shared across queries.
    reuse_set     : optional set of path strings allowed to be reused (the SHARED,
                    recurring paths with cross-query count r>1, as in deployed
                    GraphRAG-Opt / table2_worker). Paths NOT in this set (r=1) are
                    recomputed fully in-context (exact). When None, every path is
                    reused (the reuse-everything regime).

    Returns (answer_string, recompute_fraction)."""
    full_ids, spans, qlast, path_strs = build_prompt(tok, query, paths, max_paths, template)
    if not spans:
        # No reusable paths: plain in-context prefill over the whole prompt.
        pos = list(range(len(full_ids)))
        K, V, last_logits, N = engine.prefill_full_kv(full_ids, pos)
        first = int(last_logits.argmax().item())
        gen = engine.decode_greedy(K, V, first, N, max_new_tokens, eos_ids)
        return extract_answer(tok.decode(gen, skip_special_tokens=True)), 0.0

    path_lens = [e - s for s, e in spans]
    path_kvs = []
    for i, (s, e) in enumerate(spans):
        ps = path_strs[i]
        if path_cache is not None and ps in path_cache:
            path_kvs.append(path_cache[ps])
        else:
            pk = engine.encode_path_isolated(full_ids[s:e], prefix_ids=encode_prefix)
            if path_cache is not None:
                path_cache[ps] = pk
            path_kvs.append(pk)

    spec = dict(POLICY_SPECS[policy])
    if spec["policy"] == "cacheblend":
        ref_v = engine.encode_path_isolated(full_ids).v[CHECK_LAYER]
        spec["deviation"] = [((ref_v[s:e] - path_kvs[pi].v[CHECK_LAYER]) ** 2).sum(dim=(1, 2))
                             for pi, (s, e) in enumerate(spans)]
    sel = select_recompute_tokens(path_local_lens=path_lens, **spec)
    if reuse_set is not None:
        # selective reuse (deployed GraphRAG-Opt): only recurring shared paths
        # (r>1) are reused; one-off paths (r=1) are recomputed fully in context.
        for pi in range(len(spans)):
            if path_strs[pi] not in reuse_set:
                sel[pi] = list(range(path_lens[pi]))
    # recompute fraction reported over reusable (shared) paths only
    _den = (sum(path_lens[pi] for pi in range(len(spans)) if reuse_set is None
                or path_strs[pi] in reuse_set) if reuse_set is not None
            else sum(path_lens))
    _num = (sum(len(sel[pi]) for pi in range(len(spans))
                if reuse_set is None or path_strs[pi] in reuse_set)
            if reuse_set is not None else sum(len(s) for s in sel))
    rec_frac = _num / max(1, _den)

    carried_ids, carried, reuse, reuse_index = _carried_reuse_split(
        full_ids, spans, qlast, sel, path_kvs)
    K, V, last_logits, N = engine.prefill_full_kv(carried_ids, carried, reuse, reuse_index)
    first = int(last_logits.argmax().item())
    gen = engine.decode_greedy(K, V, first, N, max_new_tokens, eos_ids)
    return extract_answer(tok.decode(gen, skip_special_tokens=True)), rec_frac
