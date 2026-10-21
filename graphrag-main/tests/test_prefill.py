"""
test_prefill.py — Measure actual prefill time: baseline vs subgraph KV reuse.

No decode. Just prefill (forward pass to build KV cache).

  Baseline:  prefill every query's full prompt independently
  Proposed:  prefill shared paths once, reuse cached KV, prefill only
             remaining tokens per query

Usage:
  python test_prefill.py --dataset metaqa-3hop --model Qwen/Qwen3-8B \
      --test_size 256 --batch_size 64
"""

import argparse, copy, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
# tests/ lives one level below the flat root modules.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import set_reproducible_seed
from generation import format_prompt
from metaqa_adapter import load_metaqa
from retrieval import BatchOptimizedRetrieval


def _path_key(path) -> str:
    if isinstance(path, str):
        return path
    if not path:
        return ""
    parts = [str(path[0][0]).strip()]
    for _h, r, t in path:
        parts.append(str(r).strip())
        parts.append(str(t).strip())
    return " -> ".join(parts)


def clone_kv(past_kv):
    """Clone KV cache without deepcopy overhead — just clone the tensors."""
    if past_kv is None:
        return None
    if isinstance(past_kv, tuple):
        return tuple((k.clone(), v.clone()) for k, v in past_kv)
    # DynamicCache: clone layer tensors directly
    from transformers.cache_utils import DynamicCache
    new_cache = DynamicCache()
    for layer in past_kv.layers:
        new_cache.update(layer.keys.clone(), layer.values.clone(),
                         len(new_cache.layers))
    return new_cache


def kv_seq_len(past_kv) -> int:
    if past_kv is None:
        return 0
    if isinstance(past_kv, tuple):
        return past_kv[0][0].shape[2]
    return past_kv.get_seq_length()


@torch.no_grad()
def measure_baseline_prefill(model, tokenizer, prompts, device):
    """Batched prefill: pad all prompts to max length, single forward pass."""
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                       truncation=True, max_length=2048).to(device)
    total_tokens = inputs["attention_mask"].sum().item()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model(**inputs, use_cache=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed, int(total_tokens)


@torch.no_grad()
def measure_proposed_prefill(model, tokenizer, prompts, all_paths,
                             cached_keys, device):
    """
    Prefill with subgraph KV reuse — BATCHED suffix approach.

    1. Prefill shared prefix once (1 forward pass)
    2. Pad all suffixes to same length, replicate prefix KV across batch dim
    3. One batched forward pass for all suffixes (1 forward pass)
    Total: 2 forward passes vs N in baseline.
    """
    from transformers.cache_utils import DynamicCache
    cached_set = set(cached_keys)

    _HEADER = (
        "Below is an instruction that describes a task, paired with an input "
        "that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        "Please reasoning over the following knowledge graph paths, "
        "and then answer the question.\n\n"
        "### Input:\n"
        "Knowledge Graph Paths:\n"
    )

    total_tokens = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # ── Step 1: prefill shared prefix once ───────────────────────────
    cached_block = "\n".join(cached_keys)
    prefix_text = f"{_HEADER}{cached_block}\n"
    prefix_ids = tokenizer(prefix_text, return_tensors="pt",
                           add_special_tokens=True)["input_ids"].to(device)
    prefix_len = prefix_ids.shape[1]
    prefix_out = model(prefix_ids, use_cache=True)
    shared_kv = prefix_out.past_key_values
    total_tokens += prefix_len

    # ── Step 2: build and pad all suffixes ───────────────────────────
    suffix_id_list = []
    for prompt, paths in zip(prompts, all_paths):
        unique_strs = []
        for p in paths[:10]:
            pk = _path_key(p)
            if pk not in cached_set:
                unique_strs.append(pk)
        if "answer the question:" in prompt:
            question = prompt.split("answer the question:")[-1].split("\n")[0].strip()
        elif "Q:" in prompt:
            question = prompt.split("Q:")[-1].split("\n")[0].strip()
        else:
            question = ""
        unique_block = "\n".join(unique_strs) if unique_strs else ""
        suffix_text = f"{unique_block}\nQuestion: {question}\n\n### Response:"
        sids = tokenizer(suffix_text, return_tensors="pt",
                         add_special_tokens=False)["input_ids"][0]
        suffix_id_list.append(sids)

    # Left-pad suffixes to same length
    max_suf_len = max(s.shape[0] for s in suffix_id_list)
    pad_id = tokenizer.pad_token_id or 0
    batch_size = len(suffix_id_list)

    padded = torch.full((batch_size, max_suf_len), pad_id,
                        dtype=torch.long, device=device)
    attn_suffix = torch.zeros((batch_size, max_suf_len),
                              dtype=torch.long, device=device)
    for i, sids in enumerate(suffix_id_list):
        slen = sids.shape[0]
        padded[i, max_suf_len - slen:] = sids.to(device)
        attn_suffix[i, max_suf_len - slen:] = 1

    total_tokens += sum(s.shape[0] for s in suffix_id_list)

    # ── Step 3: replicate prefix KV for batch, run batched suffix ────
    batched_kv = DynamicCache()
    for layer in shared_kv.layers:
        # Expand prefix KV: (1, heads, seq, dim) → (batch, heads, seq, dim)
        batched_kv.update(
            layer.keys.expand(batch_size, -1, -1, -1).contiguous(),
            layer.values.expand(batch_size, -1, -1, -1).contiguous(),
            len(batched_kv.layers),
        )

    # Build position_ids: prefix_len + offset for each non-pad token
    position_ids = torch.zeros_like(padded)
    for i in range(batch_size):
        slen = suffix_id_list[i].shape[0]
        start = max_suf_len - slen
        position_ids[i, start:] = torch.arange(prefix_len, prefix_len + slen,
                                                device=device)

    # Full attention mask: [prefix attended] + [suffix attended/padded]
    prefix_attn = torch.ones((batch_size, prefix_len),
                             dtype=torch.long, device=device)
    full_attn_mask = torch.cat([prefix_attn, attn_suffix], dim=1)

    # One batched forward pass
    model(input_ids=padded, position_ids=position_ids,
          attention_mask=full_attn_mask,
          past_key_values=batched_kv, use_cache=False)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed, total_tokens, 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="metaqa-3hop")
    parser.add_argument("--metaqa_dir", default="metaQA")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--test_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hop_constraint", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1,
                        help="Warmup batches (not timed)")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="Single GPU to use")
    args = parser.parse_args()

    set_reproducible_seed(args.seed)

    # ── Dataset ──────────────────────────────────────────────────────
    _metaqa = {"metaqa-1hop": 1, "metaqa-2hop": 2, "metaqa-3hop": 3}
    if args.dataset in _metaqa:
        dataset = load_metaqa(hops=_metaqa[args.dataset], split="test",
                              data_dir=args.metaqa_dir)
    else:
        dataset = load_dataset(args.dataset, split="test")

    # Entity-sorted batching
    ent_to_idx = defaultdict(list)
    for i in range(len(dataset)):
        e = dataset[i].get("q_entity", [])
        if isinstance(e, list) and e:
            e = e[0].lower()
        elif isinstance(e, str):
            e = e.lower()
        else:
            e = f"_none_{i}"
        ent_to_idx[e].append(i)
    sorted_indices = []
    for ent in sorted(ent_to_idx, key=lambda x: len(ent_to_idx[x]),
                      reverse=True):
        sorted_indices.extend(ent_to_idx[ent])
    sorted_indices = sorted_indices[:args.test_size]
    dataset = dataset.select(sorted_indices)
    print(f"Dataset: {args.dataset}, {len(dataset)} samples (entity-sorted)")

    # ── Model ────────────────────────────────────────────────────────
    print(f"Loading {args.model} ...")
    is_qwen = "qwen" in args.model.lower()
    if is_qwen:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map={"": args.gpu_id})
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16,
            use_safetensors=True, device_map={"": args.gpu_id})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    device = next(model.parameters()).device
    d = model.config.hidden_size
    L = model.config.num_hidden_layers
    print(f"Model on {device}, d={d}, L={L}")

    # ── Run batches ──────────────────────────────────────────────────
    num_batches = (len(dataset) + args.batch_size - 1) // args.batch_size

    bl_times = []
    prop_times = []
    bl_tokens_total = 0
    prop_tokens_total = 0
    total_reused = 0
    total_unique = 0
    total_cached_prefixes = 0

    print(f"\n{'='*70}")
    print(f"  PREFILL BENCHMARK: {num_batches} batches x {args.batch_size}")
    print(f"{'='*70}\n")

    for batch_idx in range(num_batches):
        s = batch_idx * args.batch_size
        e = min(s + args.batch_size, len(dataset))
        batch = dataset.select(range(s, e))

        # Retrieve
        retriever = BatchOptimizedRetrieval(batch, device="cuda")
        ret_results, _ = retriever.retrieve_batch(
            args.hop_constraint, enable_batch_opt=True)
        all_paths = [r["paths"] for r in ret_results]
        prompts = [format_prompt(r["query"], r["paths"], args.model)
                   for r in ret_results]

        # Analyze reuse
        path_counter: Counter = Counter()
        for paths in all_paths:
            seen = set()
            for p in paths[:10]:
                key = _path_key(p)
                if key and key not in seen:
                    path_counter[key] += 1
                    seen.add(key)
        reused_keys = [k for k, v in path_counter.items() if v > 1]
        n_reused = len(reused_keys)
        n_unique = len(path_counter)
        total_reused += n_reused
        total_unique += n_unique

        is_warmup = batch_idx < args.warmup

        # ── Baseline prefill ─────────────────────────────────────────
        bl_t, bl_tok = measure_baseline_prefill(
            model, tokenizer, prompts, device)

        # ── Proposed prefill ─────────────────────────────────────────
        prop_t, prop_tok, n_prefixes = measure_proposed_prefill(
            model, tokenizer, prompts, all_paths, reused_keys, device)

        if not is_warmup:
            bl_times.append(bl_t)
            prop_times.append(prop_t)
            bl_tokens_total += bl_tok
            prop_tokens_total += prop_tok
            total_cached_prefixes += n_prefixes

        tag = " (warmup)" if is_warmup else ""
        print(f"  Batch {batch_idx}{tag}: {len(prompts)} queries, "
              f"{n_reused}/{n_unique} paths reused")
        print(f"    Baseline prefill:  {bl_t*1000:7.1f} ms  ({bl_tok} tokens)")
        print(f"    Proposed prefill:  {prop_t*1000:7.1f} ms  ({prop_tok} tokens, "
              f"{n_prefixes} cached prefixes)")
        if bl_t > 0:
            speedup = bl_t / max(prop_t, 1e-9)
            saved_pct = (1 - prop_t / bl_t) * 100
            print(f"    Speedup: {speedup:.2f}x  "
                  f"(saved {(bl_t - prop_t)*1000:.1f} ms, {saved_pct:.1f}%)")
        print()

    # ── Summary ──────────────────────────────────────────────────────
    total_bl = sum(bl_times)
    total_prop = sum(prop_times)

    print(f"{'='*70}")
    print(f"  PREFILL RESULTS (excluding {args.warmup} warmup batch(es))")
    print(f"{'='*70}")
    print(f"  Model:   {args.model} (d={d}, L={L})")
    print(f"  Dataset: {args.dataset}, {len(dataset)} samples")
    print(f"  Batches measured: {len(bl_times)}")
    print()
    print(f"  Baseline total prefill:  {total_bl*1000:.1f} ms  "
          f"({bl_tokens_total} tokens)")
    print(f"  Proposed total prefill:  {total_prop*1000:.1f} ms  "
          f"({prop_tokens_total} tokens)")
    print()
    if total_bl > 0 and total_prop > 0:
        speedup = total_bl / total_prop
        saved = total_bl - total_prop
        print(f"  Prefill speedup:    {speedup:.2f}x")
        print(f"  Time saved:         {saved*1000:.1f} ms "
              f"({saved/total_bl*100:.1f}%)")
    print()
    print(f"  Paths: {total_unique} unique, {total_reused} reused (r>1)")
    print(f"  Cached prefixes: {total_cached_prefixes}")
    if bl_tokens_total > 0:
        print(f"  Avg prefill cost:   "
              f"{total_bl/bl_tokens_total*1e6:.1f} µs/token (baseline)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
