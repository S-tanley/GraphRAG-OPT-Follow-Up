"""
experiment_scheduling.py — Path-Unit Scheduling Policy for GraphRAG
===================================================================
Profiles prefill and decode separately, runs the path-unit scheduler,
and reports measured time savings with full breakdowns.

  1. Profiles real per-token prefill cost on this GPU/model
  2. Runs vanilla GraphRAG with separate prefill/decode timing
  3. Runs the scheduler to partition paths into GPU / CPU / recompute
  4. Projects savings using measured (not theoretical) prefill cost

Usage
-----
  python experiment_scheduling.py --dataset metaqa-3hop --model Qwen/Qwen3-8B \\
      --test_size 256 --batch_size 64 --memory_budget_mb 512
"""

import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import QUICK_MODE_CONFIGS, set_reproducible_seed
from flops import flops_prefill, get_arch
from generation import format_prompt
from graph_utils import path_to_str
from metrics import evaluate_metrics
from metaqa_adapter import load_metaqa
from retrieval import BatchOptimizedRetrieval


# ═══════════════════════════════════════════════════════════════════════
# Profiler: measure real per-token prefill cost
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def profile_prefill_cost(model, tokenizer, device, n_warmup=3, n_measure=10):
    """
    Measure wall-clock prefill cost per token on this GPU/model.
    Runs forward passes at several sequence lengths and returns the
    average cost in seconds per token.
    """
    print("[Profile] Measuring per-token prefill cost ...")
    lengths = [32, 64, 128, 256, 512]
    dummy_text = "The quick brown fox jumps over the lazy dog. " * 100
    dummy_ids = tokenizer(dummy_text, return_tensors="pt",
                          truncation=True, max_length=max(lengths))["input_ids"].to(device)

    costs = []
    for seq_len in lengths:
        ids = dummy_ids[:, :seq_len]
        # Warmup
        for _ in range(n_warmup):
            model(ids, use_cache=False)
        torch.cuda.synchronize()
        # Measure
        times = []
        for _ in range(n_measure):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(ids, use_cache=False)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        avg = np.median(times)
        per_tok = avg / seq_len
        costs.append(per_tok)
        print(f"  seq_len={seq_len:4d}: {avg*1000:.2f} ms total, "
              f"{per_tok*1e6:.1f} µs/token")

    avg_cost = np.mean(costs)
    print(f"[Profile] Average prefill cost: {avg_cost*1e6:.1f} µs/token")
    return avg_cost


# ═══════════════════════════════════════════════════════════════════════
# Path-Unit Scheduling Policy
# ═══════════════════════════════════════════════════════════════════════

class PathScheduler:
    """
    Fractional-knapsack scheduler operating on path units (π, Φ(π)).

    Tiers:
      GPU-resident  – KV stays on GPU, zero reload cost
      CPU-offloaded – KV in pinned CPU memory, reload via PCIe
      Recompute     – KV discarded, recomputed on demand
    """

    def __init__(self, gpu_budget_bytes: int, cpu_budget_bytes: int,
                 d_model: int, n_layers: int,
                 measured_prefill_cost_per_token: float,
                 bandwidth_bytes_per_s: float = 12e9,
                 dtype_bytes: int = 2):
        self.gpu_budget = gpu_budget_bytes
        self.cpu_budget = cpu_budget_bytes
        self.d = d_model
        self.L = n_layers
        self.cost_per_tok = measured_prefill_cost_per_token
        self.B = bandwidth_bytes_per_s
        self.dtype = dtype_bytes

    def _kv_mem(self, n_tokens: int) -> int:
        """Memory for K+V across all layers for n_tokens."""
        return 2 * self.L * n_tokens * self.d * self.dtype

    def _prefill_time(self, n_tokens: int) -> float:
        """Measured prefill wall-clock time for n_tokens."""
        return n_tokens * self.cost_per_tok

    def _reload_time(self, n_tokens: int) -> float:
        """Time to reload KV from CPU via PCIe."""
        return self._kv_mem(n_tokens) / self.B

    def schedule(self, reuse_counts: Dict[str, int],
                 token_lengths: Dict[str, int]) -> Dict[str, dict]:
        """Partition paths into tiers."""
        candidates = []
        for key, r in reuse_counts.items():
            tlen = token_lengths[key]
            mem = self._kv_mem(tlen)
            # Benefit: time saved = (r-1) prefills avoided
            time_saved = (r - 1) * self._prefill_time(tlen) if r > 1 else 0.0
            priority = time_saved / max(mem, 1)
            candidates.append({
                "key": key, "r": r, "tokens": tlen,
                "mem": mem, "time_saved": time_saved, "priority": priority,
            })

        candidates.sort(key=lambda x: x["priority"], reverse=True)

        result = {}
        gpu_remaining = self.gpu_budget
        cpu_remaining = self.cpu_budget

        for c in candidates:
            key = c["key"]
            if c["r"] <= 1:
                result[key] = {**c, "tier": "recompute"}
                continue
            if c["mem"] <= gpu_remaining:
                result[key] = {**c, "tier": "gpu"}
                gpu_remaining -= c["mem"]
            elif c["mem"] <= cpu_remaining:
                # Offload only if reload is cheaper than recompute
                if self._reload_time(c["tokens"]) < self._prefill_time(c["tokens"]):
                    result[key] = {**c, "tier": "cpu"}
                    cpu_remaining -= c["mem"]
                else:
                    result[key] = {**c, "tier": "recompute"}
            else:
                result[key] = {**c, "tier": "recompute"}

        return result


# ═══════════════════════════════════════════════════════════════════════
# Path reuse analysis
# ═══════════════════════════════════════════════════════════════════════

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


def analyze_batch_reuse(all_query_paths: List[List], tokenizer) -> Tuple[dict, dict]:
    path_counter: Counter = Counter()
    for paths in all_query_paths:
        seen = set()
        for p in paths[:10]:
            key = _path_key(p)
            if key and key not in seen:
                path_counter[key] += 1
                seen.add(key)
    token_lengths = {}
    for key in path_counter:
        toks = tokenizer(key, add_special_tokens=False)["input_ids"]
        token_lengths[key] = len(toks)
    return dict(path_counter), token_lengths


# ═══════════════════════════════════════════════════════════════════════
# Per-query prefill/decode timing
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_with_timing(model, tokenizer, prompts: List[str], device,
                         max_new_tokens: int = 50) -> Tuple[List[str], dict]:
    """
    Generate one query at a time, measuring prefill and decode separately.
    Returns (predictions, timing_dict).
    """
    predictions = []
    prefill_times = []
    decode_times = []
    total_prompt_tokens = 0
    total_gen_tokens = 0

    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = ids["input_ids"]
        prompt_len = input_ids.shape[1]
        total_prompt_tokens += prompt_len

        # ── Prefill: forward pass to build KV cache ──────────────────
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(input_ids, use_cache=True)
        torch.cuda.synchronize()
        t_prefill_end = time.perf_counter()
        prefill_times.append(t_prefill_end - t0)

        # ── Decode: autoregressive token generation ──────────────────
        past_kv = out.past_key_values
        next_logits = out.logits[:, -1, :]
        generated_ids = []

        t_decode_start = time.perf_counter()
        for _ in range(max_new_tokens):
            next_id = next_logits.argmax(dim=-1, keepdim=True)
            if next_id.item() == tokenizer.eos_token_id:
                break
            generated_ids.append(next_id)
            step_out = model(input_ids=next_id, past_key_values=past_kv,
                             use_cache=True)
            next_logits = step_out.logits[:, -1, :]
            past_kv = step_out.past_key_values
        torch.cuda.synchronize()
        decode_times.append(time.perf_counter() - t_decode_start)

        n_gen = len(generated_ids)
        total_gen_tokens += n_gen
        if generated_ids:
            text = tokenizer.decode(
                torch.cat(generated_ids, dim=-1)[0],
                skip_special_tokens=True).strip()
        else:
            text = ""
        predictions.append(text)

    stats = {
        "prefill_total_s": sum(prefill_times),
        "decode_total_s": sum(decode_times),
        "prefill_times": prefill_times,
        "decode_times": decode_times,
        "prompt_tokens": total_prompt_tokens,
        "gen_tokens": total_gen_tokens,
        "n_queries": len(prompts),
    }
    return predictions, stats


# ═══════════════════════════════════════════════════════════════════════
# Experiment runner
# ═══════════════════════════════════════════════════════════════════════

def run_experiment(args):
    set_reproducible_seed(args.seed)

    # ── Load dataset ─────────────────────────────────────────────────
    print(f"Dataset: {args.dataset}")
    _metaqa = {"metaqa-1hop": 1, "metaqa-2hop": 2, "metaqa-3hop": 3}
    if args.dataset in _metaqa:
        dataset = load_metaqa(hops=_metaqa[args.dataset], split="test",
                              data_dir=args.metaqa_dir)
    else:
        dataset = load_dataset(args.dataset, split="test")

    # Entity-sorted batching (Zipfian workload simulation)
    try:
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
        print(f"Entity-sorted: top-frequency entities placed in consecutive batches")
    except Exception:
        if len(dataset) > args.test_size:
            dataset = dataset.shuffle(seed=args.seed).select(range(args.test_size))
    print(f"Samples: {len(dataset)}")

    # ── Load model ───────────────────────────────────────────────────
    print(f"Loading model: {args.model}")
    is_qwen = "qwen" in args.model.lower()
    is_gemma = "gemma" in args.model.lower()
    if is_qwen or is_gemma:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16,
            use_safetensors=True, device_map="auto")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    device = next(model.parameters()).device
    cfg = model.config
    d_model = cfg.hidden_size
    n_layers = cfg.num_hidden_layers
    dtype_bytes = 2
    print(f"Model on {device}, d={d_model}, L={n_layers}")

    # ── Profile prefill cost ─────────────────────────────────────────
    cost_per_tok = profile_prefill_cost(model, tokenizer, device)

    # ── Scheduler ────────────────────────────────────────────────────
    gpu_budget = args.memory_budget_mb * 1024 * 1024
    cpu_budget = args.cpu_budget_mb * 1024 * 1024
    scheduler = PathScheduler(
        gpu_budget, cpu_budget, d_model, n_layers,
        measured_prefill_cost_per_token=cost_per_tok,
        bandwidth_bytes_per_s=args.bandwidth * 1e9,
        dtype_bytes=dtype_bytes,
    )

    # ── Accumulators ─────────────────────────────────────────────────
    all_preds = []
    all_answers = []
    batch_stats = []         # per-batch timing
    batch_schedules = []     # per-batch scheduler output

    num_batches = (len(dataset) + args.batch_size - 1) // args.batch_size
    print(f"\nBatches: {num_batches} x {args.batch_size}")
    print("=" * 80)

    for batch_idx in tqdm(range(num_batches), desc="Batches"):
        s = batch_idx * args.batch_size
        e = min(s + args.batch_size, len(dataset))
        batch = dataset.select(range(s, e))
        all_answers.extend(batch["answer"])

        # ── Retrieval ────────────────────────────────────────────────
        torch.cuda.synchronize()
        t_ret_start = time.perf_counter()
        retriever = BatchOptimizedRetrieval(batch, device="cuda")
        ret_results, _ = retriever.retrieve_batch(
            args.hop_constraint, enable_batch_opt=True)
        torch.cuda.synchronize()
        retrieval_time = time.perf_counter() - t_ret_start

        all_paths = [r["paths"] for r in ret_results]
        prompts = [format_prompt(r["query"], r["paths"], args.model)
                   for r in ret_results]

        # ── Generation with separate prefill/decode timing ───────────
        preds, gen_stats = generate_with_timing(
            model, tokenizer, prompts, device, args.max_new_tokens)
        all_preds.extend(preds)

        # ── Scheduling analysis ──────────────────────────────────────
        reuse_counts, token_lengths = analyze_batch_reuse(all_paths, tokenizer)
        schedule = scheduler.schedule(reuse_counts, token_lengths)

        # Compute per-batch scheduler stats
        n_gpu = sum(1 for v in schedule.values() if v["tier"] == "gpu")
        n_cpu = sum(1 for v in schedule.values() if v["tier"] == "cpu")
        n_recomp = sum(1 for v in schedule.values() if v["tier"] == "recompute")
        gpu_mem = sum(v["mem"] for v in schedule.values() if v["tier"] == "gpu")
        cpu_mem = sum(v["mem"] for v in schedule.values() if v["tier"] == "cpu")
        saved_time = sum(v["time_saved"] for v in schedule.values()
                         if v["tier"] in ("gpu", "cpu"))
        # CPU paths pay reload cost
        reload_cost = sum(scheduler._reload_time(v["tokens"])
                          for v in schedule.values() if v["tier"] == "cpu")

        reused = sum(1 for r in reuse_counts.values() if r > 1)
        total_reuse_count = sum(r - 1 for r in reuse_counts.values() if r > 1)
        saved_tokens = sum(
            (v["r"] - 1) * v["tokens"] for v in schedule.values()
            if v["tier"] in ("gpu", "cpu") and v["r"] > 1)

        sched_stats = {
            "n_gpu": n_gpu, "n_cpu": n_cpu, "n_recomp": n_recomp,
            "gpu_mem_mb": gpu_mem / (1024**2), "cpu_mem_mb": cpu_mem / (1024**2),
            "saved_time_s": saved_time, "reload_cost_s": reload_cost,
            "net_saving_s": saved_time - reload_cost,
            "reused_paths": reused,
            "total_reuse_count": total_reuse_count,
            "saved_tokens": saved_tokens,
        }
        batch_schedules.append(sched_stats)

        bstat = {
            "retrieval_s": retrieval_time,
            "prefill_s": gen_stats["prefill_total_s"],
            "decode_s": gen_stats["decode_total_s"],
            "prompt_tokens": gen_stats["prompt_tokens"],
            "gen_tokens": gen_stats["gen_tokens"],
            "n_queries": gen_stats["n_queries"],
        }
        batch_stats.append(bstat)

        if batch_idx < 3 or batch_idx == num_batches - 1:
            print(f"\n  Batch {batch_idx}: {bstat['n_queries']} queries")
            print(f"    Retrieval: {bstat['retrieval_s']*1000:.0f}ms | "
                  f"Prefill: {bstat['prefill_s']*1000:.0f}ms | "
                  f"Decode: {bstat['decode_s']*1000:.0f}ms")
            print(f"    Tokens: {bstat['prompt_tokens']} prompt, "
                  f"{bstat['gen_tokens']} generated")
            print(f"    Reuse: {reused}/{len(reuse_counts)} paths reused "
                  f"({total_reuse_count} redundant prefills)")
            print(f"    Schedule: GPU={n_gpu} ({gpu_mem/(1024**2):.0f}MB) "
                  f"CPU={n_cpu} Recomp={n_recomp}")
            print(f"    Saved prefill: {sched_stats['saved_tokens']} tokens "
                  f"= {saved_time*1000:.1f}ms")

    # ══════════════════════════════════════════════════════════════════
    # FINAL REPORT
    # ══════════════════════════════════════════════════════════════════
    metrics = evaluate_metrics(all_preds, all_answers)

    total_retrieval = sum(b["retrieval_s"] for b in batch_stats)
    total_prefill = sum(b["prefill_s"] for b in batch_stats)
    total_decode = sum(b["decode_s"] for b in batch_stats)
    total_e2e = total_retrieval + total_prefill + total_decode
    total_prompt_toks = sum(b["prompt_tokens"] for b in batch_stats)
    total_gen_toks = sum(b["gen_tokens"] for b in batch_stats)

    total_saved = sum(s["net_saving_s"] for s in batch_schedules)
    total_saved_tokens = sum(s["saved_tokens"] for s in batch_schedules)
    total_gpu_paths = sum(s["n_gpu"] for s in batch_schedules)
    total_cpu_paths = sum(s["n_cpu"] for s in batch_schedules)
    total_recomp_paths = sum(s["n_recomp"] for s in batch_schedules)
    total_gpu_mem = sum(s["gpu_mem_mb"] for s in batch_schedules)
    total_reused = sum(s["reused_paths"] for s in batch_schedules)
    total_reuse_events = sum(s["total_reuse_count"] for s in batch_schedules)

    projected_prefill = total_prefill - total_saved
    projected_e2e = total_retrieval + projected_prefill + total_decode

    print("\n" + "=" * 80)
    print("  FINAL RESULTS")
    print("=" * 80)
    print(f"  Dataset: {args.dataset}")
    print(f"  Model:   {args.model} (d={d_model}, L={n_layers})")
    print(f"  Samples: {len(dataset)} | Batch: {args.batch_size} | "
          f"Hop: {args.hop_constraint}")
    print(f"  GPU budget: {args.memory_budget_mb}MB | "
          f"CPU budget: {args.cpu_budget_mb}MB | "
          f"PCIe BW: {args.bandwidth} GB/s")

    print("\n" + "-" * 80)
    print("  ACCURACY")
    print("-" * 80)
    print(f"  Hits@1: {metrics['hits_at_1']:.4f}   "
          f"F1: {metrics['f1']:.4f}   "
          f"Acc: {metrics['accuracy']:.4f}")

    print("\n" + "-" * 80)
    print("  BASELINE TIMING (measured)")
    print("-" * 80)
    print(f"  Retrieval:   {total_retrieval:8.2f}s  "
          f"({total_retrieval/total_e2e*100:5.1f}%)")
    print(f"  Prefill:     {total_prefill:8.2f}s  "
          f"({total_prefill/total_e2e*100:5.1f}%)")
    print(f"  Decode:      {total_decode:8.2f}s  "
          f"({total_decode/total_e2e*100:5.1f}%)")
    print(f"  End-to-end:  {total_e2e:8.2f}s")
    print(f"  Tokens:      {total_prompt_toks} prompt + {total_gen_toks} gen")
    print(f"  Profiled prefill cost: {cost_per_tok*1e6:.1f} µs/token")

    print("\n" + "-" * 80)
    print("  SCHEDULER DECISIONS (across all batches)")
    print("-" * 80)
    total_unique = total_gpu_paths + total_cpu_paths + total_recomp_paths
    print(f"  Unique paths:     {total_unique}")
    print(f"  Reused (r>1):     {total_reused} "
          f"({total_reuse_events} redundant prefill events)")
    print(f"  GPU-resident:     {total_gpu_paths:4d} paths  "
          f"({total_gpu_mem:.0f} MB)")
    print(f"  CPU-offloaded:    {total_cpu_paths:4d} paths")
    print(f"  Recompute:        {total_recomp_paths:4d} paths")
    print(f"  Saved tokens:     {total_saved_tokens} "
          f"(would-be-redundant prefill tokens eliminated)")

    print("\n" + "-" * 80)
    print("  PROJECTED TIMING (with scheduler)")
    print("-" * 80)
    print(f"  Retrieval:   {total_retrieval:8.2f}s  (unchanged)")
    print(f"  Prefill:     {projected_prefill:8.2f}s  "
          f"(was {total_prefill:.2f}s, "
          f"saved {total_saved*1000:.1f}ms)")
    print(f"  Decode:      {total_decode:8.2f}s  (unchanged)")
    print(f"  End-to-end:  {projected_e2e:8.2f}s")

    print("\n" + "-" * 80)
    print("  SPEEDUP SUMMARY")
    print("-" * 80)
    if total_prefill > 0:
        prefill_speedup = total_prefill / max(projected_prefill, 1e-6)
        print(f"  Prefill speedup:   {prefill_speedup:.2f}x  "
              f"({total_saved*1000:.1f}ms saved)")
    if total_e2e > 0:
        e2e_speedup = total_e2e / max(projected_e2e, 1e-6)
        print(f"  End-to-end speedup: {e2e_speedup:.3f}x")
    print(f"  Prefill reduction: {total_saved/max(total_prefill,1e-6)*100:.1f}%")

    # Per-token economics
    print("\n" + "-" * 80)
    print("  PER-TOKEN ECONOMICS")
    print("-" * 80)
    kv_per_tok = 2 * n_layers * d_model * dtype_bytes
    print(f"  KV memory per token:      {kv_per_tok/1024:.1f} KB  "
          f"(2 × {n_layers} layers × {d_model} dim × {dtype_bytes}B)")
    print(f"  Prefill cost per token:   {cost_per_tok*1e6:.1f} µs  (measured)")
    if args.bandwidth > 0:
        reload_per_tok = kv_per_tok / (args.bandwidth * 1e9)
        print(f"  PCIe reload per token:    {reload_per_tok*1e6:.1f} µs  "
              f"(at {args.bandwidth} GB/s)")
        print(f"  Recompute/reload ratio:   {cost_per_tok/reload_per_tok:.1f}x  "
              f"({'offload wins' if reload_per_tok < cost_per_tok else 'recompute wins'})")

    # Theoretical FLOPs
    print("\n" + "-" * 80)
    print("  THEORETICAL FLOP ANALYSIS")
    print("-" * 80)
    arch = get_arch(model)
    print(f"  Architecture: d={arch['d']}, L={arch['nl']}, nh={arch['nh']}, "
          f"nkv={arch['nkv']}, fi={arch['fi']}, V={arch['V']}")

    # Avg prompt length
    avg_prompt_len = total_prompt_toks // max(len(dataset), 1)
    avg_saved_per_query = total_saved_tokens // max(len(dataset), 1)

    # Baseline: prefill FLOPs for avg prompt
    bl_pf = flops_prefill(avg_prompt_len, arch)
    # Proposed: prefill FLOPs for reduced prompt (avg - saved tokens)
    proposed_prompt_len = max(avg_prompt_len - avg_saved_per_query, 1)
    prop_pf = flops_prefill(proposed_prompt_len, arch)

    # Total across all queries
    n_queries = len(dataset)
    bl_total_flops = bl_pf["total"] * n_queries
    prop_total_flops = prop_pf["total"] * n_queries
    saved_flops = bl_total_flops - prop_total_flops

    print(f"\n  Per-query (avg {avg_prompt_len} tokens):")
    print(f"    Baseline prefill:     {bl_pf['total']/1e9:>10.2f} GFLOP")
    print(f"      Attention:          {sum(bl_pf[k] for k in ['q_proj','k_proj','v_proj','attn_qk','attn_softmax','attn_sv','o_proj'])/1e9:>10.2f} GFLOP")
    print(f"      FFN (SwiGLU):       {sum(bl_pf[k] for k in ['ffn_gate','ffn_up','ffn_down','ffn_act'])/1e9:>10.2f} GFLOP")

    print(f"\n  Per-query after scheduling (avg {proposed_prompt_len} tokens, "
          f"−{avg_saved_per_query} saved):")
    print(f"    Proposed prefill:     {prop_pf['total']/1e9:>10.2f} GFLOP")
    print(f"    FLOP reduction:       {(1 - prop_pf['total']/bl_pf['total'])*100:.1f}%")

    print(f"\n  Across all {n_queries} queries:")
    print(f"    Baseline total:       {bl_total_flops/1e12:>10.4f} TFLOP")
    print(f"    Proposed total:       {prop_total_flops/1e12:>10.4f} TFLOP")
    print(f"    Saved:                {saved_flops/1e12:>10.4f} TFLOP "
          f"({saved_flops/bl_total_flops*100:.1f}%)")
    print(f"    FLOP speedup:         {bl_total_flops/prop_total_flops:.2f}x")

    # KV cache memory analysis
    print(f"\n  KV Cache Memory:")
    kv_per_token_bytes = 2 * arch['nl'] * arch['nkv'] * arch['dh'] * dtype_bytes
    bl_kv_total = avg_prompt_len * kv_per_token_bytes * n_queries
    prop_kv_total = proposed_prompt_len * kv_per_token_bytes * n_queries
    print(f"    Per token (GQA-aware): {kv_per_token_bytes/1024:.1f} KB  "
          f"(2 × {arch['nl']}L × {arch['nkv']}kv_heads × {arch['dh']}dh × {dtype_bytes}B)")
    print(f"    Baseline total KV:    {bl_kv_total/(1024**2):.0f} MB  "
          f"(across {n_queries} queries)")
    print(f"    Proposed total KV:    {prop_kv_total/(1024**2):.0f} MB  "
          f"(saved {(bl_kv_total-prop_kv_total)/(1024**2):.0f} MB)")

    print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Path-Unit Scheduling Policy Experiment")
    parser.add_argument("--dataset", default="rmanluo/RoG-cwq",
                        help="HF dataset or metaqa-1hop|metaqa-2hop|metaqa-3hop")
    parser.add_argument("--metaqa_dir", default="metaQA")
    parser.add_argument("--model", default="rmanluo/RoG")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_size", type=int, default=128)
    parser.add_argument("--hop_constraint", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=50)

    parser.add_argument("--memory_budget_mb", type=int, default=512)
    parser.add_argument("--cpu_budget_mb", type=int, default=2048)
    parser.add_argument("--bandwidth", type=float, default=12.0,
                        help="PCIe bandwidth GB/s")

    parser.add_argument("--fast_mode",
                        choices=list(QUICK_MODE_CONFIGS.keys()), default=None)

    args = parser.parse_args()
    if args.fast_mode:
        cfg = QUICK_MODE_CONFIGS[args.fast_mode]
        args.test_size = cfg["samples"]
        args.batch_size = cfg["batch_size"]
        args.max_new_tokens = cfg.get("max_new_tokens", args.max_new_tokens)

    print(f"Config: {vars(args)}")
    run_experiment(args)


if __name__ == "__main__":
    main()
