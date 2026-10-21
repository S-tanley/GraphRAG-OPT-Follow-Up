#!/usr/bin/env python3
"""
exp_batch_sweep.py  —  Reviewer-B Q3 (batch-size sensitivity) + Q1a (SubGCache).

Measures GENERATION-side TTFT and throughput as a function of BATCH SIZE for
GraphRAG-Opt's cross-query subgraph-KV reuse, on a RoPE backbone, with the reuse
mechanism realized HONESTLY via the faithful per-path KV engine (hf_reuse.py):

  full     : full in-context prefill of every retrieved path  (== GraphRAG-R, the
             no-reuse baseline).
  reuse    : GraphRAG-Opt — shared paths (r(pi)>1 within the batch) are prefilled
             ONCE into a shared KV region and injected (RoPE-realigned) per query;
             unique paths are computed in context.  (== partial_reuse_generate.)
  noalign  : same as `reuse` but with positional realignment DISABLED
             (GROPT_NO_REALIGN=1) = the SubGCache-style "no positional handling"
             design contract, reimplemented in our engine (Q1a). Timing only; its
             accuracy cost is measured separately by exp_subgcache_accuracy.

Cross-query reuse is the SOURCE of the speedup, and it grows with batch size
because within-batch path redundancy grows with batch size — exactly the reviewer's
point. We therefore sweep batch sizes and report the curve honestly, under two
batch-formation policies:
  arrival : seed-42 shuffled order (= arrival-time across independent users;
            CONSERVATIVE / lower redundancy).
  topical : queries sharing a topic entity batched together (= entity locality,
            the paper's deployment assumption; UPPER bound).

TTFT  = wall-clock from prefill start to first-token logits (CUDA-event timed),
        per query, measured against a WARM shared-KV region (region built once at
        batch admission; that one-time build is counted in throughput, not TTFT).
Throughput = (all prompt+gen tokens in the batch) / (batch wall-clock incl. the
        one-time shared-region build).  decode budget = --decode_tokens.

Output: results/<out>.csv  (one row per model x dataset x policy x batch x condition).

Example:
  python exp_batch_sweep.py --model meta-llama/Llama-2-7b-hf \
     --datasets rmanluo/RoG-cwq rmanluo/RoG-webqsp \
     --batches 16 32 64 100 128 200 256 512 --policies arrival topical \
     --n_queries 1024 --decode_tokens 32 --out results/batch_sweep_llama.csv
"""
import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))

from faithful_pfkv import build_prompt, path_to_str_rog, encode_prefix_ids  # noqa: E402
from hf_reuse import HFPathCache, build_preamble_suffix                      # noqa: E402

PATHS_FILE = {
    "rmanluo/RoG-cwq": "precomputed_paths/paths_rmanluo-RoG-cwq.json",
    "rmanluo/RoG-webqsp": "precomputed_paths/paths_rmanluo-RoG-webqsp.json",
    "metaqa-3hop": "precomputed_paths/paths_metaqa-3hop.json",
}


# ───────────────────────── data loading ─────────────────────────

def load_queries(dataset, n_queries, max_paths, seed=42):
    """Return list of (question, paths) in the SAME seed-42 order the eval uses."""
    pf = json.load(open(HERE / PATHS_FILE[dataset]))
    if dataset in ("metaqa-3hop",):
        from metaqa_adapter import load_metaqa
        ds = load_metaqa(hops=3, split="test", data_dir=str(HERE / "metaQA")).shuffle(seed=seed)
    else:
        ds = load_dataset(dataset, split="test").shuffle(seed=seed)
    out = []
    for i in range(min(n_queries, len(ds))):
        q = ds[i]["question"]
        paths = pf.get(q, [])
        paths = [[tuple(t) for t in p] for p in paths][:max_paths]
        out.append((q, paths))
    return out


def topic_entity(paths):
    c = defaultdict(int)
    for p in paths:
        for h, r, t in p:
            c[h] += 1
            c[t] += 1
    return max(c, key=c.get) if c else ""


def order_queries(items, policy, seed=42):
    if policy == "arrival":
        return items                       # already seed-42 shuffled
    if policy == "topical":
        return sorted(items, key=lambda it: topic_entity(it[1]))
    raise ValueError(policy)


# ───────────────────────── timed prefill primitives ─────────────────────────

@torch.no_grad()
def _decode_for_throughput(model, cache, logits, start_pos, n, eos, device):
    cur, gen = start_pos, 0
    for _ in range(n):
        tid = int(logits.argmax())
        if eos is not None and tid == eos:
            break
        gen += 1
        am = torch.ones(1, cur + 1, device=device, dtype=torch.long)
        out = model(input_ids=torch.tensor([[tid]], device=device), past_key_values=cache,
                    position_ids=torch.tensor([[cur]], device=device),
                    attention_mask=am, use_cache=True)
        cache, logits = out.past_key_values, out.logits[0, -1]
        cur += 1
    return gen


@torch.no_grad()
def time_full(model, tok, query, paths, max_paths, decode_tokens, device, template):
    """GraphRAG-R baseline: full in-context prefill of all path text."""
    full_ids, _, _, _ = build_prompt(tok, query, paths, max_paths, template)
    eos = tok.eos_token_id
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(input_ids=torch.tensor([full_ids], device=device), use_cache=True)
    cache, logits = out.past_key_values, out.logits[0, -1]
    torch.cuda.synchronize()
    ttft = time.perf_counter() - t0
    gen = _decode_for_throughput(model, cache, logits, len(full_ids), decode_tokens, eos, device)
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    return ttft, total, len(full_ids), gen


@torch.no_grad()
def time_reuse(model, tok, pcache, query, reused_paths, computed_paths,
               decode_tokens, device, template):
    """GraphRAG-Opt: shared paths injected from the (warm) shared KV region,
    unique paths computed in context. pcache must be PRE-WARMED for reused_paths
    so .get() is a cache hit (the shared region is built once per batch)."""
    pre_ids, suf_ids = build_preamble_suffix(tok, query, template)
    eos = tok.eos_token_id
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(input_ids=torch.tensor([pre_ids], device=device), use_cache=True)
    cache = out.past_key_values
    cur = len(pre_ids)
    for p in reused_paths:
        k_list, v_list, L, ptoks = pcache.get(p)
        if L == 0:
            continue
        delta = cur - pcache.P_enc
        from hf_reuse import _layer_kv_device
        for li in range(len(k_list)):
            dev_li = _layer_kv_device(cache, li, device)
            k = pcache._realign_k(k_list[li].to(dev_li), delta)
            cache.update(k.to(model.dtype), v_list[li].to(dev_li, model.dtype), li)
        cur += L
    tail = "".join(path_to_str_rog(p) + "\n" for p in computed_paths)
    tail_ids = tok.encode(tail, add_special_tokens=False) + suf_ids
    pos = torch.arange(cur, cur + len(tail_ids), device=device).unsqueeze(0)
    am = torch.ones(1, cur + len(tail_ids), device=device, dtype=torch.long)
    out = model(input_ids=torch.tensor([tail_ids], device=device), past_key_values=cache,
                position_ids=pos, attention_mask=am, use_cache=True)
    cache, logits = out.past_key_values, out.logits[0, -1]
    torch.cuda.synchronize()
    ttft = time.perf_counter() - t0
    promptlen = cur + len(tail_ids)
    gen = _decode_for_throughput(model, cache, logits, promptlen, decode_tokens, eos, device)
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    return ttft, total, promptlen, gen


@torch.no_grad()
def warm_shared_region(pcache, shared_paths, device):
    """Build the shared KV region once: encode each shared path's KV. Returns the
    wall-clock build time (counted in throughput)."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for p in shared_paths:
        pcache.get(p)          # encodes on miss -> populates cache
    torch.cuda.synchronize()
    return time.perf_counter() - t0


# ───────────────────────── batch evaluation ─────────────────────────

def shared_paths_in_batch(batch):
    """paths appearing in >1 query of the batch (cross-query reuse units)."""
    seen = defaultdict(int)
    str2path = {}
    for _q, paths in batch:
        for p in set(path_to_str_rog(pp) for pp in paths):
            seen[p] += 1
        for pp in paths:
            str2path[path_to_str_rog(pp)] = pp
    shared = {s for s, c in seen.items() if c > 1}
    return shared, str2path


def run_condition(model, tok, items, B, condition, decode_tokens, device, template,
                  max_paths, enc_prefix):
    """Run all non-overlapping size-B batches; return aggregate metrics."""
    ttfts, prompt_toks, gen_toks = [], 0, 0
    batch_walls = []
    shared_fracs = []
    nbatch = 0
    for i in range(0, len(items) - B + 1, B):
        batch = items[i:i + B]
        nbatch += 1
        if condition == "full":
            torch.cuda.synchronize()
            t_b0 = time.perf_counter()
            for q, paths in batch:
                ttft, total, pl, g = time_full(model, tok, q, paths, max_paths,
                                                decode_tokens, device, template)
                ttfts.append(ttft); prompt_toks += pl; gen_toks += g
            torch.cuda.synchronize()
            batch_walls.append(time.perf_counter() - t_b0)
        else:
            # reuse / noalign : build shared region once, then per-query reuse
            shared, str2path = shared_paths_in_batch(batch)
            shared_fracs.append(
                (sum(1 for _q, ps in batch for pp in ps
                     if path_to_str_rog(pp) in shared)) /
                max(1, sum(len(ps) for _q, ps in batch)))
            pcache = HFPathCache(model, tok, device, encode_prefix_ids=enc_prefix)
            os.environ["GROPT_NO_REALIGN"] = "1" if condition == "noalign" else "0"
            torch.cuda.synchronize()
            t_b0 = time.perf_counter()
            setup = warm_shared_region(pcache, [str2path[s] for s in shared], device)
            for q, paths in batch:
                reused = [pp for pp in paths if path_to_str_rog(pp) in shared]
                computed = [pp for pp in paths if path_to_str_rog(pp) not in shared]
                ttft, total, pl, g = time_reuse(model, tok, pcache, q, reused, computed,
                                                decode_tokens, device, template)
                ttfts.append(ttft); prompt_toks += pl; gen_toks += g
            torch.cuda.synchronize()
            batch_walls.append(time.perf_counter() - t_b0)
            del pcache
            torch.cuda.empty_cache()
    os.environ.pop("GROPT_NO_REALIGN", None)
    import numpy as np
    total_wall = sum(batch_walls)
    return {
        "n_batches": nbatch,
        "n_queries": len(ttfts),
        "mean_ttft_ms": float(np.mean(ttfts)) * 1000 if ttfts else float("nan"),
        "median_ttft_ms": float(np.median(ttfts)) * 1000 if ttfts else float("nan"),
        "p95_ttft_ms": float(np.percentile(ttfts, 95)) * 1000 if ttfts else float("nan"),
        "throughput_tok_s": (prompt_toks + gen_toks) / total_wall if total_wall > 0 else 0.0,
        "shared_path_frac": float(np.mean(shared_fracs)) if shared_fracs else 0.0,
    }


def main():
    import csv
    import numpy as np  # noqa
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--datasets", nargs="+", default=["rmanluo/RoG-cwq", "rmanluo/RoG-webqsp"])
    ap.add_argument("--batches", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512])
    ap.add_argument("--policies", nargs="+", default=["arrival", "topical"])
    ap.add_argument("--conditions", nargs="+", default=["full", "reuse", "noalign"])
    ap.add_argument("--n_queries", type=int, default=1024)
    ap.add_argument("--max_paths", type=int, default=8)
    ap.add_argument("--decode_tokens", type=int, default=32)
    ap.add_argument("--out", default="results/batch_sweep.csv")
    a = ap.parse_args()

    device = "cuda"
    template = "rog" if "rog" in a.model.lower() else "fewshot"
    tok_src = "meta-llama/Llama-2-7b-hf" if "rog" in a.model.lower() else a.model
    print(f"Loading {a.model} (tok={tok_src}, template={template}) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16).to(device).eval()
    enc_prefix = encode_prefix_ids(tok, template)

    rows = []
    for ds in a.datasets:
        items_all = load_queries(ds, a.n_queries, a.max_paths)
        print(f"\n[{ds}] {len(items_all)} queries loaded", flush=True)
        for policy in a.policies:
            items = order_queries(items_all, policy)
            for B in a.batches:
                if B > len(items):
                    continue
                for cond in a.conditions:
                    m = run_condition(model, tok, items, B, cond, a.decode_tokens,
                                      device, template, a.max_paths, enc_prefix)
                    row = {"model": a.model, "dataset": ds, "policy": policy,
                           "batch": B, "condition": cond, **m}
                    rows.append(row)
                    print(f"  {ds.split('/')[-1]:12} {policy:7} B={B:<4} {cond:8} "
                          f"TTFT={m['mean_ttft_ms']:8.2f}ms (med {m['median_ttft_ms']:7.2f}) "
                          f"tput={m['throughput_tok_s']:8.1f}tok/s "
                          f"shared={m['shared_path_frac']*100:4.1f}%", flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}")

    # quick speedup summary (reuse vs full) per dataset/policy/batch
    print(f"\n{'='*78}\nTTFT & THROUGHPUT SPEEDUP (reuse vs full)\n{'='*78}")
    idx = {(r["dataset"], r["policy"], r["batch"], r["condition"]): r for r in rows}
    for ds in a.datasets:
        for policy in a.policies:
            print(f"\n  {ds} / {policy}:")
            for B in a.batches:
                f_ = idx.get((ds, policy, B, "full"))
                r_ = idx.get((ds, policy, B, "reuse"))
                if f_ and r_:
                    ts = f_["mean_ttft_ms"] / r_["mean_ttft_ms"] if r_["mean_ttft_ms"] else float("nan")
                    qs = r_["throughput_tok_s"] / f_["throughput_tok_s"] if f_["throughput_tok_s"] else float("nan")
                    print(f"    B={B:<4} TTFT speedup={ts:4.2f}x  throughput speedup={qs:4.2f}x  "
                          f"(shared {r_['shared_path_frac']*100:4.1f}%)")


if __name__ == "__main__":
    main()
