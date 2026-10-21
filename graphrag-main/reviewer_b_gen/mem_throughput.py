#!/usr/bin/env python3
"""
mem_throughput.py — Reviewer-B: KV-memory & generation-throughput vs batch size
(isolated). Reproduces the paper's memory-driven throughput-scaling story
(Fig:throughput_line_plot): cross-query subgraph caching stores the shared
subgraph ONCE and serves it to all queries, cutting per-batch KV memory so larger
batches fit and peak throughput rises.

Mechanism (faithful, truly batched via PREFIX BROADCAST):
  topical batch of B queries about related entities -> their SHARED subgraph
  (union of paths with r(pi)>1) is a common prefix. We prefill it ONCE -> past KV
  [1, L_shared]; broadcast to all B; each query then prefills only its UNIQUE
  remainder + question and decodes. Baseline replicates the full prompt per query.

We measure, per (model, B):
  * baseline: real batched generate over [preamble+ALL paths+q], use_cache=True
  * reuse:    shared prefix prefilled once, broadcast to B; batched generate over
              [UNIQUE paths+q] with that common past
  throughput = (prompt+gen tokens)/wall ; peak_mem = torch.cuda.max_memory_allocated
  Sweep B until OOM. Both use the SAME standard decode KV cache (apples-to-apples;
  the only difference is shared-KV deduplication).
"""
import argparse, csv, json, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from faithful_pfkv import path_to_str_rog                                    # noqa
from hf_reuse import build_preamble_suffix                                   # noqa

PATHS_FILE = {
    "rmanluo/RoG-cwq": "precomputed_paths/paths_rmanluo-RoG-cwq.json",
    "rmanluo/RoG-webqsp": "precomputed_paths/paths_rmanluo-RoG-webqsp.json",
    "metaqa-3hop": "precomputed_paths/paths_metaqa-3hop.json",
}


def load_queries(dataset, n, max_paths, seed=42):
    pf = json.load(open(ROOT / PATHS_FILE[dataset]))
    if dataset == "metaqa-3hop":
        from metaqa_adapter import load_metaqa
        ds = load_metaqa(hops=3, split="test", data_dir=str(ROOT / "metaQA")).shuffle(seed=seed)
    else:
        from datasets import load_dataset
        ds = load_dataset(dataset, split="test").shuffle(seed=seed)
    out = []
    for i in range(min(n, len(ds))):
        q = ds[i]["question"]
        paths = [[tuple(t) for t in p] for p in pf.get(q, [])][:max_paths]
        out.append((q, paths))
    return out


def topic_entity(paths):
    c = defaultdict(int)
    for p in paths:
        for h, r, t in p:
            c[h] += 1; c[t] += 1
    return max(c, key=c.get) if c else ""


def order_topical(items):
    return sorted(items, key=lambda it: topic_entity(it[1]))


def _paths_text(paths):
    return "".join(path_to_str_rog(p) + "\n" for p in paths)


@torch.no_grad()
def gen_batch(model, tok, input_texts, common_past=None, common_len=0, decode_tokens=32,
              device="cuda"):
    """Batched prefill+decode. If common_past given (DynamicCache [1,...]), broadcast
    it to the batch as a shared prefix. Returns (wall_s, prompt_tok, gen_tok, peak_mem_gb).
    Multi-GPU safe: each layer's broadcast KV stays on that layer's own device (for
    device_map='auto' model parallelism), so big batches fit across all GPUs."""
    B = len(input_texts)
    enc = tok(input_texts, return_tensors="pt", padding=True).to(device)
    ids, am = enc["input_ids"], enc["attention_mask"]
    for d in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(d)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    past = None
    if common_past is not None:
        from transformers import DynamicCache
        past = DynamicCache()
        # decode_tokens==0 (prefill/TTFT mode): broadcast as a VIEW (no .contiguous())
        # so the shared KV is stored ONCE, not B times -> reuse uses LESS memory than
        # baseline and fits much larger batches (the paper's shared-region claim). With
        # decode>0, DynamicCache.update would cat into the view and materialize B copies,
        # so we force contiguous there.
        materialize = decode_tokens > 0
        for li, layer in enumerate(common_past.layers):
            k = layer.keys.expand(B, -1, -1, -1)
            v = layer.values.expand(B, -1, -1, -1)
            if materialize:
                k = k.contiguous(); v = v.contiguous()
            past.update(k, v, li)
        full_am = torch.cat([torch.ones(B, common_len, device=device, dtype=am.dtype), am], dim=1)
        pos = torch.arange(common_len, common_len + ids.shape[1], device=device).unsqueeze(0).expand(B, -1)
        out = model(input_ids=ids, attention_mask=full_am, past_key_values=past,
                    position_ids=pos, use_cache=(decode_tokens > 0))
        am = full_am
    else:
        out = model(input_ids=ids, attention_mask=am, use_cache=(decode_tokens > 0))
    cache = out.past_key_values
    logits = out.logits[:, -1, :]
    cur = am.shape[1]
    gen = 0
    eos = tok.eos_token_id
    for _ in range(decode_tokens):
        nxt = logits.argmax(-1, keepdim=True)
        gen += B
        am = torch.cat([am, torch.ones(B, 1, device=device, dtype=am.dtype)], dim=1)
        pos = torch.full((B, 1), cur, device=device, dtype=torch.long)
        out = model(input_ids=nxt, attention_mask=am, past_key_values=cache,
                    position_ids=pos, use_cache=True)
        cache = out.past_key_values; logits = out.logits[:, -1, :]; cur += 1
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    peak = sum(torch.cuda.max_memory_allocated(d) for d in range(torch.cuda.device_count())) / 1e9
    prompt_tok = int(am[:, :ids.shape[1] + common_len].sum().item()) if common_past is not None else int(enc["attention_mask"].sum().item())
    return wall, prompt_tok, gen, peak


@torch.no_grad()
def prefill_shared(model, tok, shared_paths, template, device):
    """Prefill [preamble + union shared paths] once -> common past + length + wall_s."""
    pre_ids, _ = build_preamble_suffix(tok, "x", template)
    text = _paths_text(shared_paths)
    sids = tok.encode(text, add_special_tokens=False)
    ids = pre_ids + sids
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = model(input_ids=torch.tensor([ids], device=device), use_cache=True)
    torch.cuda.synchronize()
    return out.past_key_values, len(ids), time.perf_counter() - t0


def run(model, tok, items, batches, template, decode_tokens, device, out_rows, model_name, dataset):
    _, suf_ids = build_preamble_suffix(tok, "x", template)
    for B in batches:
        if B > len(items):
            continue
        # take first B topical queries as one batch
        batch = items[:B]
        # shared = union of paths with r>1 in batch
        cnt = defaultdict(int); str2p = {}
        for _q, ps in batch:
            for s in set(path_to_str_rog(p) for p in ps):
                cnt[s] += 1
            for p in ps:
                str2p[path_to_str_rog(p)] = p
        shared_strs = {s for s, c in cnt.items() if c > 1}
        shared_paths = [str2p[s] for s in shared_strs]
        # baseline prompts: preamble + ALL paths + query
        base_prompts, reuse_tails = [], []
        for q, ps in batch:
            pre_txt = tok.decode(build_preamble_suffix(tok, q, template)[0])
            suf_txt = tok.decode(suf_ids)  # generic; query embedded below
            # rebuild exact strings
            pre_ids, sfx_ids = build_preamble_suffix(tok, q, template)
            base_prompts.append(tok.decode(pre_ids) + _paths_text(ps) + tok.decode(sfx_ids))
            uniq = [p for p in ps if path_to_str_rog(p) not in shared_strs]
            reuse_tails.append(_paths_text(uniq) + tok.decode(sfx_ids))
        res = {"model": model_name, "dataset": dataset, "batch": B,
               "n_shared_paths": len(shared_paths),
               "shared_frac": sum(1 for _q, ps in batch for p in ps if path_to_str_rog(p) in shared_strs) / max(1, sum(len(ps) for _q, ps in batch))}
        # baseline
        try:
            w, pt, gt, pk = gen_batch(model, tok, base_prompts, None, 0, decode_tokens, device)
            res.update(base_tput=(pt + gt) / w, base_wall=w, base_peak_gb=pk, base_oom=0,
                       base_prompt_tok=pt)
        except Exception:
            torch.cuda.empty_cache(); res.update(base_tput=float("nan"), base_peak_gb=float("nan"), base_oom=1, base_prompt_tok=0, base_wall=float("nan"))
        # reuse: shared prefix prefilled ONCE (amortized over B) + batched unique
        try:
            cpast, clen, sw = prefill_shared(model, tok, shared_paths, template, device)
            w, pt, gt, pk = gen_batch(model, tok, reuse_tails, cpast, clen, decode_tokens, device)
            del cpast
            wtot = w + sw  # include one-time shared prefill (honest amortization)
            # ANALYTICAL ideal-reuse peak mem: shared stored ONCE (HF replicates it via
            # broadcast, so measured peak is an upper bound; ideal = base - replicated-shared).
            res.update(reuse_tput=(pt + gt) / wtot, reuse_wall=wtot, reuse_peak_gb=pk, reuse_oom=0,
                       reuse_prompt_tok=pt, shared_prefix_len=clen, shared_prefill_s=sw)
        except Exception:
            torch.cuda.empty_cache(); res.update(reuse_tput=float("nan"), reuse_peak_gb=float("nan"), reuse_oom=1, reuse_prompt_tok=0, reuse_wall=float("nan"), shared_prefix_len=0)
        sp = (res.get("reuse_tput") or 0) / res["base_tput"] if res.get("base_tput") else float("nan")
        res["tput_speedup"] = sp
        out_rows.append(res)
        print(f"  B={B:<5} shared={res['shared_frac']*100:4.1f}% | base {res.get('base_tput',float('nan')):8.1f} tok/s "
              f"({res.get('base_peak_gb',0):.1f}GB oom={res.get('base_oom')}) | "
              f"reuse {res.get('reuse_tput',float('nan')):8.1f} tok/s ({res.get('reuse_peak_gb',0):.1f}GB oom={res.get('reuse_oom')}) "
              f"| {sp:.2f}x", flush=True)
        torch.cuda.empty_cache()
        # Once EITHER condition OOMs, all larger batches OOM too AND pushing further
        # risks a non-recoverable device-side assert near the memory wall -> stop.
        if res.get("base_oom") or res.get("reuse_oom"):
            print(f"  [stop sweep: OOM at B={B}]", flush=True)
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b")
    ap.add_argument("--dataset", default="rmanluo/RoG-cwq")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 16, 32, 64, 128, 256, 512, 1024])
    ap.add_argument("--n_queries", type=int, default=1024)
    ap.add_argument("--max_paths", type=int, default=15)
    ap.add_argument("--decode_tokens", type=int, default=32)
    ap.add_argument("--device_map", default=None, help="'auto' = model-parallel across all visible GPUs (big batches)")
    ap.add_argument("--out", default="reviewer_b_gen/results/mem_throughput.csv")
    a = ap.parse_args()
    template = "rog" if "rog" in a.model.lower() else "fewshot"
    tok_src = "meta-llama/Llama-2-7b-hf" if "rog" in a.model.lower() else a.model
    print(f"Loading {a.model} (tmpl={template}, max_paths={a.max_paths}, device_map={a.device_map})", flush=True)
    tok = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    if a.device_map == "auto":
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16, device_map="auto").eval()
        device = model.get_input_embeddings().weight.device  # inputs go to embedding device
    else:
        device = "cuda"
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16).to(device).eval()
    items = order_topical(load_queries(a.dataset, a.n_queries, a.max_paths))
    # warmup (autotune CUDA kernels) so the first timed batch isn't cold-biased
    try:
        _w = [tok.decode(build_preamble_suffix(tok, q, template)[0]) + _paths_text(ps)
              for q, ps in items[:8]]
        gen_batch(model, tok, _w, None, 0, 4, device)
        torch.cuda.synchronize(); torch.cuda.empty_cache()
    except Exception as e:
        print(f"[warmup skipped: {e}]", flush=True)
    rows = []
    print(f"[{a.dataset}] {len(items)} topical queries", flush=True)
    run(model, tok, items, a.batches, template, a.decode_tokens, device, rows, a.model, a.dataset)
    out = Path(ROOT / a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
