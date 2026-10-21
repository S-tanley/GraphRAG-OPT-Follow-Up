#!/usr/bin/env python3
"""Reviewer B batch-size sweep using the KV-cache ON vs OFF configuration (the config
that reproduces the paper-magnitude generation speedup). Measures batched generation
throughput (tok/s) with standard decode KV cache enabled (use_cache=True) vs disabled
(use_cache=False), across batch sizes, on the reviewer's models. HONEST label: this is
the standard decode-KV-cache (use_cache) effect, i.e. the paper's profiler graphrag_kv-vs-
graphrag comparison -- NOT the subgraph path reuse (that is W_red ~1.4-1.7x work-reduction).

Self-contained (AutoModelForCausalLM handles OPT .bin)."""
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from faithful_pfkv import path_to_str_rog                              # noqa
from hf_reuse import build_preamble_suffix                            # noqa

PF = {"rmanluo/RoG-cwq": "precomputed_paths/paths_rmanluo-RoG-cwq.json"}


def load_prompts(tok, template, n, max_paths=8):
    pf = json.load(open(ROOT / PF["rmanluo/RoG-cwq"]))
    from datasets import load_dataset
    ds = load_dataset("rmanluo/RoG-cwq", split="test").shuffle(seed=42)
    out = []
    for i in range(min(n, len(ds))):
        q = ds[i]["question"]
        paths = [[tuple(t) for t in p] for p in pf.get(q, [])][:max_paths]
        pre, suf = build_preamble_suffix(tok, q, template)
        txt = tok.decode(pre) + "".join(path_to_str_rog(p) + "\n" for p in paths) + tok.decode(suf)
        out.append(txt)
    return out


@torch.no_grad()
def throughput(model, tok, prompts, use_cache, decode_tokens, device):
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    ids, am = enc["input_ids"], enc["attention_mask"]
    B = ids.shape[0]
    for d in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(d)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = model(input_ids=ids, attention_mask=am, use_cache=use_cache)
    past = out.past_key_values if use_cache else None
    logits = out.logits[:, -1, :]; cur = ids.shape[1]; gen = 0
    full = ids
    for _ in range(decode_tokens):
        nxt = logits.argmax(-1, keepdim=True); gen += B
        am = torch.cat([am, torch.ones(B, 1, device=device, dtype=am.dtype)], dim=1)
        if use_cache:
            out = model(input_ids=nxt, attention_mask=am, past_key_values=past, use_cache=True)
            past = out.past_key_values
        else:  # no cache: re-feed the whole growing sequence (standard non-cached decode)
            full = torch.cat([full, nxt], dim=1)
            out = model(input_ids=full, attention_mask=am, use_cache=False)
        logits = out.logits[:, -1, :]; cur += 1
    torch.cuda.synchronize(); wall = time.perf_counter() - t0
    ptok = int(enc["attention_mask"].sum().item())
    return (ptok + gen) / wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--batches", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256, 512])
    ap.add_argument("--decode_tokens", type=int, default=16)
    ap.add_argument("--max_paths", type=int, default=8)
    ap.add_argument("--device_map", default=None)
    ap.add_argument("--out", default="reviewer_cde/results/kv_sweep.csv")
    a = ap.parse_args()
    template = "rog" if "rog" in a.model.lower() else "fewshot"
    tok_src = "meta-llama/Llama-2-7b-hf" if "rog" in a.model.lower() else a.model
    tok = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    if a.device_map == "auto":
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16, device_map="auto").eval()
        device = model.get_input_embeddings().weight.device
    else:
        device = "cuda"
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16).to(device).eval()
    prompts_all = load_prompts(tok, template, max(a.batches), a.max_paths)
    # warmup
    try: throughput(model, tok, prompts_all[:4], True, 2, device)
    except Exception: pass
    rows = []
    for B in a.batches:
        if B > len(prompts_all): continue
        pr = prompts_all[:B]
        try:
            on = throughput(model, tok, pr, True, a.decode_tokens, device)
        except Exception as e:
            print(f"B={B} KVon FAIL {type(e).__name__}"); torch.cuda.empty_cache(); break
        try:
            off = throughput(model, tok, pr, False, a.decode_tokens, device)
        except Exception as e:
            off = float("nan"); print(f"B={B} KVoff FAIL {type(e).__name__}"); torch.cuda.empty_cache()
        sp = on / off if off == off and off > 0 else float("nan")
        rows.append({"model": a.model, "batch": B, "tput_kv_on": on, "tput_kv_off": off, "speedup": sp})
        print(f"  B={B:<5} KVon {on:8.0f} t/s  KVoff {off:8.0f} t/s  -> {sp:.2f}x", flush=True)
        torch.cuda.empty_cache()
        if off != off:  # KVoff OOM -> stop
            break
    out = Path(ROOT / a.out); out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header: w.writeheader()
        w.writerows(rows)
    print(f"Appended {len(rows)} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
