#!/usr/bin/env python3
"""
gen_repro2.py — Reviewer-B generation reproduction, v2 (isolated).

Fixes vs v1: models the paper's ACTUAL mechanism — a GLOBAL persistent path-KV
cache (generation.py keeps `self._pfkv_path_cache = {}` for cross-query reuse, and
the cache persists across the whole run). Each distinct path is prefilled ONCE EVER
and reused (RoPE-realigned) whenever it recurs. So the reused fraction is the
GLOBAL cache-hit rate, which warms up far past the ~30% within-batch redundancy and
is what drives the paper's "removes the prefill stage" / 2.5x-TTFT claim.

Per query q (paths P), in arrival order, with a warm global cache C:
  hit  = paths of P already in C   (served from region, realigned; NOT re-forwarded)
  miss = paths of P not in C       (forwarded in-context, then ADDED to C)
  preamble KV is query-independent -> computed once, cloned per query.
  TTFT_reuse(q) = forward([miss-text + query-suffix]) on past=[preamble ++ hit-KV].
  TTFT_full(q)  = forward([preamble + all-paths + query]).

We report TTFT (cold-to-warm and warm-tail), the cache-hit token fraction, and
batch throughput. Accuracy fidelity is established elsewhere (realignment engine).
"""
import argparse, csv, json, os, sys, time, copy
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from faithful_pfkv import build_prompt, path_to_str_rog, encode_prefix_ids   # noqa
from hf_reuse import HFPathCache, build_preamble_suffix, _layer_kv_device     # noqa
from selective_recompute import _rope_cos_sin, _apply_rope                    # noqa
try:
    from transformers import DynamicCache
except Exception:
    DynamicCache = None

PATHS_FILE = {
    "rmanluo/RoG-cwq": "precomputed_paths/paths_rmanluo-RoG-cwq.json",
    "rmanluo/RoG-webqsp": "precomputed_paths/paths_rmanluo-RoG-webqsp.json",
    "metaqa-3hop": "precomputed_paths/paths_metaqa-3hop.json",
}


def load_queries(dataset, n_queries, max_paths, seed=42):
    pf = json.load(open(ROOT / PATHS_FILE[dataset]))
    if dataset == "metaqa-3hop":
        from metaqa_adapter import load_metaqa
        ds = load_metaqa(hops=3, split="test", data_dir=str(ROOT / "metaQA")).shuffle(seed=seed)
    else:
        from datasets import load_dataset
        ds = load_dataset(dataset, split="test").shuffle(seed=seed)
    out = []
    for i in range(min(n_queries, len(ds))):
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


def order_queries(items, policy):
    if policy == "arrival":
        return items
    if policy == "topical":
        return sorted(items, key=lambda it: topic_entity(it[1]))
    raise ValueError(policy)


def _cuda_ms(fn):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    r = fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) * 1000.0, r


@torch.no_grad()
def build_preamble_kv(model, tok, template, device):
    """Preamble is query-independent -> compute its KV once; store per-layer K,V."""
    pre_ids, _ = build_preamble_suffix(tok, "PLACEHOLDER", template)
    out = model(input_ids=torch.tensor([pre_ids], device=device), use_cache=True)
    c = out.past_key_values
    kv = [(layer.keys.clone(), layer.values.clone()) for layer in c.layers]
    return kv, len(pre_ids)


def _clone_preamble_cache(pre_kv, device):
    dyn = DynamicCache()
    for li, (k, v) in enumerate(pre_kv):
        dyn.update(k.clone(), v.clone(), li)
    return dyn


@torch.no_grad()
def ttft_full(model, tok, query, paths, max_paths, template, device):
    full_ids, _, _, _ = build_prompt(tok, query, paths, max_paths, template)
    ms, _ = _cuda_ms(lambda: model(input_ids=torch.tensor([full_ids], device=device),
                                    use_cache=True).logits[0, -1])
    return ms, len(full_ids)


@torch.no_grad()
def ttft_reuse(model, tok, pcache, pre_kv, P_pre, query, hit_paths, miss_paths,
               template, device):
    """Single forward over [miss-text + suffix] given past = preamble ++ realigned
    hit-KV. Returns (ttft_ms, forwarded_tokens, cached_tokens)."""
    _, suf_ids = build_preamble_suffix(tok, query, template)
    nlayers = len(pre_kv)

    def go():
        cache = _clone_preamble_cache(pre_kv, device)
        cur = P_pre
        # assemble hit-path KV: per layer, concat + single rotary realign
        per_k = [[] for _ in range(nlayers)]
        per_v = [[] for _ in range(nlayers)]
        deltas = []
        for p in hit_paths:
            k_list, v_list, L, _ = pcache.get(p)
            if L == 0:
                continue
            for li in range(nlayers):
                per_k[li].append(k_list[li]); per_v[li].append(v_list[li])
            deltas.append((cur - pcache.P_enc, L)); cur += L
        if deltas:
            for li in range(nlayers):
                dev_li = _layer_kv_device(cache, li, device)
                kcat = torch.cat([k.to(dev_li) for k in per_k[li]], dim=2)
                vcat = torch.cat([v.to(dev_li, model.dtype) for v in per_v[li]], dim=2)
                if pcache.uses_rope and os.environ.get("GROPT_NO_REALIGN") != "1":
                    dvec = torch.cat([torch.full((L,), float(d), device=dev_li) for d, L in deltas])
                    cos, sin = _rope_cos_sin(dvec, pcache.hd, pcache.theta, dev_li, pcache.dtype)
                    kk = _apply_rope(kcat[0].transpose(0, 1), cos, sin)
                    kcat = kk.transpose(0, 1).unsqueeze(0)
                cache.update(kcat.to(model.dtype), vcat, li)
        tail = "".join(path_to_str_rog(p) + "\n" for p in miss_paths)
        tail_ids = tok.encode(tail, add_special_tokens=False) + suf_ids
        total = cur
        pos = torch.arange(total, total + len(tail_ids), device=device).unsqueeze(0)
        am = torch.ones(1, total + len(tail_ids), device=device, dtype=torch.long)
        model(input_ids=torch.tensor([tail_ids], device=device), past_key_values=cache,
              position_ids=pos, attention_mask=am, use_cache=True)
        return len(tail_ids), cur - P_pre
    ms, (fwd_tok, cached_tok) = _cuda_ms(go)
    return ms, fwd_tok, cached_tok


def run_stream(model, tok, items, template, max_paths, enc_prefix, device,
               cache_cap=1500, warm_tail_frac=0.5):
    """Stream queries in given order through a GLOBAL warming path cache."""
    pcache = HFPathCache(model, tok, device, encode_prefix_ids=enc_prefix)
    pre_kv, P_pre = build_preamble_kv(model, tok, template, device)
    full_ms, reuse_ms, cached_frac, hitrate = [], [], [], []
    fwd_full, fwd_reuse = [], []
    for qi, (q, paths) in enumerate(items):
        seen = set(pcache.cache.keys())
        hit = [p for p in paths if path_to_str_rog(p) in seen]
        miss = [p for p in paths if path_to_str_rog(p) not in seen]
        tf, lf = ttft_full(model, tok, q, paths, max_paths, template, device)
        tr, fr, ct = ttft_reuse(model, tok, pcache, pre_kv, P_pre, q, hit, miss, template, device)
        # warm the cache with miss paths (so future queries reuse them)
        for p in miss:
            pcache.get(p)
        if len(pcache.cache) > cache_cap:
            pcache.cache.clear(); torch.cuda.empty_cache()
        full_ms.append(tf); reuse_ms.append(tr)
        fwd_full.append(lf); fwd_reuse.append(fr)
        cached_frac.append(ct / max(1, lf))
        hitrate.append(len(hit) / max(1, len(paths)))
    n = len(full_ms); tail0 = int(n * (1 - warm_tail_frac))
    def agg(xs, s=0):
        a = xs[s:]
        return float(np.mean(a)), float(np.median(a))
    mf, _ = agg(full_ms); mr, _ = agg(reuse_ms)
    mf_w, _ = agg(full_ms, tail0); mr_w, _ = agg(reuse_ms, tail0)
    del pcache; torch.cuda.empty_cache()
    return {
        "n_q": n,
        "ttft_full_ms": mf, "ttft_reuse_ms": mr, "ttft_speedup": mf / mr if mr else float("nan"),
        "ttft_full_warm_ms": mf_w, "ttft_reuse_warm_ms": mr_w,
        "ttft_speedup_warm": mf_w / mr_w if mr_w else float("nan"),
        "hit_rate": float(np.mean(hitrate)), "hit_rate_warm": float(np.mean(hitrate[tail0:])),
        "cached_token_frac": float(np.mean(cached_frac)),
        "cached_token_frac_warm": float(np.mean(cached_frac[tail0:])),
        "mean_fwd_tokens_full": float(np.mean(fwd_full)),
        "mean_fwd_tokens_reuse": float(np.mean(fwd_reuse)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rmanluo/RoG")
    ap.add_argument("--datasets", nargs="+", default=["rmanluo/RoG-cwq"])
    ap.add_argument("--policies", nargs="+", default=["arrival", "topical"])
    ap.add_argument("--n_queries", type=int, default=512)
    ap.add_argument("--max_paths", type=int, default=30)
    ap.add_argument("--out", default="reviewer_b_gen/results/gen_repro2.csv")
    a = ap.parse_args()
    device = "cuda"
    template = "rog" if "rog" in a.model.lower() else "fewshot"
    tok_src = "meta-llama/Llama-2-7b-hf" if "rog" in a.model.lower() else a.model
    print(f"Loading {a.model} (tok={tok_src}, tmpl={template}, max_paths={a.max_paths})", flush=True)
    tok = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16).to(device).eval()
    enc = encode_prefix_ids(tok, template)
    rows = []
    for ds in a.datasets:
        items_all = load_queries(ds, a.n_queries, a.max_paths)
        print(f"\n[{ds}] {len(items_all)} queries", flush=True)
        for policy in a.policies:
            items = order_queries(items_all, policy)
            m = run_stream(model, tok, items, template, a.max_paths, enc, device)
            m = {"model": a.model, "dataset": ds, "policy": policy, **m}
            rows.append(m)
            print(f"  {ds.split('/')[-1]:10} {policy:7} hit={m['hit_rate']*100:4.1f}% "
                  f"(warm {m['hit_rate_warm']*100:4.1f}%)  cachedTok={m['cached_token_frac']*100:4.1f}% "
                  f"(warm {m['cached_token_frac_warm']*100:4.1f}%)  "
                  f"TTFT full={m['ttft_full_ms']:6.1f} reuse={m['ttft_reuse_ms']:6.1f} "
                  f"({m['ttft_speedup']:.2f}x; warm {m['ttft_speedup_warm']:.2f}x)", flush=True)
    out = Path(ROOT / a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
