"""
Generic (architecture-agnostic) faithful subgraph KV reuse via HuggingFace.

Works for any RoPE-based HF causal LM (Llama, Gemma-2, Qwen3, ...). Realizes the
GraphRAG-Opt reuse honestly:

    1. forward the preamble (instruction + query + "KG Paths:") -> KV cache
    2. inject each retrieved path's cached KV, RoPE-realigned to its correct
       position AFTER the preamble (no path text in the prompt -> no
       double-representation)
    3. forward the suffix ("### Response:") on the combined cache and greedily
       decode the answer

KV is captured directly from model() (post-RoPE keys) and realigned with the
correct *rotate_half* convention HF uses. (subgraph_kv_cache.py rotates with the
interleaved even/odd convention, which is wrong for these models and yields
0 accuracy — hence this module.) A path's KV is encoded once and realigned per
query (global cross-query reuse).
"""

import os

import torch

from faithful_pfkv import path_to_str_rog, extract_answer
from selective_recompute import _rope_cos_sin, _apply_rope

_ROPE_TYPES = {"llama", "gemma", "gemma2", "qwen2", "qwen3", "mistral", "phi", "phi3"}


def _layer_kv_device(cache, li, default):
    """Device where layer li's KV currently lives. For a model loaded with
    device_map='auto' (model parallel, e.g. 72B across 4 GPUs) different layers
    sit on different GPUs, so injected/realigned KV for layer li must land on
    that layer's device. Single-GPU models fall back to `default` (cuda:0)."""
    try:
        k = getattr(cache.layers[li], "keys", None)
        if k is not None and k.numel() > 0:
            return k.device
    except (AttributeError, IndexError, TypeError):
        pass
    try:                                   # older cache API
        return cache.key_cache[li].device
    except (AttributeError, IndexError, TypeError):
        pass
    return default


class HFPathCache:
    """Encodes path KV with model() and realigns with rotate_half RoPE."""

    def __init__(self, model, tok, device="cuda", encode_prefix_ids=None):
        self.model, self.tok, self.device = model, tok, device
        self.dtype = next(model.parameters()).dtype
        cfg = model.config
        self.hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        # rope_theta location varies by model/transformers version (e.g. Qwen3
        # stores it under config.rope_parameters). Wrong theta => wrong realign.
        theta = getattr(cfg, "rope_theta", None)
        if theta is None:
            for attr in ("rope_parameters", "rope_scaling"):
                rp = getattr(cfg, attr, None)
                if isinstance(rp, dict) and rp.get("rope_theta"):
                    theta = rp["rope_theta"]
                    break
        self.theta = float(theta or 10000.0)
        self.uses_rope = (getattr(cfg, "model_type", "") in _ROPE_TYPES) or hasattr(cfg, "rope_theta")
        self.prefix = list(encode_prefix_ids) if encode_prefix_ids else []
        self.P_enc = len(self.prefix)
        self.cache = {}   # path_str -> (k_list, v_list, length)  [post-RoPE, at enc positions]

    @torch.no_grad()
    def _encode(self, path_ids):
        ids = self.prefix + list(path_ids)
        out = self.model(input_ids=torch.tensor([ids], device=self.device), use_cache=True)
        c = out.past_key_values
        L = len(path_ids)
        k_list, v_list = [], []
        for layer in c.layers:
            k_list.append(layer.keys[:, :, self.P_enc:, :].clone())   # [1,Hkv,L,hd] post-RoPE
            v_list.append(layer.values[:, :, self.P_enc:, :].clone())
        return k_list, v_list, L, list(path_ids)

    def get(self, path):
        s = path_to_str_rog(path)
        if s not in self.cache:
            # GROPT_PATH_NEWLINE=1 encodes the path line WITH its trailing "\n", as
            # build_prompt (full prefill) does; default keeps the original behavior.
            if os.environ.get("GROPT_PATH_NEWLINE") == "1":
                ids = self.tok.encode(s + "\n", add_special_tokens=False)
            else:
                ids = self.tok.encode(s, add_special_tokens=False)
            self.cache[s] = self._encode(ids) if ids else ([], [], 0, [])
        return self.cache[s]

    def _realign_k(self, k, delta):
        """Rotate post-RoPE keys (encoded at positions enc..) to target by a
        uniform position delta, using rotate_half. k: [1,Hkv,L,hd].
        Device-aware: the RoPE tables are built on k's OWN device so reuse works
        under model parallelism (device_map='auto'), where layer li's KV lives on
        a different GPU than cuda:0."""
        if not self.uses_rope or delta == 0 or os.environ.get("GROPT_NO_REALIGN") == "1":
            # GROPT_NO_REALIGN=1 disables positional realignment -> SubGCache-style
            # "no positional handling" ablation (council B2): exposes the cost of
            # reusing KV verbatim at a shifted offset without re-rotating keys.
            return k
        L = k.shape[2]
        dev = k.device
        pos = torch.full((L,), float(delta), device=dev)
        cos, sin = _rope_cos_sin(pos, self.hd, self.theta, dev, self.dtype)  # [L,hd]
        kk = k[0].transpose(0, 1)            # [L,Hkv,hd]
        kk = _apply_rope(kk, cos, sin)       # apply R(delta) -> R(enc+delta)
        return kk.transpose(0, 1).unsqueeze(0)


@torch.no_grad()
def full_generate(model, tok, query, paths, template="rog", max_paths=8,
                  max_new_tokens=16, device="cuda"):
    """Full in-context prefill + greedy decode over the SAME prompt build_prompt
    produces (path text included, no reuse). Used for Direct / GraphRAG /
    GraphRAG-R so they share the exact prompt + decode as GraphRAG-Opt, making
    GraphRAG-R literally 'GraphRAG-Opt with reuse off'."""
    from faithful_pfkv import build_prompt
    full_ids, _, _, _ = build_prompt(tok, query, paths, max_paths, template)
    eos = tok.eos_token_id
    out = model(input_ids=torch.tensor([full_ids], device=device), use_cache=True)
    cache, logits = out.past_key_values, out.logits[0, -1]
    cur = len(full_ids)
    gen = []
    for _ in range(max_new_tokens):
        tid = int(logits.argmax())
        if eos is not None and tid == eos:
            break
        gen.append(tid)
        am = torch.ones(1, cur + 1, device=device, dtype=torch.long)
        out = model(input_ids=torch.tensor([[tid]], device=device), past_key_values=cache,
                    position_ids=torch.tensor([[cur]], device=device), attention_mask=am, use_cache=True)
        cache, logits = out.past_key_values, out.logits[0, -1]
        cur += 1
    return extract_answer(tok.decode(gen, skip_special_tokens=True))


def build_preamble_suffix(tok, query, template="rog"):
    """MUST match faithful_pfkv.build_prompt's preamble/suffix EXACTLY so the
    reuse path (graphrag_opt) and the full-prefill path (graphrag_r) share one
    prompt format -- otherwise the RoG model (fine-tuned on the [INST] reasoning
    format) sees a different prompt and the measured 'reuse cost' is actually a
    prompt mismatch. The query goes in the SUFFIX (after the paths), as in RoG."""
    if template == "fewshot":
        from faithful_pfkv import _FEWSHOT
        pre, suf = _FEWSHOT, f"\nQ: {query}\nA:"
    else:
        pre = ("[INST] <<SYS>>\n<</SYS>>\nBased on the reasoning paths, please answer "
               "the given question. Please keep the answer as simple as possible and "
               "return all the possible answers as a list.\n\nReasoning Paths:\n")
        suf = f"\nQuestion:\n{query} [/INST]"
    return tok.encode(pre, add_special_tokens=True), tok.encode(suf, add_special_tokens=False)


@torch.no_grad()
def generic_reuse_generate(model, tok, pcache, query, paths, template="rog",
                           max_paths=8, max_new_tokens=16, device="cuda", recompute_k=0):
    """recompute_k: EPIC-style # of first tokens of each path to RECOMPUTE
    in-context (fresh KV) instead of reusing cached KV; the rest are injected
    from cache. recompute_k=0 = pure reuse (lossy); larger k recovers accuracy."""
    pre_ids, suf_ids = build_preamble_suffix(tok, query, template)
    P = len(pre_ids)
    eos = tok.eos_token_id

    # 1) preamble forward -> cache
    out = model(input_ids=torch.tensor([pre_ids], device=device), use_cache=True)
    cache = out.past_key_values

    # 2) place each path after the preamble: recompute first-k tokens in-context,
    #    inject the rest from cache (RoPE-realigned).
    cur_pos = P
    for p in paths[:max_paths]:
        k_list, v_list, L, ptoks = pcache.get(p)
        if L == 0:
            continue
        krec = min(recompute_k, L)
        if krec > 0:                          # (a) recompute first krec tokens fresh
            rpos = torch.arange(cur_pos, cur_pos + krec, device=device).unsqueeze(0)
            am = torch.ones(1, cur_pos + krec, device=device, dtype=torch.long)
            out = model(input_ids=torch.tensor([ptoks[:krec]], device=device),
                        past_key_values=cache, position_ids=rpos,
                        attention_mask=am, use_cache=True)
            cache = out.past_key_values
        if L > krec:                          # (b) inject remaining cached KV
            delta = cur_pos - pcache.P_enc
            for li in range(len(k_list)):
                dev_li = _layer_kv_device(cache, li, device)   # model-parallel safe
                k = pcache._realign_k(k_list[li][:, :, krec:, :].to(dev_li), delta)
                cache.update(k.to(model.dtype),
                             v_list[li][:, :, krec:, :].to(dev_li, model.dtype), li)
        cur_pos += L
    total = cur_pos

    # 3) suffix forward + manual greedy decode (robust to the 5.x generate/cache bug)
    pos = torch.arange(total, total + len(suf_ids), device=device).unsqueeze(0)
    am = torch.ones(1, total + len(suf_ids), device=device, dtype=torch.long)
    out = model(input_ids=torch.tensor([suf_ids], device=device), past_key_values=cache,
                position_ids=pos, attention_mask=am, use_cache=True)
    cache, logits = out.past_key_values, out.logits[0, -1]
    cur = total + len(suf_ids)
    gen_ids = []
    for _ in range(max_new_tokens):
        tid = int(logits.argmax())
        if eos is not None and tid == eos:
            break
        gen_ids.append(tid)
        am = torch.ones(1, cur + 1, device=device, dtype=torch.long)
        out = model(input_ids=torch.tensor([[tid]], device=device), past_key_values=cache,
                    position_ids=torch.tensor([[cur]], device=device), attention_mask=am, use_cache=True)
        cache, logits = out.past_key_values, out.logits[0, -1]
        cur += 1
    return extract_answer(tok.decode(gen_ids, skip_special_tokens=True))


@torch.no_grad()
def partial_reuse_generate(model, tok, pcache, query, reused_paths, computed_paths,
                           template="rog", max_new_tokens=16, device="cuda", recompute_k=0):
    """The paper's / SubGCache's actual scenario: reuse the cached KV of the
    SHARED (cross-query, recurring) paths; compute the UNIQUE paths in-context.
    reused_paths are injected as cached KV (optional EPIC first-k recompute);
    computed_paths are serialized as text after them so they get full attention."""
    pre_ids, suf_ids = build_preamble_suffix(tok, query, template)
    eos = tok.eos_token_id
    out = model(input_ids=torch.tensor([pre_ids], device=device), use_cache=True)
    cache = out.past_key_values
    cur = len(pre_ids)
    for p in reused_paths:
        k_list, v_list, L, ptoks = pcache.get(p)
        if L == 0:
            continue
        krec = min(recompute_k, L)
        if krec > 0:
            rpos = torch.arange(cur, cur + krec, device=device).unsqueeze(0)
            am = torch.ones(1, cur + krec, device=device, dtype=torch.long)
            out = model(input_ids=torch.tensor([ptoks[:krec]], device=device),
                        past_key_values=cache, position_ids=rpos, attention_mask=am, use_cache=True)
            cache = out.past_key_values
        if L > krec:
            delta = cur - pcache.P_enc
            for li in range(len(k_list)):
                dev_li = _layer_kv_device(cache, li, device)   # model-parallel safe
                k = pcache._realign_k(k_list[li][:, :, krec:, :].to(dev_li), delta)
                cache.update(k.to(model.dtype),
                             v_list[li][:, :, krec:, :].to(dev_li, model.dtype), li)
        cur += L
    tail = "".join(path_to_str_rog(p) + "\n" for p in computed_paths)
    tail_ids = tok.encode(tail, add_special_tokens=False) + suf_ids
    pos = torch.arange(cur, cur + len(tail_ids), device=device).unsqueeze(0)
    am = torch.ones(1, cur + len(tail_ids), device=device, dtype=torch.long)
    out = model(input_ids=torch.tensor([tail_ids], device=device), past_key_values=cache,
                position_ids=pos, attention_mask=am, use_cache=True)
    cache, logits = out.past_key_values, out.logits[0, -1]
    c = cur + len(tail_ids)
    gen = []
    for _ in range(max_new_tokens):
        tid = int(logits.argmax())
        if eos is not None and tid == eos:
            break
        gen.append(tid)
        am = torch.ones(1, c + 1, device=device, dtype=torch.long)
        out = model(input_ids=torch.tensor([[tid]], device=device), past_key_values=cache,
                    position_ids=torch.tensor([[c]], device=device), attention_mask=am, use_cache=True)
        cache, logits = out.past_key_values, out.logits[0, -1]
        c += 1
    return extract_answer(tok.decode(gen, skip_special_tokens=True))


# --------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from config import set_reproducible_seed
    from faithful_pfkv import encode_prefix_ids
    from metrics import evaluate_metrics
    from retrieval import BatchOptimizedRetrieval

    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rmanluo/RoG")
    ap.add_argument("--dataset", default="rmanluo/RoG-cwq")
    ap.add_argument("--n_samples", type=int, default=100, help="-1 = full test set")
    ap.add_argument("--max_paths", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--hop", type=int, default=3)
    ap.add_argument("--encode_prefix", default="template", choices=["none", "template"])
    ap.add_argument("--cache_cap", type=int, default=4000)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    set_reproducible_seed(42)
    dev = "cuda"
    template = "rog" if "rog" in args.model.lower() else "fewshot"
    _metaqa = {"metaqa-1hop": 1, "metaqa-2hop": 2, "metaqa-3hop": 3}
    if args.dataset in _metaqa:
        from metaqa_adapter import load_metaqa
        mdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metaQA")
        ds = load_metaqa(hops=_metaqa[args.dataset], split="test", data_dir=mdir).shuffle(seed=42)
    else:
        ds = load_dataset(args.dataset, split="test").shuffle(seed=42)
    if args.n_samples and args.n_samples > 0:
        ds = ds.select(range(min(args.n_samples, len(ds))))
    if args.num_shards > 1:
        ds = ds.select([i for i in range(len(ds)) if i % args.num_shards == args.shard_id])
    results, _ = BatchOptimizedRetrieval(ds, device=dev).retrieve_batch(args.hop, True, args.max_paths)
    answers = [ds[i]["answer"] for i in range(len(ds))]

    tok_src = "meta-llama/Llama-2-7b-hf" if "rog" in args.model.lower() else args.model
    tok = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(dev).eval()
    enc = encode_prefix_ids(tok, template) if args.encode_prefix == "template" else None
    pcache = HFPathCache(model, tok, dev, encode_prefix_ids=enc)

    preds, gold = [], []
    for qi, r in enumerate(results):
        # Cover ALL queries (no-path ones generate from the query alone) so the
        # denominator matches the other Table-2 columns (regen_table2.py).
        if len(pcache.cache) > args.cache_cap:
            pcache.cache.clear(); torch.cuda.empty_cache()
        gold.append([list(a) if not isinstance(a, str) else a for a in [answers[qi]]][0])
        preds.append(generic_reuse_generate(model, tok, pcache, r["query"], r["paths"],
                                             template, args.max_paths, args.max_new_tokens, dev))
        if len(gold) % 100 == 0:
            print(f"[shard {args.shard_id}/{args.num_shards}] {len(gold)} done", flush=True)

    if args.out_json:
        json.dump({"answers": gold, "predictions": preds}, open(args.out_json, "w"))
        print(f"[shard {args.shard_id}/{args.num_shards}] wrote {len(gold)} -> {args.out_json}", flush=True)
    else:
        m = evaluate_metrics(preds, gold)
        print(f"\n{args.model} x {args.dataset}  GraphRAG-Opt (generic HF reuse, "
              f"encode_prefix={args.encode_prefix})  n={len(gold)}  "
              f"Hits@1={m['hits_at_1']:.4f}  F1={m['f1']:.4f}  substr={m['accuracy']:.4f}")
