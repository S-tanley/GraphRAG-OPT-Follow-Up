#!/usr/bin/env python3
"""
flops.py  —  Compare analytical FLOPs: RAG vs GraphRAG (hop 1–4)
=================================================================
Model : Qwen/Qwen3-8B
Datasets: rmanluo/RoG-cwq, metaqa-3hop

Runs real retrieval to get actual prompt lengths, then computes
analytical FLOPs for every stage (retrieval + prefill + decode),
accounting for GQA, attention scores, embedding/LM-head, and
retrieval-side compute.

Usage:
    python flops.py [--samples 64] [--datasets rmanluo/RoG-cwq metaqa-3hop]
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset

warnings.filterwarnings("ignore", category=FutureWarning)

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import set_reproducible_seed
from generation import PathFirstGraphRAGGeneration, format_prompt
from retrieval import SimpleGraphRAGRetrieval
from metaqa_adapter import load_metaqa

# ── Optional RAG deps ────────────────────────────────────────────────────────
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    _RAG_DEPS = True
except ImportError:
    _RAG_DEPS = False

MODEL_NAME = "Qwen/Qwen3-8B"
DATASETS   = ["rmanluo/RoG-cwq", "metaqa-3hop"]
HOPS       = [1, 2, 3, 4]
MAX_NEW_TOKENS = 128
NUM_BEAMS  = 1
BATCH_SIZE = 1
SEED       = 42
RAG_TOP_K  = 5


# ═══════════════════════════════════════════════════════════════════════════════
# Architecture extraction — keeps raw intermediate_size, no integer division
# ═══════════════════════════════════════════════════════════════════════════════

def get_arch(model) -> Dict:
    cfg = model.config
    d   = getattr(cfg, "hidden_size",         getattr(cfg, "d_model", 4096))
    nl  = getattr(cfg, "num_hidden_layers",   getattr(cfg, "n_layer", 32))
    nh  = getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", 32))
    nkv = getattr(cfg, "num_key_value_heads", nh)
    dh  = d // nh
    fi  = getattr(cfg, "intermediate_size",   getattr(cfg, "ffn_dim", 4 * d))
    V   = getattr(cfg, "vocab_size", 32000)
    return {"d": d, "nl": nl, "nh": nh, "nkv": nkv, "dh": dh, "fi": fi, "V": V}


# ═══════════════════════════════════════════════════════════════════════════════
# Analytical FLOP formulas — GQA-aware, includes all components
# ═══════════════════════════════════════════════════════════════════════════════

def flops_prefill(seq: int, arch: Dict, B: int = 1) -> Dict[str, float]:
    """
    FLOPs for the prefill phase (full-sequence forward pass).
    Returns breakdown dict and total.
    """
    d, nl, nh, nkv, dh, fi, V = (
        arch["d"], arch["nl"], arch["nh"], arch["nkv"],
        arch["dh"], arch["fi"], arch["V"])

    # ── Per-layer attention ───────────────────────────────────────────────
    # Q projection:  (B, seq, d) × (d, nh*dh) = (B, seq, d) × (d, d)
    q_proj  = 2 * B * seq * d * (nh * dh)
    # K projection:  (B, seq, d) × (d, nkv*dh)   ← GQA: nkv ≤ nh
    k_proj  = 2 * B * seq * d * (nkv * dh)
    # V projection:  same as K
    v_proj  = 2 * B * seq * d * (nkv * dh)

    # Attention scores: Q @ K^T  →  (B, nh, seq, dh) × (B, nh, dh, seq)
    # With GQA, K is broadcast from nkv to nh groups, but FLOPs are same
    attn_qk = 2 * B * nh * seq * seq * dh

    # Softmax: ~5 ops per element (exp, sum, div, sub-max, compare)
    attn_softmax = 5 * B * nh * seq * seq

    # Score @ V:  (B, nh, seq, seq) × (B, nh, seq, dh)
    attn_sv = 2 * B * nh * seq * seq * dh

    # Output projection: (B, seq, d) × (d, d)
    o_proj  = 2 * B * seq * d * d

    attn_total = q_proj + k_proj + v_proj + attn_qk + attn_softmax + attn_sv + o_proj

    # ── Per-layer FFN (SwiGLU: gate_proj, up_proj, down_proj) ─────────────
    gate = 2 * B * seq * d * fi      # (B, seq, d) × (d, fi)
    up   = 2 * B * seq * d * fi      # (B, seq, d) × (d, fi)
    down = 2 * B * seq * fi * d      # (B, seq, fi) × (fi, d)
    # SiLU activation: ~4 ops per element (exp, add, mul, neg)
    silu = 4 * B * seq * fi
    # Element-wise gate * up
    gate_mul = B * seq * fi

    ffn_total = gate + up + down + silu + gate_mul

    # ── Per-layer RMSNorm (×2 per layer: pre-attn, pre-ffn) ──────────────
    rmsnorm = 2 * (5 * B * seq * d)  # square, mean, rsqrt, mul, scale

    layer_total = attn_total + ffn_total + rmsnorm

    # ── Non-layer components ──────────────────────────────────────────────
    # Token embedding lookup: negligible FLOPs (just gather)
    embed = 0
    # Final RMSNorm
    final_norm = 5 * B * seq * d
    # LM head: (B, seq, d) × (d, V) — but only last token in practice,
    # however during prefill the full sequence is projected for KV building
    # In many implementations only last-token logits are computed
    lm_head = 2 * B * 1 * d * V  # only last token logits during prefill

    total = float(nl * layer_total + embed + final_norm + lm_head)

    return {
        "q_proj": q_proj * nl, "k_proj": k_proj * nl, "v_proj": v_proj * nl,
        "attn_qk": attn_qk * nl, "attn_softmax": attn_softmax * nl,
        "attn_sv": attn_sv * nl, "o_proj": o_proj * nl,
        "ffn_gate": gate * nl, "ffn_up": up * nl, "ffn_down": down * nl,
        "ffn_act": (silu + gate_mul) * nl,
        "rmsnorm": rmsnorm * nl, "final_norm": final_norm,
        "lm_head": lm_head,
        "total": total,
    }


def flops_decode(seq_prefill: int, arch: Dict, n_steps: int, B: int = 1) -> Dict[str, float]:
    """
    FLOPs for the decode phase (n_steps autoregressive steps).
    Each step generates 1 token attending over the growing KV cache.
    seq_prefill = prompt length (initial KV cache size).
    """
    d, nl, nh, nkv, dh, fi, V = (
        arch["d"], arch["nl"], arch["nh"], arch["nkv"],
        arch["dh"], arch["fi"], arch["V"])

    total_proj = 0.0
    total_attn = 0.0
    total_ffn = 0.0
    total_other = 0.0

    for step in range(n_steps):
        cache_len = seq_prefill + step  # KV cache length at this step

        # ── Per-layer attention ───────────────────────────────────────────
        # QKV projections: seq=1
        q_proj = 2 * B * 1 * d * (nh * dh)
        k_proj = 2 * B * 1 * d * (nkv * dh)
        v_proj = 2 * B * 1 * d * (nkv * dh)
        # O projection
        o_proj = 2 * B * 1 * d * d

        # Attention scores: (B, nh, 1, dh) × (B, nh, dh, cache_len)
        attn_qk = 2 * B * nh * 1 * cache_len * dh
        # Softmax over cache_len
        attn_softmax = 5 * B * nh * cache_len
        # Score @ V: (B, nh, 1, cache_len) × (B, nh, cache_len, dh)
        attn_sv = 2 * B * nh * 1 * cache_len * dh

        # ── Per-layer FFN ─────────────────────────────────────────────────
        gate = 2 * B * 1 * d * fi
        up   = 2 * B * 1 * d * fi
        down = 2 * B * 1 * fi * d
        silu = 4 * B * 1 * fi
        gate_mul = B * 1 * fi

        # ── Per-layer norms ───────────────────────────────────────────────
        rmsnorm = 2 * (5 * B * 1 * d)

        proj_step = (q_proj + k_proj + v_proj + o_proj) * nl
        attn_step = (attn_qk + attn_softmax + attn_sv) * nl
        ffn_step  = (gate + up + down + silu + gate_mul) * nl
        norm_step = rmsnorm * nl

        total_proj += proj_step
        total_attn += attn_step
        total_ffn  += ffn_step
        total_other += norm_step

    # Final norm + LM head per step
    final_norm = 5 * B * 1 * d * n_steps
    lm_head    = 2 * B * 1 * d * V * n_steps
    total_other += final_norm + lm_head

    total = float(total_proj + total_attn + total_ffn + total_other)

    return {
        "projections": total_proj,
        "attention_scores": total_attn,
        "ffn": total_ffn,
        "other": total_other,
        "total": total,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Retrieval FLOP estimation
# ═══════════════════════════════════════════════════════════════════════════════

# MiniLM-L6-v2 architecture constants
MINILM_D = 384
MINILM_NH = 12
MINILM_DH = 32
MINILM_FI = 1536  # intermediate_size
MINILM_NL = 6
MINILM_V = 30522


def flops_rag_retrieval(n_queries: int, avg_query_tokens: int,
                        n_passages: int, embed_dim: int = 384) -> Dict[str, float]:
    """
    FLOPs for RAG retrieval:
      1) Encode queries with MiniLM-L6-v2 (transformer forward pass)
      2) FAISS inner-product search (matrix multiply queries × passages^T)
    """
    seq = avg_query_tokens
    B = n_queries

    # ── MiniLM-L6-v2 forward pass (same structure as prefill) ─────────────
    d, nl, nh, dh, fi = MINILM_D, MINILM_NL, MINILM_NH, MINILM_DH, MINILM_FI

    # Per layer: QKV proj + attn scores + attn values + O proj + FFN (GELU, not SwiGLU)
    qkv    = 2 * B * seq * d * d * 3          # Q, K, V (MHA, not GQA)
    attn   = 2 * B * nh * seq * seq * dh * 2  # QK^T + score@V
    o_proj = 2 * B * seq * d * d
    # FFN: two linear layers (no gate)
    ff     = 2 * B * seq * d * fi + 2 * B * seq * fi * d
    embed_flops = float(nl * (qkv + attn + o_proj + ff))

    # ── FAISS inner-product search ────────────────────────────────────────
    # Brute-force IndexFlatIP: queries (n_queries, embed_dim) × passages^T (embed_dim, n_passages)
    faiss_flops = float(2 * n_queries * embed_dim * n_passages)

    total = embed_flops + faiss_flops
    return {
        "embedding_model": embed_flops,
        "faiss_search": faiss_flops,
        "total": total,
    }


def flops_graphrag_retrieval(n_queries: int, avg_edges_traversed: int) -> Dict[str, float]:
    """
    FLOPs for GraphRAG BFS retrieval (CPU graph traversal).
    Each edge traversal involves ~10 ops (hash lookup, comparison, queue ops).
    This is CPU-bound and small relative to LLM FLOPs, but included for completeness.
    """
    ops_per_edge = 10
    total = float(n_queries * avg_edges_traversed * ops_per_edge)
    return {
        "graph_traversal": total,
        "total": total,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RAG retrieval class
# ═══════════════════════════════════════════════════════════════════════════════

class RAGRetrieval:
    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self):
        if not _RAG_DEPS:
            raise ImportError("faiss / sentence-transformers not installed")
        self._embedder = SentenceTransformer(self.EMBED_MODEL, device="cpu")
        self._dim = self._embedder.get_sentence_embedding_dimension()
        self._index = None
        self._passages: List[str] = []

    def build_global_index(self, dataset) -> None:
        seen, passages = set(), []
        for i in range(len(dataset)):
            for triplet in dataset[i].get("graph", []):
                key = tuple(str(x).strip() for x in triplet)
                if key not in seen:
                    seen.add(key)
                    passages.append(f"{key[0]} | {key[1]} | {key[2]}")
        embs = self._embedder.encode(
            passages, batch_size=512, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=len(passages) > 500,
        ).astype(np.float32)
        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(embs)
        self._passages = passages
        print(f"  [RAG] Indexed {len(passages):,} passages")

    def retrieve(self, queries: List[str], top_k: int = 5) -> List[List[str]]:
        k = min(top_k, len(self._passages))
        qembs = self._embedder.encode(
            queries, batch_size=512, convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        _, idxs = self._index.search(qembs, k)
        results = []
        for row in idxs:
            results.append([self._passages[i] for i in row if 0 <= i < len(self._passages)])
        return results

    @property
    def n_passages(self) -> int:
        return len(self._passages)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset loading (same split as profile_rag_vs_graphrag.py)
# ═══════════════════════════════════════════════════════════════════════════════

def load_ds(name: str, n: int, seed: int):
    short = {"metaqa-1hop": 1, "metaqa-2hop": 2, "metaqa-3hop": 3}
    if name in short:
        hops = short[name]
        kb = SCRIPT_DIR / "metaQA" / "kb.txt"
        if kb.exists():
            ds = load_metaqa(hops=hops, split="test",
                             data_dir=str(SCRIPT_DIR / "metaQA"))
        elif hops == 3:
            ds = load_dataset("mhivanov123/metaqa-3hop", split="test")
        else:
            raise FileNotFoundError(f"MetaQA local files not found for {hops}hop")
    else:
        ds = load_dataset(name, split="test")
    if len(ds) > n:
        ds = ds.shuffle(seed=seed).select(range(n))
    return ds


def _rag_prompt(query: str, passages: List[str], model_name: str) -> str:
    ctx = "\n".join(passages) if passages else "N/A"
    if "rog" in model_name.lower():
        return (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            "Please reason over the following retrieved knowledge, "
            f"and then answer the question: {query}\n\n"
            "### Input:\n"
            f"Retrieved Knowledge:\n{ctx}\n\n"
            "### Response:"
        )
    return (
        "You are a concise, factual assistant. Use the following retrieved "
        "knowledge to answer the question.\n\n"
        f"Retrieved Knowledge:\n{ctx}\n\n"
        f"Q: {query}\nA:"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RAG vs GraphRAG FLOP comparison")
    parser.add_argument("--samples", type=int, default=64,
                        help="Number of samples per dataset (default 64)")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    set_reproducible_seed(SEED)

    # ── Load model (once) to get arch + tokenizer ─────────────────────────
    print(f"\n{'='*70}")
    print(f"  Loading model: {args.model}")
    print(f"{'='*70}")
    gen = PathFirstGraphRAGGeneration(
        args.model,
        device=f"cuda:{args.device}",
        use_kv_cache=True,
        use_pfkv=False,
        device_map=None,
        max_new_tokens=args.max_new_tokens,
        num_beams=NUM_BEAMS,
    )
    tokenizer = gen.tokenizer
    arch = get_arch(gen.model)
    params = sum(p.numel() for p in gen.model.parameters())

    print(f"  Arch: d={arch['d']}, nl={arch['nl']}, nh={arch['nh']}, "
          f"nkv={arch['nkv']}, dh={arch['dh']}, fi={arch['fi']}, V={arch['V']}")
    print(f"  Parameters: {params/1e9:.2f}B")

    # Free model GPU memory — we only need tokenizer + arch from here
    del gen.model
    del gen
    torch.cuda.empty_cache()
    gc.collect()

    # ── Collect results ───────────────────────────────────────────────────
    rows = []

    for ds_name in args.datasets:
        print(f"\n{'─'*70}")
        print(f"  Dataset: {ds_name}")
        print(f"{'─'*70}")

        dataset = load_ds(ds_name, args.samples, SEED)
        queries = [dataset[i]["question"] for i in range(len(dataset))]
        n = len(queries)

        # Avg query token length (for retrieval FLOP calc)
        avg_query_tokens = int(np.mean([len(tokenizer.encode(q)) for q in queries]))

        # ── RAG retrieval ─────────────────────────────────────────────────
        n_passages = 0
        if _RAG_DEPS:
            print("  Building RAG index...")
            rag_ret = RAGRetrieval()
            rag_ret.build_global_index(dataset)
            n_passages = rag_ret.n_passages
            passages_batch = rag_ret.retrieve(queries, RAG_TOP_K)
            rag_prompts = [_rag_prompt(q, p, args.model)
                           for q, p in zip(queries, passages_batch)]
            del rag_ret
        else:
            print("  [WARN] faiss/sentence-transformers not installed, skipping RAG")
            rag_prompts = [_rag_prompt(q, [], args.model) for q in queries]

        rag_tok_lens = [len(tokenizer.encode(p)) for p in rag_prompts]
        avg_rag_seq = int(np.mean(rag_tok_lens))

        # RAG FLOPs
        rag_retr = flops_rag_retrieval(n, avg_query_tokens, max(n_passages, 1))
        rag_pf   = flops_prefill(avg_rag_seq, arch, BATCH_SIZE)
        rag_dc   = flops_decode(avg_rag_seq, arch, args.max_new_tokens, BATCH_SIZE)
        rag_total = rag_retr["total"] + rag_pf["total"] + rag_dc["total"]

        rows.append({
            "dataset": ds_name,
            "config": "RAG",
            "hop": "-",
            "avg_prompt_tokens": avg_rag_seq,
            "retrieval_TFLOP": rag_retr["total"] / 1e12,
            "prefill_TFLOP": rag_pf["total"] / 1e12,
            "decode_TFLOP": rag_dc["total"] / 1e12,
            "total_TFLOP": rag_total / 1e12,
            "ratio_vs_rag": 1.0,
            # breakdown
            "retr_detail": rag_retr,
            "pf_detail": rag_pf,
            "dc_detail": rag_dc,
        })

        # ── GraphRAG retrieval at each hop ────────────────────────────────
        for hop in HOPS:
            print(f"  GraphRAG hop={hop} retrieval...")
            retriever = SimpleGraphRAGRetrieval(dataset)
            retr_results, _ = retriever.retrieve_batch(hop)

            # Count edges traversed for retrieval FLOP estimate
            total_edges = 0
            for i in range(len(dataset)):
                g_data = dataset[i].get("graph", [])
                total_edges += len(g_data)
            avg_edges = total_edges // max(n, 1)

            graphrag_prompts = [
                format_prompt(r["query"], r["paths"], args.model)
                for r in retr_results
            ]
            gr_tok_lens = [len(tokenizer.encode(p)) for p in graphrag_prompts]
            avg_gr_seq = int(np.mean(gr_tok_lens))

            gr_retr = flops_graphrag_retrieval(n, avg_edges * hop)
            gr_pf   = flops_prefill(avg_gr_seq, arch, BATCH_SIZE)
            gr_dc   = flops_decode(avg_gr_seq, arch, args.max_new_tokens, BATCH_SIZE)
            gr_total = gr_retr["total"] + gr_pf["total"] + gr_dc["total"]

            rows.append({
                "dataset": ds_name,
                "config": "GraphRAG",
                "hop": hop,
                "avg_prompt_tokens": avg_gr_seq,
                "retrieval_TFLOP": gr_retr["total"] / 1e12,
                "prefill_TFLOP": gr_pf["total"] / 1e12,
                "decode_TFLOP": gr_dc["total"] / 1e12,
                "total_TFLOP": gr_total / 1e12,
                "ratio_vs_rag": gr_total / rag_total if rag_total > 0 else 0,
                "retr_detail": gr_retr,
                "pf_detail": gr_pf,
                "dc_detail": gr_dc,
            })

    # ═══════════════════════════════════════════════════════════════════════
    # Print results
    # ═══════════════════════════════════════════════════════════════════════

    print(f"\n\n{'='*100}")
    print(f"  FLOP Comparison: RAG vs GraphRAG  |  Model: {args.model}")
    print(f"  Arch: d={arch['d']}, nl={arch['nl']}, nh={arch['nh']}, "
          f"nkv={arch['nkv']}, fi={arch['fi']}, V={arch['V']}")
    print(f"  Samples: {args.samples}  |  Decode tokens: {args.max_new_tokens}  |  Batch: {BATCH_SIZE}")
    print(f"{'='*100}")

    header = (f"{'Dataset':<16} {'Config':<10} {'Hop':>4} {'Tokens':>7} "
              f"{'Retrieval':>12} {'Prefill':>12} {'Decode':>12} {'Total':>12} {'vs RAG':>8}")
    print(header)
    print("─" * 100)

    prev_ds = None
    for r in rows:
        ds_short = r["dataset"].split("/")[-1] if "/" in r["dataset"] else r["dataset"]
        if prev_ds and prev_ds != r["dataset"]:
            print("─" * 100)
        prev_ds = r["dataset"]

        hop_str = str(r["hop"]) if r["hop"] != "-" else "  -"
        print(f"{ds_short:<16} {r['config']:<10} {hop_str:>4} {r['avg_prompt_tokens']:>7} "
              f"{r['retrieval_TFLOP']:>10.4f} T {r['prefill_TFLOP']:>10.2f} T "
              f"{r['decode_TFLOP']:>10.2f} T {r['total_TFLOP']:>10.2f} T {r['ratio_vs_rag']:>7.2f}x")

    print("─" * 100)
    print("  T = TeraFLOPs (1e12 FLOPs).  All values are per-query (batch=1).")

    # ── Detailed breakdown for each config ────────────────────────────────
    print(f"\n\n{'='*100}")
    print("  Detailed FLOP Breakdown (per query)")
    print(f"{'='*100}")

    for r in rows:
        ds_short = r["dataset"].split("/")[-1] if "/" in r["dataset"] else r["dataset"]
        hop_str = f"hop={r['hop']}" if r["hop"] != "-" else ""
        print(f"\n  {ds_short} / {r['config']} {hop_str}  "
              f"(avg {r['avg_prompt_tokens']} tokens)")
        print(f"  {'─'*60}")

        # Retrieval
        rd = r["retr_detail"]
        print(f"    Retrieval:          {rd['total']/1e9:>12.4f} GFLOP")
        for k, v in rd.items():
            if k != "total":
                print(f"      {k:<20s} {v/1e9:>12.4f} GFLOP")

        # Prefill
        pf = r["pf_detail"]
        print(f"    Prefill:            {pf['total']/1e12:>12.4f} TFLOP")
        # Group into attention vs FFN vs other
        attn_flops = sum(pf[k] for k in ["q_proj","k_proj","v_proj","attn_qk",
                                           "attn_softmax","attn_sv","o_proj"])
        ffn_flops = sum(pf[k] for k in ["ffn_gate","ffn_up","ffn_down","ffn_act"])
        other_flops = pf["rmsnorm"] + pf["final_norm"] + pf["lm_head"]
        print(f"      attention         {attn_flops/1e12:>12.4f} TFLOP  "
              f"(QKV proj: {(pf['q_proj']+pf['k_proj']+pf['v_proj'])/1e12:.4f}, "
              f"scores: {(pf['attn_qk']+pf['attn_sv'])/1e12:.4f}, "
              f"O proj: {pf['o_proj']/1e12:.4f})")
        print(f"      FFN (SwiGLU)      {ffn_flops/1e12:>12.4f} TFLOP")
        print(f"      other (norm+head) {other_flops/1e12:>12.4f} TFLOP")

        # Decode
        dc = r["dc_detail"]
        print(f"    Decode:             {dc['total']/1e12:>12.4f} TFLOP")
        print(f"      projections       {dc['projections']/1e12:>12.4f} TFLOP")
        print(f"      attention_scores  {dc['attention_scores']/1e12:>12.4f} TFLOP")
        print(f"      FFN               {dc['ffn']/1e12:>12.4f} TFLOP")
        print(f"      other             {dc['other']/1e12:>12.4f} TFLOP")

        print(f"    TOTAL:              {r['total_TFLOP']:>12.4f} TFLOP")

    # ── Per-dataset summary ───────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("  Summary: GraphRAG FLOP overhead vs RAG")
    print(f"{'='*70}")
    for ds_name in args.datasets:
        ds_rows = [r for r in rows if r["dataset"] == ds_name]
        rag_row = [r for r in ds_rows if r["config"] == "RAG"][0]
        ds_short = ds_name.split("/")[-1] if "/" in ds_name else ds_name
        print(f"\n  {ds_short}  (RAG baseline: {rag_row['total_TFLOP']:.4f} TFLOP, "
              f"avg {rag_row['avg_prompt_tokens']} tokens)")
        for r in ds_rows:
            if r["config"] == "GraphRAG":
                overhead = (r["ratio_vs_rag"] - 1) * 100
                tok_increase = r["avg_prompt_tokens"] - rag_row["avg_prompt_tokens"]
                pf_ratio = r["prefill_TFLOP"] / max(rag_row["prefill_TFLOP"], 1e-15)
                print(f"    Hop {r['hop']}: {r['total_TFLOP']:.4f} TFLOP "
                      f"({r['ratio_vs_rag']:.2f}x total, "
                      f"prefill {pf_ratio:.2f}x) "
                      f"  [+{tok_increase} tokens, {r['avg_prompt_tokens']} total]")


if __name__ == "__main__":
    main()
