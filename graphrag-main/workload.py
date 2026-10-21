#!/usr/bin/env python3
"""
workload.py  —  RAG vs GraphRAG Workload Characterization  (ISCA paper)
=======================================================================

FOUR CONFIGURATIONS
-------------------
  ┌──────────────┬──────────────────────────┬──────────────────────────┐
  │  Config      │  Retrieval               │  LLM decode KV cache     │
  ├──────────────┼──────────────────────────┼──────────────────────────┤
  │  rag         │  FAISS global index      │  OFF  (use_cache=False)  │
  ├──────────────┼──────────────────────────┼──────────────────────────┤
  │  rag_kv      │  FAISS global index      │  ON   (use_cache=True)   │
  ├──────────────┼──────────────────────────┼──────────────────────────┤
  │  graphrag    │  BFS path enumeration    │  OFF  (use_cache=False)  │
  ├──────────────┼──────────────────────────┼──────────────────────────┤
  │  graphrag_kv │  BFS path enumeration    │  ON   (use_cache=True)   │
  └──────────────┴──────────────────────────┴──────────────────────────┘

All four use:
  • the same model, dataset, seed, generation config, prompt template
  • no cross-query KV sharing, no PFKV, no PassageKVCache
  • the only KV cache present is the standard autoregressive decode KV
    cache built inside model.generate() (on/off per config)

FAIRNESS CONTROLS
-----------------
  Same information: RAG indexes ALL KG triplets across the dataset
    (global corpus).  GraphRAG uses the per-query local KG subgraph.
    Both access the same knowledge — RAG searches it, GraphRAG traverses it.

  Same prompt template: Alpaca format for both RAG and GraphRAG.

OFFLINE vs ONLINE
-----------------
  OFFLINE (one-time, not in query latency):
    RAG:  build global FAISS index, encode all passages (~seconds–minutes)
    Both: load LLM weights into GPU HBM
  ONLINE (measured per-batch):
    RAG:      encode query + FAISS search (~5–10 ms/batch)
    GraphRAG: BFS graph traversal (100–2000 ms/batch)
    Both:     tokenise, prefill, decode

METRICS
-------
  Latency:   Retrieval / TTFT (≈prefill) / Decode / E2E  →  mean/median/p95/p99
  Throughput: Prefill tok/s, Decode tok/s, Queries/s
  Decode BW:  (model_size + KV_bytes) / decode_time_per_token  [GB/s, analytical]
  HBM:        Peak active bytes, cumulative allocations, model footprint
  Analytic:   Per-component HBM traffic (attn / FFN / KV-cache) + FLOPs
  Roofline:   Arithmetic intensity vs A100 ridge point (161 FLOP/byte)
  Profiler:   Per-kernel CUDA timing, FLOP distribution, memory events

KV CACHE VERIFICATION (traceable to source)
-------------------------------------------
  *_kv configs:   PathFirstGraphRAGGeneration(use_kv_cache=True)
                  → generate_batch_standard()      [generation.py]
                  → model.generate(use_cache=True, ...)

  non-kv configs: PathFirstGraphRAGGeneration(use_kv_cache=False)
                  → generate_batch_standard()
                  → model.generate(use_cache=False, ...)

Usage
-----
  python workload.py --samples 32 --batch_size 8
  python workload.py --neurips                        # 64 samples, beam=4
  python workload.py --configs rag graphrag           # two configs only
  python workload.py --neurips --profile              # +torch profiler
  python workload.py --dataset metaqa-3hop --samples 64

  /home/am137688/anaconda3/envs/graphrag/bin/python workload.py [args]
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import traceback
import warnings
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*pynvml.*")

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import set_reproducible_seed
from generation import PathFirstGraphRAGGeneration, format_prompt
from graph_utils import build_graph, get_paths_bfs
from metrics import evaluate_metrics
from metaqa_adapter import load_metaqa
from retrieval import SimpleGraphRAGRetrieval

try:
    import pynvml                        # type: ignore
    pynvml.nvmlInit()
    _NVML = True
except Exception:
    _NVML = False

try:
    import faiss                         # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore
    _RAG_DEPS = True
    _RAG_ERR  = ""
except ImportError as _ie:
    _RAG_DEPS = False
    _RAG_ERR  = str(_ie)


# ═══════════════════════════════════════════════════════════════════════════
# §1  Hardware constants  (A100 PCIe 80 GB)
# ═══════════════════════════════════════════════════════════════════════════

A100_HBM_BW_GBS   = 1_935.0        # GB/s  HBM2e (PCIe variant)
A100_FP16_TFLOPS  = 312.0          # TFLOP/s  (Tensor Core peak)
A100_PCIE_BW_GBS  = 64.0           # GB/s  PCIe 4.0 ×16 unidirectional
A100_RIDGE        = (A100_FP16_TFLOPS * 1e12) / (A100_HBM_BW_GBS * 1e9)  # ~161 FLOP/byte


# ═══════════════════════════════════════════════════════════════════════════
# §2  RAGRetrieval  (DPR / ATLAS-style global FAISS index)
# ═══════════════════════════════════════════════════════════════════════════

class RAGRetrieval:
    """
    Dense-passage retrieval following DPR / ATLAS conventions.

    OFFLINE  build_global_index():
      Deduplicates ALL KG triplets from ALL dataset samples into a single
      global passage corpus.  Encodes once with sentence-transformers.
      Builds a FAISS IndexFlatIP (exact inner-product search, fp32).
      This is the one-time indexing cost; NOT counted in query latency.

    ONLINE   retrieve_batch():
      Encodes each query (~5–10 ms/batch) → FAISS search (~1 ms).
      Returns top_k passages per query from the shared global corpus.

    Why per-query FAISS is wrong:
      Building a per-query index rebuilds FAISS from scratch for each
      query's local KG, making RAG appear ~20× slower than GraphRAG —
      the opposite of production systems.  The global index amortises
      encoding across all queries (Karpukhin 2020, Izacard & Grave 2022).
    """

    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, embed_model: str = EMBED_MODEL):
        if not _RAG_DEPS:
            raise ImportError(f"sentence_transformers/faiss not found: {_RAG_ERR}")
        print(f"[RAG] Loading embedding model: {embed_model}")
        self._embedder  = SentenceTransformer(embed_model, device="cpu")
        self._dim       = self._embedder.get_sentence_embedding_dimension()
        self._index: Optional[faiss.IndexFlatIP] = None
        self._passages: List[str] = []
        self.build_time_s = 0.0

    def _encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return self._embedder.encode(
            texts, batch_size=512, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=len(texts) > 500,
        ).astype(np.float32)

    def build_global_index(self, dataset) -> None:
        t0 = time.perf_counter()
        seen, passages = set(), []
        for i in range(len(dataset)):
            for triplet in dataset[i].get("graph", []):
                key = tuple(str(x).strip() for x in triplet)
                if key not in seen:
                    seen.add(key)
                    passages.append(f"{key[0]} | {key[1]} | {key[2]}")

        print(f"[RAG] Encoding {len(passages):,} unique passages ...")
        embs = self._encode(passages)
        self._index    = faiss.IndexFlatIP(self._dim)
        self._index.add(embs)
        self._passages = passages
        self.build_time_s = time.perf_counter() - t0
        print(f"[RAG] Index built: {len(passages):,} passages  "
              f"in {self.build_time_s:.1f}s  (dim={self._dim})")

    def retrieve_batch(self, queries: List[str], top_k: int = 5
                       ) -> Tuple[List[Dict], float, float]:
        """
        Returns:
          results    [{query, passages:[str]}, ...]
          embed_s    wall time for query embedding
          search_s   wall time for FAISS search
        """
        assert self._index is not None, "Call build_global_index() first."
        k = min(top_k, len(self._passages))

        t_emb = time.perf_counter()
        qembs = self._encode(queries)
        embed_s = time.perf_counter() - t_emb

        t_srch = time.perf_counter()
        _, idxs = self._index.search(qembs, k)
        search_s = time.perf_counter() - t_srch

        results = []
        for qi, row in enumerate(idxs):
            top = [self._passages[i] for i in row if 0 <= i < len(self._passages)]
            results.append({"query": queries[qi], "passages": top})
        return results, embed_s, search_s

    @property
    def n_passages(self) -> int:
        return len(self._passages)

    def embed_model_MB(self) -> float:
        return sum(p.numel() * p.element_size()
                   for p in self._embedder.parameters()) / 1e6


# ═══════════════════════════════════════════════════════════════════════════
# §3  Prompt formatters
# ═══════════════════════════════════════════════════════════════════════════

def _rag_prompt(query: str, passages: List[str], model_name: str) -> str:
    """
    RAG prompt in the same Alpaca format used by format_prompt() for GraphRAG.
    Keeps generation conditions identical across all 4 configs.
    """
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


# ═══════════════════════════════════════════════════════════════════════════
# §4  Hardware utilities
# ═══════════════════════════════════════════════════════════════════════════

class CUDATimer:
    """GPU-accurate timing via CUDA Events."""
    def __init__(self, dev: int = 0):
        self.dev = dev
        self._s  = torch.cuda.Event(enable_timing=True)
        self._e  = torch.cuda.Event(enable_timing=True)
        self.elapsed_s: float = 0.0

    def __enter__(self):  self._s.record(); return self
    def __exit__(self, *_):
        self._e.record()
        torch.cuda.synchronize(self.dev)
        self.elapsed_s = self._s.elapsed_time(self._e) / 1e3


class HBMSnap:
    """Capture torch.cuda.memory_stats before/after a block."""
    def __init__(self, dev: int = 0):
        self.dev = dev
        self.alloc_delta_B: int = 0
        self.peak_B:        int = 0
        self._before:       Dict = {}

    def __enter__(self):
        torch.cuda.reset_peak_memory_stats(self.dev)
        self._before = dict(torch.cuda.memory_stats(self.dev))
        return self

    def __exit__(self, *_):
        after = dict(torch.cuda.memory_stats(self.dev))
        self.alloc_delta_B = (
            int(after.get("allocated_bytes.all.allocated", 0))
            - int(self._before.get("allocated_bytes.all.allocated", 0))
        )
        self.peak_B = int(after.get("active_bytes.all.peak", 0))


def nvml_snap(dev: int = 0) -> Dict:
    if not _NVML:
        return {}
    try:
        h   = pynvml.nvmlDeviceGetHandleByIndex(dev)
        ut  = pynvml.nvmlDeviceGetUtilizationRates(h)
        pw  = pynvml.nvmlDeviceGetPowerUsage(h)
        clk = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        return {"sm_%": ut.gpu, "mem_%": ut.memory,
                "power_W": pw / 1e3, "sm_MHz": clk,
                "used_GB": mem.used / 1e9}
    except Exception:
        return {}


class PowerSampler:
    """
    GPU utilisation + power sampler using ``nvidia-smi dmon`` subprocess.

    ``nvmlDeviceGetUtilizationRates`` has a coarse internal window that
    often misaligns with the polling cadence and returns 0%.  ``dmon``
    uses a different driver path and reliably reports SM%.  It also runs
    in its own process, so model.generate()'s GIL cannot starve it.

        with PowerSampler(dev=0) as ps:
            do_work()
        print(ps.mean_sm_pct, ps.mean_W)
    """

    def __init__(self, dev: int = 0, interval_s: float = 1):
        self.dev         = dev
        # nvidia-smi dmon -d accepts integer seconds (min 1)
        self._interval   = max(1, int(interval_s))
        self._proc: Optional[subprocess.Popen] = None
        self._t0         = 0.0
        self.energy_J    = 0.0
        self.mean_W      = 0.0
        self.peak_W      = 0.0
        self.duration_s  = 0.0
        self.mean_sm_pct = 0.0
        self.peak_sm_pct = 0.0

    def __enter__(self):
        try:
            self._proc = subprocess.Popen(
                ["nvidia-smi", "dmon",
                 "-i", str(self.dev),
                 "-s", "pu",           # power + utilisation
                 "-d", str(self._interval)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except FileNotFoundError:
            self._proc = None
        self._t0 = time.time()
        return self

    def __exit__(self, *_):
        self.duration_s = time.time() - self._t0
        if self._proc is None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        raw = self._proc.stdout.read()
        self._parse(raw)

    def _parse(self, raw: str):
        sm_vals: List[float] = []
        pw_vals: List[float] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # dmon -s pu columns: idx pwr gtemp mtemp sm mem enc dec ...
            try:
                pwr = float(parts[1])
                sm  = float(parts[4])
                pw_vals.append(pwr)
                sm_vals.append(sm)
            except (IndexError, ValueError):
                continue

        if not pw_vals:
            return
        self.mean_W      = float(np.mean(pw_vals))
        self.peak_W      = float(np.max(pw_vals))
        self.mean_sm_pct = float(np.mean(sm_vals))
        self.peak_sm_pct = float(np.max(sm_vals))
        if self.duration_s > 0:
            self.energy_J = self.mean_W * self.duration_s


def model_info(m: torch.nn.Module) -> Tuple[int, float, str]:
    params = sum(p.numel() for p in m.parameters())
    dt     = next(m.parameters()).dtype
    bpp    = torch.finfo(dt).bits // 8
    return params, params * bpp / 1e9, str(dt)


def get_arch(gen) -> Dict:
    cfg = gen.model.config
    d   = getattr(cfg, "hidden_size",         getattr(cfg, "d_model",  4096))
    nl  = getattr(cfg, "num_hidden_layers",   getattr(cfg, "n_layer",   32))
    nh  = getattr(cfg, "num_attention_heads", getattr(cfg, "n_head",    32))
    nkv = getattr(cfg, "num_key_value_heads", nh)
    dh  = d // nh
    fi  = getattr(cfg, "intermediate_size",   getattr(cfg, "ffn_dim", 4 * d))
    return {"d": d, "nl": nl, "nh": nh, "nkv": nkv, "dh": dh,
            "ffn_mult": max(1, fi // d)}


# ═══════════════════════════════════════════════════════════════════════════
# §5  Analytical models
# ═══════════════════════════════════════════════════════════════════════════

def analytic_hbm(seq: int, d: int, nl: int, nh: int, nkv: int, dh: int,
                 ffn_mult: int, B: int, use_kv: bool = True,
                 dtype_B: int = 2) -> Dict[str, float]:
    """
    Analytic HBM traffic estimate (bytes) for one forward pass.

    When use_kv=True (decode KV cache ON), decode steps only recompute
    Q and read all cached K,V; prefill still processes full sequence.
    When use_kv=False, every decode step recomputes the full context.

    We model the PREFILL pass here (dominant for characterization).
    """
    ffn = ffn_mult * d
    e   = dtype_B

    # Attention per layer
    attn_wt  = (3 * d * d + d * d) * e          # QKV + out projection weights
    attn_act = (3 * B * seq * d + B * nh * seq * seq + B * seq * d) * e
    kv_write = 2 * B * nkv * seq * dh * e       # write K,V to KV cache (if on)
    attn_lay = attn_wt + attn_act + (kv_write if use_kv else 0)

    # FFN per layer
    ffn_wt   = (2 * d * ffn + ffn * d) * e
    ffn_act  = (B * seq * d + 2 * B * seq * ffn) * e
    ffn_lay  = ffn_wt + ffn_act

    ta = nl * attn_lay
    tf = nl * ffn_lay
    kv = nl * (kv_write if use_kv else 0)
    return {"attn_GB":  ta / 1e9,
            "ffn_GB":   tf / 1e9,
            "kv_GB":    kv / 1e9,
            "total_GB": (ta + tf) / 1e9}


def analytic_flops(seq: int, d: int, nl: int, nh: int, dh: int,
                   ffn_mult: int, B: int) -> float:
    ffn   = ffn_mult * d
    qkv   = 2 * B * seq * d * d * 3
    score = 2 * B * nh * seq * seq * dh
    out   = 2 * B * seq * d * d
    ff    = 2 * B * seq * d * ffn * 3
    return float(nl * (qkv + score + out + ff))


def kv_cache_bytes(nl: int, nkv: int, dh: int, seq: int,
                   B: int, dtype_B: int = 2) -> float:
    """HBM occupied by the decode KV cache (K + V per layer)."""
    return 2 * nl * nkv * dh * seq * B * dtype_B


def decode_bw_GBs(model_GB: float, nl: int, nkv: int, dh: int,
                  prompt_len: int, B: int, decode_s_per_tok: float,
                  use_kv: bool = True, dtype_B: int = 2) -> float:
    """
    Effective HBM bandwidth during one decode step (GB/s).

    Model weights are loaded ONCE per step and broadcast across the batch —
    weight bytes do NOT scale with B.  Only the KV cache scales with B.

    KV=ON:  bytes = model_weights + KV_cache(nl, nkv, dh, seq, B)
            → memory-BW-bound; should approach A100 HBM peak (~1935 GB/s)
    KV=OFF: bytes = model_weights only (no KV cache; compute cost also grows
            with context length, so this is a lower-bound proxy)
    """
    if decode_s_per_tok <= 0:
        return 0.0
    if use_kv:
        kv_b = kv_cache_bytes(nl, nkv, dh, prompt_len, B, dtype_B)
        bytes_per_step = model_GB * 1e9 + kv_b     # weights once + full KV cache
    else:
        bytes_per_step = model_GB * 1e9             # weights once, no KV cache
    return bytes_per_step / decode_s_per_tok / 1e9


def roofline(flops: float, hbm_bytes: float) -> Dict:
    if hbm_bytes <= 0:
        return {"arith_intensity": float("inf"), "memory_bound": False,
                "attainable_TFLOPs": A100_FP16_TFLOPS}
    ai = flops / hbm_bytes
    return {
        "arith_intensity":  ai,
        "memory_bound":    ai < A100_RIDGE,
        "ridge_FLOP_byte": A100_RIDGE,
        "attainable_TFLOPs": min(A100_FP16_TFLOPS, ai * A100_HBM_BW_GBS / 1e3),
    }


# ═══════════════════════════════════════════════════════════════════════════
# §6  Profiler context + kernel analysis
# ═══════════════════════════════════════════════════════════════════════════

@contextmanager
def profiler_ctx(out_dir: str, tag: str, active: int = 2, enabled: bool = True):
    if not enabled:
        yield None, lambda: None
        return
    os.makedirs(out_dir, exist_ok=True)
    trace = os.path.join(out_dir, f"{tag}_trace.json")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=active),
        record_shapes=True, profile_memory=True, with_flops=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(out_dir),
    ) as prof:
        try:
            yield prof, prof.step
        finally:
            prof.export_chrome_trace(trace)
            print(f"  [Profiler] Chrome trace → {trace}")


def summarise_profiler(prof, label: str, top_n: int = 12) -> Dict:
    """Extract per-kernel stats from a torch.profiler result."""
    avgs = prof.key_averages(group_by_input_shape=False)

    by_cuda = sorted(avgs, key=lambda x: x.cuda_time_total, reverse=True)

    total_cuda_us = sum(x.cuda_time_total for x in avgs)
    total_flops   = sum(x.flops           for x in avgs if hasattr(x, "flops"))
    total_mem     = sum(getattr(x, "self_device_memory_usage", 0) for x in avgs)

    kernels = []
    for op in by_cuda[:top_n]:
        kernels.append({
            "kernel":    op.key,
            "cuda_ms":   round(op.cuda_time_total / 1e3, 3),
            "pct_total": round(100 * op.cuda_time_total / max(1, total_cuda_us), 1),
            "flops":     int(getattr(op, "flops", 0)),
            "mem_MB":    round(getattr(op, "self_device_memory_usage", 0) / 1e6, 3),
            "calls":     op.count,
        })

    attn_us = sum(x.cuda_time_total for x in avgs
                  if any(k in x.key.lower()
                         for k in ("attention", "flash_attn", "scaled_dot")))
    mm_us   = sum(x.cuda_time_total for x in avgs
                  if any(k in x.key.lower() for k in ("mm", "gemm", "matmul")))
    elem_us = sum(x.cuda_time_total for x in avgs
                  if any(k in x.key.lower() for k in ("add", "mul", "silu", "gelu", "norm")))
    mem_us  = sum(x.cuda_time_total for x in avgs
                  if any(k in x.key.lower() for k in ("copy", "cat", "to", "fill")))

    print(f"\n  ── Profiler [{label}] ──────────────────────────────────────────")
    print(f"  Total CUDA time: {total_cuda_us/1e6:.1f} ms   "
          f"FLOPs: {total_flops/1e9:.1f} GFLOP   "
          f"Mem events: {total_mem/1e6:.1f} MB")
    print(f"  {'Kernel':<44} {'ms':>8} {'%':>6} {'GFLOP':>8} {'calls':>6}")
    print(f"  {'─'*72}")
    for k in kernels:
        print(f"  {k['kernel']:<44} {k['cuda_ms']:>8.2f} {k['pct_total']:>5.1f}% "
              f"{k['flops']/1e9:>8.2f} {k['calls']:>6}")
    print(f"  {'─'*72}")
    if total_cuda_us:
        print(f"  Attn: {100*attn_us/total_cuda_us:.1f}%   "
              f"GEMM: {100*mm_us/total_cuda_us:.1f}%   "
              f"Elemwise: {100*elem_us/total_cuda_us:.1f}%   "
              f"MemOps: {100*mem_us/total_cuda_us:.1f}%")

    return {
        "label":          label,
        "total_cuda_ms":  total_cuda_us / 1e3,
        "total_GFLOP":    total_flops / 1e9,
        "total_mem_MB":   total_mem / 1e6,
        "categories_%": {
            "attention":  round(100 * attn_us / max(1, total_cuda_us), 1),
            "gemm":       round(100 * mm_us   / max(1, total_cuda_us), 1),
            "elemwise":   round(100 * elem_us / max(1, total_cuda_us), 1),
            "memory_ops": round(100 * mem_us  / max(1, total_cuda_us), 1),
        },
        "top_kernels": kernels,
    }


# ═══════════════════════════════════════════════════════════════════════════
# §7  Dataset loading
# ═══════════════════════════════════════════════════════════════════════════

def load_ds(name: str, split: str, n: int, seed: int):
    short = {"metaqa-1hop": 1, "metaqa-2hop": 2, "metaqa-3hop": 3}
    if name in short:
        hops = short[name]
        kb   = SCRIPT_DIR / "metaQA" / "kb.txt"
        if kb.exists():
            ds = load_metaqa(hops=hops, split=split,
                             data_dir=str(SCRIPT_DIR / "metaQA"))
        elif hops == 3:
            ds = load_dataset("mhivanov123/metaqa-3hop",
                              split="validation" if split == "dev" else split)
        else:
            raise FileNotFoundError("MetaQA local files not found.")
    else:
        ds = load_dataset(name, split=split)
    if len(ds) > n:
        ds = ds.shuffle(seed=seed).select(range(n))
    return ds


# ═══════════════════════════════════════════════════════════════════════════
# §8  Results dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RunResult:
    config:         str
    model_name:     str
    dataset:        str
    n_samples:      int
    batch_size:     int
    num_beams:      int
    max_new_tokens: int
    seed:           int
    use_kv:         bool    # standard decode KV cache on/off

    # offline one-time
    model_load_s:   float = 0.0
    index_build_s:  float = 0.0
    n_indexed:      int   = 0
    embed_model_MB: float = 0.0

    # per-batch arrays
    retrieval_s:    List[float] = field(default_factory=list)
    embed_query_s:  List[float] = field(default_factory=list)  # RAG: query encode time
    ttft_s:         List[float] = field(default_factory=list)  # ≈ prefill
    decode_s:       List[float] = field(default_factory=list)
    gen_s:          List[float] = field(default_factory=list)  # ttft + decode
    gpu_gen_s:      List[float] = field(default_factory=list)  # CUDA-event
    e2e_s:          List[float] = field(default_factory=list)

    input_tok:      List[int]   = field(default_factory=list)
    output_tok:     List[int]   = field(default_factory=list)
    ctx_tok:        List[int]   = field(default_factory=list)
    n_retrieved:    List[int]   = field(default_factory=list)

    hbm_peak_B:     List[int]   = field(default_factory=list)
    hbm_alloc_B:    List[int]   = field(default_factory=list)

    analytic_total_GB: List[float] = field(default_factory=list)
    analytic_attn_GB:  List[float] = field(default_factory=list)
    analytic_ffn_GB:   List[float] = field(default_factory=list)
    analytic_kv_GB:    List[float] = field(default_factory=list)
    analytic_flops:    List[float] = field(default_factory=list)

    sm_pct:         List[float] = field(default_factory=list)
    power_W:        List[float] = field(default_factory=list)

    # GraphRAG-specific path statistics (graphrag / graphrag_kv only)
    # Per-query, collected across all batches
    paths_per_query:     List[int]   = field(default_factory=list)  # #paths found
    tokens_per_path:     List[float] = field(default_factory=list)  # avg tok per path
    path_prefix_sharing: List[float] = field(default_factory=list)  # shared-prefix ratio
    graph_nodes:         List[int]   = field(default_factory=list)  # nodes in subgraph
    graph_edges:         List[int]   = field(default_factory=list)  # edges in subgraph

    # Counterfactual latency (if retrieval cost were zero)
    gen_only_e2e_s:     List[float] = field(default_factory=list)

    hits_at_1:      float = 0.0
    f1:             float = 0.0

    model_params:   int   = 0
    model_size_GB:  float = 0.0
    model_dtype:    str   = ""
    arch:           Dict  = field(default_factory=dict)
    gpu_name:       str   = ""
    gpu_mem_GB:     float = 0.0

    profiler_summary: Dict = field(default_factory=dict)
    trace_path:       str  = ""


# ═══════════════════════════════════════════════════════════════════════════
# §9  Path statistics (GraphRAG only)
# ═══════════════════════════════════════════════════════════════════════════

def _path_token_len(path, tokenizer) -> int:
    """Tokenize a single path (list of strings or already a string)."""
    if isinstance(path, str):
        s = path
    elif isinstance(path, (list, tuple)):
        s = " -> ".join(str(e) for e in path)
    else:
        s = str(path)
    return len(tokenizer.encode(s, add_special_tokens=False))


def _shared_prefix_ratio(paths: List) -> float:
    """
    Fraction of (path, other_path) pairs that share at least the first node.
    Indicates sub-path reuse potential — motivates SubgraphKVCache.
    Returns 0 if < 2 paths.
    """
    if len(paths) < 2:
        return 0.0
    def first_node(p):
        if isinstance(p, (list, tuple)) and len(p) > 0:
            return str(p[0])
        if isinstance(p, str):
            parts = p.split("->")
            return parts[0].strip() if parts else ""
        return str(p)

    first_nodes = [first_node(p) for p in paths]
    pairs = len(paths) * (len(paths) - 1) / 2
    shared = sum(
        1 for i in range(len(first_nodes))
          for j in range(i + 1, len(first_nodes))
          if first_nodes[i] == first_nodes[j]
    )
    return shared / pairs if pairs > 0 else 0.0


def _collect_path_stats(retr_result, tokenizer, result: "RunResult") -> None:
    """
    For each query in a GraphRAG batch, record:
      - number of paths found
      - average tokens per path
      - shared-prefix ratio (sub-path reuse potential)
    """
    for r in retr_result:
        paths = r.get("paths", [])
        n_paths = len(paths)
        result.paths_per_query.append(n_paths)

        if n_paths > 0:
            tok_lens = [_path_token_len(p, tokenizer) for p in paths]
            result.tokens_per_path.append(float(np.mean(tok_lens)))
            result.path_prefix_sharing.append(_shared_prefix_ratio(paths))
        else:
            result.tokens_per_path.append(0.0)
            result.path_prefix_sharing.append(0.0)


def _collect_graph_stats(bds, result: "RunResult") -> None:
    """Record per-query subgraph size (nodes, edges)."""
    for i in range(len(bds)):
        g = bds[i].get("graph", [])
        edges  = len(g)
        nodes  = len({str(t[0]) for t in g} | {str(t[2]) for t in g})
        result.graph_nodes.append(nodes)
        result.graph_edges.append(edges)


# ═══════════════════════════════════════════════════════════════════════════
# §10  Core runner
# ═══════════════════════════════════════════════════════════════════════════

_LABELS = {
    "rag":         "RAG (KV=OFF)",
    "rag_kv":      "RAG (KV=ON)",
    "graphrag":    "GraphRAG (KV=OFF)",
    "graphrag_kv": "GraphRAG (KV=ON)",
}


def run_config(
    config:         str,
    dataset,
    model_name:     str,
    batch_size:     int,
    num_beams:      int,
    max_new_tokens: int,
    seed:           int,
    hop:            int,
    top_k:          int,
    rag_retriever:  Optional[RAGRetrieval],
    device:         int,
    profile:        bool,
    prof_dir:       str,
) -> RunResult:
    """Execute one complete workload for `config` and collect all metrics."""

    is_rag   = config.startswith("rag")
    use_kv   = config.endswith("_kv")    # standard decode KV cache on/off
    label    = _LABELS[config]

    set_reproducible_seed(seed)
    result = RunResult(
        config=config, model_name=model_name,
        dataset=getattr(dataset, "info", None) and dataset.info.description or str(dataset),
        n_samples=len(dataset), batch_size=batch_size,
        num_beams=num_beams, max_new_tokens=max_new_tokens, seed=seed,
        use_kv=use_kv,
    )

    # GPU info
    if torch.cuda.is_available():
        result.gpu_name   = torch.cuda.get_device_name(device)
        result.gpu_mem_GB = torch.cuda.get_device_properties(device).total_memory / 1e9

    # ── OFFLINE: load model ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"[{label}] Loading model: {model_name}")
    print(f"  use_kv_cache={use_kv}  is_rag={is_rag}")

    t0  = time.perf_counter()
    gen = PathFirstGraphRAGGeneration(
        model_name,
        device         = f"cuda:{device}",
        use_kv_cache   = use_kv,    # standard decode KV cache: on or off
        use_pfkv       = False,      # NO cross-query PFKV caching
        device_map     = None,       # single GPU — no model parallelism
        max_new_tokens = max_new_tokens,
        num_beams      = num_beams,
    )
    result.model_load_s = time.perf_counter() - t0

    # Verify KV cache flags propagated correctly to model.generate()
    assert gen.use_kv_cache == use_kv, \
        f"BUG: use_kv_cache mismatch: expected {use_kv}, got {gen.use_kv_cache}"
    assert gen.use_pfkv == False, "BUG: use_pfkv should be False for all configs"

    params, size_gb, dtype_s = model_info(gen.model)
    result.model_params  = params
    result.model_size_GB = size_gb
    result.model_dtype   = dtype_s
    result.arch          = get_arch(gen)
    arch = result.arch

    print(f"  Model: {params/1e9:.2f}B params, {size_gb:.2f} GB ({dtype_s})")
    print(f"  Arch:  d={arch['d']}, nl={arch['nl']}, nh={arch['nh']}, "
          f"nkv={arch['nkv']}, dh={arch['dh']}, ffn_mult={arch['ffn_mult']}")
    print(f"  Model loaded in {result.model_load_s:.1f}s")
    print(f"  [VERIFIED] decode KV cache = {use_kv}  PFKV = False")

    # ── OFFLINE: RAG index (already built by caller) ──────────────────────
    if is_rag:
        assert rag_retriever is not None
        result.index_build_s  = rag_retriever.build_time_s
        result.n_indexed      = rag_retriever.n_passages
        result.embed_model_MB = rag_retriever.embed_model_MB()
        print(f"[{label}] RAG index: {rag_retriever.n_passages:,} passages  "
              f"(built in {rag_retriever.build_time_s:.1f}s offline)")

    total  = len(dataset)
    nbatch = (total + batch_size - 1) // batch_size

    # ── Warmup ────────────────────────────────────────────────────────────
    print(f"[{label}] Warming up (1 batch) ...")
    wbatch   = dataset.select(range(min(batch_size, total)))
    wqueries = [wbatch[i]["question"] for i in range(len(wbatch))]

    if is_rag:
        wr, _, _ = rag_retriever.retrieve_batch(wqueries, top_k)
        wp = [_rag_prompt(r["query"], r["passages"], model_name) for r in wr]
    else:
        wret = SimpleGraphRAGRetrieval(wbatch)
        wrr, _ = wret.retrieve_batch(hop)
        wp = [format_prompt(r["query"], r["paths"], model_name) for r in wrr]

    _ = gen.generate_batch(wp, None)
    torch.cuda.synchronize(device)
    print(f"[{label}] Warmup done. Measuring {total} samples ...")

    prof_dir_c = os.path.join(prof_dir, config)
    os.makedirs(prof_dir_c, exist_ok=True)

    total_hits = total_f1 = 0.0
    total_proc = 0
    _profiler_summary: Dict = {}

    with profiler_ctx(prof_dir_c, config, active=min(2, nbatch),
                      enabled=profile) as (prof, pstep):
        for bi in tqdm(range(nbatch), desc=f"[{label}]"):
            si = bi * batch_size
            ei = min(si + batch_size, total)
            bds     = dataset.select(range(si, ei))
            blist   = [bds[i] for i in range(len(bds))]
            queries = [s["question"] for s in blist]
            answers = [s["answer"]   for s in blist]

            e2e_t = time.perf_counter()

            # ══ Stage 1: Retrieval ════════════════════════════════════════
            if is_rag:
                retr_res, emb_s, srch_s = rag_retriever.retrieve_batch(queries, top_k)
                retr_s = emb_s + srch_s
                result.embed_query_s.append(emb_s)
                prompts  = [_rag_prompt(r["query"], r["passages"], model_name)
                            for r in retr_res]
                n_items  = [len(r["passages"]) for r in retr_res]
            else:
                t_retr = time.perf_counter()
                retriever    = SimpleGraphRAGRetrieval(bds)
                retr_result, _ = retriever.retrieve_batch(hop)
                retr_s       = time.perf_counter() - t_retr
                prompts      = [format_prompt(r["query"], r["paths"], model_name)
                                for r in retr_result]
                n_items      = [len(r["paths"]) for r in retr_result]

                # ── GraphRAG path & graph statistics ──────────────────────
                _collect_path_stats(retr_result, gen.tokenizer, result)
                _collect_graph_stats(bds, result)

            result.retrieval_s.append(retr_s)
            result.n_retrieved.extend(n_items)

            # ══ Stage 2: Token counts ════════════════════════════════════
            toks     = [gen.tokenizer.encode(p) for p in prompts]
            plen     = [len(t) for t in toks]
            base_p   = [_rag_prompt(q, [], model_name) if is_rag
                        else format_prompt(q, [], model_name) for q in queries]
            base_len = [len(gen.tokenizer.encode(p)) for p in base_p]
            ctx_len  = [max(0, pl - bl) for pl, bl in zip(plen, base_len)]

            result.input_tok.extend(plen)
            result.ctx_tok.extend(ctx_len)

            # ══ Stage 3: LLM generation (with continuous NVML sampling) ════
            with PowerSampler(device) as ps_gen:
                with HBMSnap(device) as hbm:
                    with CUDATimer(device) as gpu_t:
                        gen_t0 = time.perf_counter()
                        # All configs: standard batched generation
                        # use_cache is already set in gen.use_kv_cache
                        preds    = gen.generate_batch(prompts, None)
                        gen_wall = time.perf_counter() - gen_t0

            result.gen_s.append(gen_wall)
            result.gpu_gen_s.append(gpu_t.elapsed_s)
            result.hbm_peak_B.append(hbm.peak_B)
            result.hbm_alloc_B.append(hbm.alloc_delta_B)

            # TTFT and decode time
            ttft = gen.ttfts[-1] if gen.ttfts else 0.0
            dec  = max(0.0, gen_wall - ttft)
            result.ttft_s.append(ttft)
            result.decode_s.append(dec)

            out_lens = [len(gen.tokenizer.encode(p)) for p in preds]
            result.output_tok.extend(out_lens)

            # ══ Analytic estimates ════════════════════════════════════════
            avg_seq = int(np.mean(plen))
            avg_B   = len(prompts)
            a = analytic_hbm(avg_seq, arch["d"], arch["nl"], arch["nh"],
                             arch["nkv"], arch["dh"], arch["ffn_mult"],
                             avg_B, use_kv=use_kv)
            f = analytic_flops(avg_seq, arch["d"], arch["nl"], arch["nh"],
                               arch["dh"], arch["ffn_mult"], avg_B)
            result.analytic_total_GB.append(a["total_GB"])
            result.analytic_attn_GB.append(a["attn_GB"])
            result.analytic_ffn_GB.append(a["ffn_GB"])
            result.analytic_kv_GB.append(a["kv_GB"])
            result.analytic_flops.append(f)

            # ══ NVML (continuous sampling during generation) ════════════
            result.sm_pct.append(ps_gen.mean_sm_pct)
            result.power_W.append(ps_gen.mean_W)

            # ══ E2E ═══════════════════════════════════════════════════════
            e2e_s = time.perf_counter() - e2e_t
            result.e2e_s.append(e2e_s)

            # Counterfactual: what if retrieval were instant?
            result.gen_only_e2e_s.append(gen_wall)

            # ══ Quality ═══════════════════════════════════════════════════
            m = evaluate_metrics(preds, answers)
            n = len(preds)
            total_hits += m["hits_at_1"] * n
            total_f1   += m["f1"] * n
            total_proc += n

            # Per-batch decode bandwidth
            dec_tok = max(1, sum(out_lens) - avg_B)
            dbw = decode_bw_GBs(size_gb, arch["nl"], arch["nkv"], arch["dh"],
                                 avg_seq, avg_B,
                                 dec / dec_tok if dec_tok and dec > 0 else 1e-9,
                                 use_kv=use_kv)

            print(f"  b{bi+1}/{nbatch}  "
                  f"retr={retr_s*1e3:.0f}ms  "
                  f"ttft={ttft*1e3:.0f}ms  "
                  f"dec={dec*1e3:.0f}ms  "
                  f"e2e={e2e_s*1e3:.0f}ms  "
                  f"hbm_peak={hbm.peak_B/1e9:.2f}GB  "
                  f"dec_BW={dbw:.0f}GB/s  "
                  f"SM={ps_gen.mean_sm_pct:.0f}%  "
                  f"H@1={m['hits_at_1']:.3f}")

            if pstep:
                pstep()

    # Profiler summary
    if profile and prof is not None:
        try:
            _profiler_summary = summarise_profiler(prof, label)
            result.profiler_summary = _profiler_summary
            result.trace_path = os.path.join(prof_dir_c, f"{config}_trace.json")
        except Exception as e:
            print(f"  [Profiler] Summary failed: {e}")

    nn = total_proc or 1
    result.hits_at_1 = total_hits / nn
    result.f1        = total_f1   / nn

    return result


# ═══════════════════════════════════════════════════════════════════════════
# §10  Stats helpers
# ═══════════════════════════════════════════════════════════════════════════

def _a(lst) -> np.ndarray:
    a = np.array(lst, dtype=float)
    return a if a.size else np.array([0.0])

def pct(lst, unit=1e3) -> Dict:
    a = _a(lst) * unit
    if a.size == 0:
        return {}
    return {"mean": float(np.mean(a)),   "med": float(np.median(a)),
            "p95":  float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
            "std":  float(np.std(a)),
            "min":  float(np.min(a)),    "max": float(np.max(a))}

def safe_mean(lst, scale=1.0) -> Optional[float]:
    if not lst:
        return None
    return float(np.mean(lst)) * scale

def _decode_bw_list(r: RunResult) -> List[float]:
    out = []
    B   = r.batch_size
    for i in range(len(r.decode_s)):
        oi  = r.output_tok[i * B: (i + 1) * B]
        ii  = r.input_tok[ i * B: (i + 1) * B]
        dec_tok = max(1, sum(oi) - B)
        avg_inp = int(np.mean(ii)) if ii else 512
        dbw = decode_bw_GBs(r.model_size_GB, r.arch["nl"], r.arch["nkv"],
                             r.arch["dh"], avg_inp, B,
                             r.decode_s[i] / dec_tok if dec_tok and r.decode_s[i] > 0 else 1e-9,
                             use_kv=r.use_kv)
        out.append(dbw)
    return out

def _gen_tps_list(r: RunResult) -> List[float]:
    out = []
    B = r.batch_size
    for i in range(len(r.gen_s)):
        ot = sum(r.output_tok[i*B:(i+1)*B]) if r.output_tok else 0
        it = sum(r.input_tok[ i*B:(i+1)*B]) if r.input_tok  else 0
        t  = r.gen_s[i]
        out.append((ot + it) / t if t > 0 else 0.0)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# §11  Reporting
# ═══════════════════════════════════════════════════════════════════════════

def _row(name, vals, unit="", w=16, fmt=".1f"):
    s = f"  {name:<38}"
    for v in vals:
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            s += " " * w
        else:
            s += format(v, f">{w}{fmt}")
    if unit:
        s += f"  {unit}"
    print(s)

_SEP = "─" * 84

def print_report(results: Dict[str, RunResult]) -> None:
    configs = list(results.keys())
    rs      = [results[c] for c in configs]
    labels  = [_LABELS.get(c, c) for c in configs]
    ref     = rs[0]

    print(f"\n{'═'*84}")
    print(f"  WORKLOAD CHARACTERIZATION  —  RAG vs GraphRAG  (ISCA)")
    print(f"  Model:   {ref.model_name}")
    print(f"  Dataset: {ref.dataset}   N={ref.n_samples}  batch={ref.batch_size}")
    print(f"  Config:  beams={ref.num_beams}  max_new_tokens={ref.max_new_tokens}")
    print(f"  GPU:     {ref.gpu_name}  ({ref.gpu_mem_GB:.0f} GB)")
    print(f"  A100 ridge: {A100_RIDGE:.0f} FLOP/byte  "
          f"(HBM {A100_HBM_BW_GBS:.0f} GB/s, FP16 {A100_FP16_TFLOPS:.0f} TFLOP/s)")
    print(f"{'═'*84}")

    def hdr():
        h = f"  {'Metric':<38}" + "".join(f"{l:>{16}}" for l in labels)
        print(f"\n{_SEP}\n{h}\n{_SEP}")

    # ── Quality ──────────────────────────────────────────────────────────
    print("\n  ANSWER QUALITY")
    hdr()
    _row("Hits@1  (primary)",   [r.hits_at_1 for r in rs], fmt=".4f")
    _row("F1  (token overlap)", [r.f1        for r in rs], fmt=".4f")

    # ── Configuration summary ─────────────────────────────────────────────
    print("\n  CONFIGURATION SUMMARY")
    hdr()
    for r in rs:
        lbl = _LABELS.get(r.config, r.config)
        print(f"  {lbl:<38}  retrieval={'FAISS' if r.config.startswith('rag') else 'BFS'}  "
              f"decode_kv={'ON' if r.use_kv else 'OFF'}")

    # ── Offline costs ─────────────────────────────────────────────────────
    print("\n  OFFLINE ONE-TIME COSTS  (amortised, not in query latency)")
    hdr()
    _row("Model load  (s)",       [r.model_load_s for r in rs])
    _row("RAG index build  (s)",  [r.index_build_s if r.index_build_s > 0 else None
                                   for r in rs])
    _row("RAG passages indexed",  [r.n_indexed if r.n_indexed else None
                                   for r in rs], fmt=",")
    _row("Embed model  (MB)",     [r.embed_model_MB if r.embed_model_MB else None
                                   for r in rs], fmt=".0f")

    # ── Retrieval config ──────────────────────────────────────────────────
    print("\n  RETRIEVAL  (per-query averages)")
    hdr()
    _row("Avg retrieved items",   [safe_mean(r.n_retrieved) for r in rs])
    _row("Avg prompt tokens",     [safe_mean(r.input_tok)   for r in rs], fmt=".0f")
    _row("Avg context tokens",    [safe_mean(r.ctx_tok)     for r in rs], fmt=".0f")
    _row("Context / prompt  (%)",
         [100 * safe_mean(r.ctx_tok) / max(1, safe_mean(r.input_tok) or 1) for r in rs])

    # ── Latency ─────────────────────────────────────────────────────────
    print("\n  LATENCY  (ms per batch)")
    hdr()
    for stage, sname in [
        ("retrieval_s",   "Retrieval"),
        ("embed_query_s", "  ↳ Query embed  (RAG online)"),
        ("ttft_s",        "TTFT / Prefill"),
        ("decode_s",      "Decode"),
        ("gen_s",         "Generation total"),
        ("e2e_s",         "E2E"),
    ]:
        for stat, sfx in [("mean", " mean"), ("med", " median"),
                          ("p95",  " p95"),  ("p99", " p99")]:
            vals = []
            for r in rs:
                lst = getattr(r, stage)
                if lst:
                    vals.append(pct(lst)[stat])
                else:
                    vals.append(None)
            if any(v is not None and v > 0.05 for v in vals):
                _row(sname + sfx, vals, "ms")

    # ── Throughput ───────────────────────────────────────────────────────
    print("\n  THROUGHPUT")
    hdr()
    _row("Generation tok/s  (mean)",
         [safe_mean(_gen_tps_list(r)) for r in rs], "tok/s", fmt=".0f")
    _row("Queries/s",
         [r.n_samples / sum(r.e2e_s) if r.e2e_s else 0 for r in rs],
         "q/s", fmt=".2f")

    # ── Decode bandwidth ─────────────────────────────────────────────────
    print("\n  DECODE BANDWIDTH  (analytical estimate)")
    print(f"  BW = (model_weights [+ KV_cache if KV=ON]) / step_time")
    print(f"  A100 HBM peak = {A100_HBM_BW_GBS:.0f} GB/s")
    print(f"  NOTE: >100% indicates L2 cache reuse — not all weights read from HBM each step.")
    hdr()
    dbws = [_decode_bw_list(r) for r in rs]
    dbw_means = [safe_mean(d) for d in dbws]
    _row("Decode BW  mean  (GB/s)", dbw_means, "GB/s", fmt=".0f")
    pct_vals = [100 * (v or 0) / A100_HBM_BW_GBS for v in dbw_means]
    _row("% of A100 HBM peak",     pct_vals, "%")
    # Flag configs where analytical BW exceeds physical peak
    over = [i for i, v in enumerate(pct_vals) if v > 100]
    if over:
        names = [_LABELS.get(rs[i].config, rs[i].config) for i in over]
        print(f"  ⚠  {', '.join(names)}: analytical BW > HBM peak → L2 cache serves "
              f"part of the working set; true HBM BW is lower.")

    # ── GPU Memory ───────────────────────────────────────────────────────
    print("\n  GPU MEMORY")
    hdr()
    _row("Model footprint  (GB)",  [r.model_size_GB for r in rs], fmt=".3f")
    _row("Peak active HBM  (GB)",  [safe_mean(r.hbm_peak_B, 1/1e9) for r in rs],
         "GB", fmt=".3f")
    _row("Cumul alloc HBM  (GB)",  [safe_mean(r.hbm_alloc_B, 1/1e9) for r in rs],
         "GB", fmt=".3f")
    _row("KV cache size  (est. GB)",
         [kv_cache_bytes(r.arch["nl"], r.arch["nkv"], r.arch["dh"],
                         int(safe_mean(r.input_tok) or 0),
                         r.batch_size) / 1e9 if r.use_kv else None
          for r in rs], "GB", fmt=".3f")

    # ── Analytic HBM breakdown ───────────────────────────────────────────
    print("\n  ESTIMATED HBM TRAFFIC  (analytical, per forward pass)")
    print(f"  KV=ON configs include KV write cost in attention traffic.")
    hdr()
    for comp, attr in [("Attention  (GB)", "analytic_attn_GB"),
                       ("FFN        (GB)", "analytic_ffn_GB"),
                       ("KV write   (GB)", "analytic_kv_GB"),
                       ("TOTAL      (GB)", "analytic_total_GB")]:
        _row(comp, [safe_mean(getattr(r, attr)) for r in rs], "GB", fmt=".3f")

    # ── Roofline ─────────────────────────────────────────────────────────
    print("\n  ROOFLINE  (arithmetic intensity)")
    print(f"  Ridge = {A100_RIDGE:.0f} FLOP/byte  "
          f"(below ridge = memory-bound; above = compute-bound)")
    hdr()
    rls = [roofline(
        safe_mean(r.analytic_flops) or 0,
        (safe_mean(r.analytic_total_GB) or 1) * 1e9
    ) for r in rs]
    _row("Arith. intensity  (FLOP/byte)",
         [rl["arith_intensity"] for rl in rls], "F/B", fmt=".0f")
    print(f"  {'Bound by':<38}" +
          "".join(f"{'memory':>16}" if rl["memory_bound"] else f"{'compute':>16}"
                  for rl in rls))
    _row("Attainable TFLOP/s",
         [rl.get("attainable_TFLOPs", 0) for rl in rls], "TFLOP/s", fmt=".0f")

    # ── NVML ─────────────────────────────────────────────────────────────
    if any(r.sm_pct for r in rs):
        print("\n  GPU UTILISATION  (NVML, continuously sampled during generation @ ~50 ms)")
        hdr()
        _row("SM utilisation mean  (%)", [safe_mean(r.sm_pct)  for r in rs], "%")
        _row("Power mean  (W)",          [safe_mean(r.power_W) for r in rs], "W", fmt=".0f")

    # ── Time fraction ─────────────────────────────────────────────────────
    print("\n  E2E TIME FRACTION BREAKDOWN")
    print(_SEP)
    for r in rs:
        e2e_total = sum(r.e2e_s) or 1
        retr_frac = 100 * sum(r.retrieval_s) / e2e_total
        gen_frac  = 100 * sum(r.gen_s)       / e2e_total
        oth_frac  = max(0.0, 100 - retr_frac - gen_frac)
        lbl = _LABELS.get(r.config, r.config)
        print(f"  {lbl:<35}  Retrieval: {retr_frac:5.1f}%  "
              f"Generate: {gen_frac:5.1f}%  "
              f"Other: {oth_frac:4.1f}%")

    # ── Speedup table ─────────────────────────────────────────────────────
    base_e2e = sum(rs[0].e2e_s) if rs[0].e2e_s else 1
    print(f"\n  SPEEDUP  (vs {labels[0]} E2E)")
    print(_SEP)
    for r in rs:
        sp  = base_e2e / (sum(r.e2e_s) or 1)
        lbl = _LABELS.get(r.config, r.config)
        print(f"  {lbl:<38}  {sp:.2f}×")

    # ── Counterfactual: retrieval bottleneck ──────────────────────────────
    print(f"\n  COUNTERFACTUAL  — if path enumeration/retrieval were instant")
    print(f"  (shows the LLM generation cost alone, isolating retrieval overhead)")
    print(_SEP)
    base_gen = sum(rs[0].gen_only_e2e_s) if rs[0].gen_only_e2e_s else 1
    for r in rs:
        gen_total = sum(r.gen_only_e2e_s) or 1
        e2e_total = sum(r.e2e_s)          or 1
        retr_overhead_pct = 100 * (1 - gen_total / e2e_total)
        sp_cf = base_gen / gen_total
        lbl = _LABELS.get(r.config, r.config)
        print(f"  {lbl:<38}  gen-only={gen_total/len(r.e2e_s)*1e3:.0f}ms/batch  "
              f"retrieval_overhead={retr_overhead_pct:.1f}%  "
              f"counterfactual_speedup={sp_cf:.2f}×")

    # ── GraphRAG path enumeration analysis ────────────────────────────────
    has_path_stats = any(r.paths_per_query for r in rs)
    if has_path_stats:
        print(f"\n  PATH ENUMERATION ANALYSIS  (GraphRAG configs only)")
        print(f"  Motivates SubgraphKVCache: paths share common sub-paths across queries.")
        print(_SEP)
        hdr()
        for attr, name, unit, fmt in [
            ("paths_per_query",     "Paths found / query  mean",     "",     ".1f"),
            ("paths_per_query",     "Paths found / query  p95",      "",     ".1f"),
            ("tokens_per_path",     "Avg tokens / path  mean",       "tok",  ".1f"),
            ("path_prefix_sharing", "Shared-prefix ratio  mean",     "",     ".3f"),
            ("graph_nodes",         "Subgraph nodes  mean",          "",     ".1f"),
            ("graph_edges",         "Subgraph edges  mean",          "",     ".1f"),
        ]:
            vals = []
            for r in rs:
                lst = getattr(r, attr)
                if lst:
                    a = np.array(lst, dtype=float)
                    if "p95" in name:
                        vals.append(float(np.percentile(a, 95)))
                    else:
                        vals.append(float(np.mean(a)))
                else:
                    vals.append(None)
            _row(name, vals, unit, fmt=fmt)

        # Path prefix sharing interpretation
        print(f"\n  Path prefix sharing: fraction of path PAIRS within a query that")
        print(f"  share the same starting node. Higher = more sub-path reuse potential.")
        print(f"  A 50%+ ratio strongly motivates caching at the sub-path granularity.")

        # Retrieval time per path
        nbatch_graph = max(len(r.retrieval_s) for r in rs if r.paths_per_query) if any(r.paths_per_query for r in rs) else 0
        print(f"\n  BFS cost per path found:")
        if nbatch_graph <= 2:
            print(f"  NOTE: only {nbatch_graph} batch(es) measured — "
                  f"small differences between configs are run-to-run noise,")
            print(f"  not a real effect of KV on/off (KV setting does not affect retrieval).")
        print(_SEP)
        for r in rs:
            if r.paths_per_query and r.retrieval_s:
                total_paths  = sum(r.paths_per_query)
                total_retr_s = sum(r.retrieval_s)
                lbl = _LABELS.get(r.config, r.config)
                ms_per_path = total_retr_s * 1e3 / max(1, total_paths)
                print(f"  {lbl:<38}  {total_paths} total paths  "
                      f"→ {ms_per_path:.2f} ms/path")

    # ── Profiler summaries ────────────────────────────────────────────────
    for r in rs:
        if r.profiler_summary:
            s   = r.profiler_summary
            cat = s.get("categories_%", {})
            lbl = _LABELS.get(r.config, r.config)
            print(f"\n  PROFILER [{lbl}]  "
                  f"total={s.get('total_cuda_ms',0):.1f}ms  "
                  f"Attn={cat.get('attention',0):.0f}%  "
                  f"GEMM={cat.get('gemm',0):.0f}%  "
                  f"Elem={cat.get('elemwise',0):.0f}%  "
                  f"MemOps={cat.get('memory_ops',0):.0f}%")

    print(f"\n{'═'*84}")


# ═══════════════════════════════════════════════════════════════════════════
# §12  Output: JSON + CSV
# ═══════════════════════════════════════════════════════════════════════════

def save_json(results: Dict[str, RunResult], path: str) -> None:
    def _r(r: RunResult) -> Dict:
        dbw = _decode_bw_list(r)
        rl  = roofline(safe_mean(r.analytic_flops) or 0,
                       (safe_mean(r.analytic_total_GB) or 1) * 1e9)
        return {
            "config":     r.config,
            "model":      r.model_name,
            "n_samples":  r.n_samples,
            "use_kv":     r.use_kv,
            "quality":    {"hits_at_1": r.hits_at_1, "f1": r.f1},
            "offline":    {"model_load_s": r.model_load_s,
                           "index_build_s": r.index_build_s,
                           "n_passages": r.n_indexed},
            "retrieval":  {"lat_ms":     pct(r.retrieval_s),
                           "embed_ms":   pct(r.embed_query_s),
                           "avg_items":  safe_mean(r.n_retrieved),
                           "avg_ctx_tok": safe_mean(r.ctx_tok)},
            "generation": {"ttft_ms":    pct(r.ttft_s),
                           "decode_ms":  pct(r.decode_s),
                           "gen_ms":     pct(r.gen_s),
                           "tok_per_s":  pct(_gen_tps_list(r), unit=1),
                           "decode_bw_GBs": pct(dbw, unit=1)},
            "e2e_ms":     pct(r.e2e_s),
            "memory":     {"model_GB":      r.model_size_GB,
                           "peak_hbm_GB":   pct(r.hbm_peak_B,  1/1e9),
                           "alloc_hbm_GB":  pct(r.hbm_alloc_B, 1/1e9)},
            "analytic":   {"total_GB":  pct(r.analytic_total_GB, 1),
                           "attn_GB":   pct(r.analytic_attn_GB,  1),
                           "ffn_GB":    pct(r.analytic_ffn_GB,   1),
                           "kv_GB":     pct(r.analytic_kv_GB,    1)},
            "roofline":   rl,
            "path_stats": {"paths_per_query":     pct(r.paths_per_query, 1),
                           "tokens_per_path":     pct(r.tokens_per_path, 1),
                           "prefix_sharing":      pct(r.path_prefix_sharing, 1),
                           "graph_nodes":         pct(r.graph_nodes, 1),
                           "graph_edges":         pct(r.graph_edges, 1),
                           } if r.paths_per_query else {},
            "counterfactual": {"gen_only_ms": pct(r.gen_only_e2e_s)},
            "hw":         {"gpu":      r.gpu_name,
                           "mem_GB":   r.gpu_mem_GB,
                           "sm_pct":   pct(r.sm_pct,  1),
                           "power_W":  pct(r.power_W, 1)},
            "profiler":   r.profiler_summary,
        }

    data = {"metadata": {
                "ts":               time.strftime("%Y-%m-%dT%H:%M:%S"),
                "a100_hbm_bw_GBs": A100_HBM_BW_GBS,
                "a100_fp16_tflops": A100_FP16_TFLOPS,
                "a100_ridge":       A100_RIDGE,
            },
            **{c: _r(r) for c, r in results.items()}}
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"[Results] JSON → {path}")


def save_csv(results: Dict[str, RunResult], path: str) -> None:
    configs = list(results.keys())
    rows    = []

    def add(m, unit, fn):
        row = [m, unit]
        for c in configs:
            try:
                v = fn(results[c])
                row.append(f"{v:.4f}" if v is not None else "")
            except Exception:
                row.append("")
        rows.append(row)

    add("Hits@1",              "",     lambda r: r.hits_at_1)
    add("F1",                  "",     lambda r: r.f1)
    add("Decode KV cache",     "bool", lambda r: int(r.use_kv))
    add("Avg prompt tokens",   "tok",  lambda r: safe_mean(r.input_tok) or 0)
    add("Avg context tokens",  "tok",  lambda r: safe_mean(r.ctx_tok)   or 0)
    add("Avg items retrieved", "",     lambda r: safe_mean(r.n_retrieved) or 0)
    add("Retrieval mean (ms)", "ms",   lambda r: pct(r.retrieval_s)["mean"] if r.retrieval_s else 0)
    add("Retrieval p95 (ms)",  "ms",   lambda r: pct(r.retrieval_s)["p95"]  if r.retrieval_s else 0)
    add("TTFT mean (ms)",      "ms",   lambda r: pct(r.ttft_s)["mean"]   if r.ttft_s else 0)
    add("Decode mean (ms)",    "ms",   lambda r: pct(r.decode_s)["mean"] if r.decode_s else 0)
    add("E2E mean (ms)",       "ms",   lambda r: pct(r.e2e_s)["mean"]   if r.e2e_s else 0)
    add("E2E p95 (ms)",        "ms",   lambda r: pct(r.e2e_s)["p95"]    if r.e2e_s else 0)
    add("E2E p99 (ms)",        "ms",   lambda r: pct(r.e2e_s)["p99"]    if r.e2e_s else 0)
    add("Gen tok/s",           "tok/s",lambda r: safe_mean(_gen_tps_list(r)) or 0)
    add("Queries/s",           "q/s",  lambda r: r.n_samples / sum(r.e2e_s) if r.e2e_s else 0)
    add("Decode BW mean (GB/s)","GB/s",lambda r: safe_mean(_decode_bw_list(r)) or 0)
    add("% HBM peak",          "%",    lambda r: 100*(safe_mean(_decode_bw_list(r)) or 0)/A100_HBM_BW_GBS)
    add("Peak HBM (GB)",       "GB",   lambda r: safe_mean(r.hbm_peak_B, 1/1e9) or 0)
    add("Model size (GB)",     "GB",   lambda r: r.model_size_GB)
    add("Analytic HBM (GB)",   "GB",   lambda r: safe_mean(r.analytic_total_GB) or 0)
    add("Arith intensity",     "F/B",  lambda r: roofline(
        safe_mean(r.analytic_flops) or 0,
        (safe_mean(r.analytic_total_GB) or 1)*1e9)["arith_intensity"])
    add("Retrieval % E2E",     "%",    lambda r: 100*sum(r.retrieval_s)/max(1,sum(r.e2e_s)))
    add("Generate % E2E",      "%",    lambda r: 100*sum(r.gen_s)/max(1,sum(r.e2e_s)))

    with open(path, "w") as f:
        f.write("Metric,Unit," + ",".join(configs) + "\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")
    print(f"[Results] CSV  → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# §13  CLI + main
# ═══════════════════════════════════════════════════════════════════════════

_ALL_CONFIGS = ["rag", "rag_kv", "graphrag", "graphrag_kv"]


def build_parser():
    p = argparse.ArgumentParser(
        description="RAG vs GraphRAG Workload Analysis (ISCA)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset",    default="rmanluo/RoG-cwq")
    p.add_argument("--llm_model",  default="rmanluo/RoG",
                   choices=[
                       "rmanluo/RoG",
                       "meta-llama/Llama-2-7b-hf",
                       "meta-llama/Llama-2-13b-hf",
                       "Qwen/Qwen3-8B",
                       "facebook/opt-6.7b",
                       "facebook/opt-13b",
                       "facebook/opt-30b",
                       "google/gemma-2-2b",
                       "google/gemma-2-9b",
                       "google/gemma-2-27b",
                       "Qwen/Qwen3-Next-80B-A3B-Thinking",
                       "Qwen/Qwen3-32B",
                   ],
                   help="HuggingFace model for generation (default: rmanluo/RoG)")
    p.add_argument("--auth_token", default=None)
    p.add_argument("--samples",    type=int, default=32)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--num_beams",  type=int, default=1)
    p.add_argument("--max_tokens", type=int, default=50)
    p.add_argument("--hop",        type=int, default=3,
                   help="GraphRAG BFS hop constraint  [3]")
    p.add_argument("--top_k",      type=int, default=5,
                   help="RAG top-k passages  [5]")
    p.add_argument("--configs",    nargs="+", choices=_ALL_CONFIGS,
                   default=_ALL_CONFIGS)
    p.add_argument("--profile",    action="store_true",
                   help="Enable torch.profiler + per-kernel analysis")
    p.add_argument("--out_dir",    default="workload_results")
    p.add_argument("--device",     type=int, default=0)
    p.add_argument("--neurips",    action="store_true",
                   help="Preset: samples=64 batch=16 beams=4 max_tokens=50")
    return p


def main():
    args = build_parser().parse_args()
    if args.neurips:
        args.samples = 64; args.batch_size = 16
        args.num_beams = 4; args.max_tokens = 50

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = (f"{Path(args.dataset).name.replace('/', '-')}"
           f"_{args.llm_model.split('/')[-1]}"
           f"_N{args.samples}_B{args.batch_size}_b{args.num_beams}")

    print(f"\n{'#'*70}")
    print(f"#  RAG vs GraphRAG — ISCA Workload Analysis")
    print(f"#  Model:   {args.llm_model}")
    print(f"#  Dataset: {args.dataset}   N={args.samples}  batch={args.batch_size}")
    print(f"#  Configs: {args.configs}")
    print(f"#  Params:  beams={args.num_beams}  max_tokens={args.max_tokens}  "
          f"hop={args.hop}  top_k={args.top_k}")
    for i in range(torch.cuda.device_count()):
        print(f"#  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"{'#'*70}\n")

    # Load dataset
    print(f"Loading dataset: {args.dataset}  N={args.samples}")
    dataset = load_ds(args.dataset, "test", args.samples, args.seed)
    print(f"Dataset loaded: {len(dataset)} samples")

    # Build RAG index ONCE (shared across rag and rag_kv)
    rag_retriever: Optional[RAGRetrieval] = None
    need_rag = any(c.startswith("rag") for c in args.configs)
    if need_rag:
        if not _RAG_DEPS:
            print(f"[WARN] RAG deps missing ({_RAG_ERR}).  Skipping rag/rag_kv.")
            args.configs = [c for c in args.configs if not c.startswith("rag")]
        else:
            rag_retriever = RAGRetrieval()
            rag_retriever.build_global_index(dataset)

    common = dict(
        dataset=dataset, model_name=args.llm_model,
        batch_size=args.batch_size, num_beams=args.num_beams,
        max_new_tokens=args.max_tokens, seed=args.seed,
        hop=args.hop, top_k=args.top_k,
        rag_retriever=rag_retriever,
        device=args.device, profile=args.profile,
        prof_dir=str(out_dir / "traces"),
    )

    results: Dict[str, RunResult] = {}
    for cfg in args.configs:
        try:
            results[cfg] = run_config(cfg, **common)
            torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            print(f"[ERROR] Config '{cfg}' failed:")
            traceback.print_exc()

    if not results:
        print("[ERROR] No configs completed."); return

    print_report(results)
    save_json(results, str(out_dir / f"{tag}_workload.json"))
    save_csv( results, str(out_dir / f"{tag}_workload.csv"))


if __name__ == "__main__":
    main()
