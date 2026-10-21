#!/usr/bin/env python3
"""
profile.py  —  RAG vs GraphRAG Deep Profiling & Ablation Study  (ISCA paper)
=============================================================================

Compares VANILLA RAG vs VANILLA GraphRAG across multiple dimensions:
  • Both with standard decode KV cache OFF  (use_cache=False)
  • Both with standard decode KV cache ON   (use_cache=True)
  • NO cross-query caching (no PFKV, no PassageKV, no SubgraphKV)

EXPERIMENTS
-----------
  1. MODEL SWEEP        — fixed dataset, sweep models → shows how model size
                          shifts the RAG-vs-GraphRAG performance & hw profile
  2. DEPTH SWEEP        — fixed model, sweep GraphRAG hop_constraint 1→4
                          → shows latency/memory/compute scaling with graph depth
  3. BATCH SIZE SWEEP   — fixed model, sweep batch size 1,2,4,8,16,32,64
                          → throughput scaling & memory pressure
  4. CONTEXT SWEEP      — RAG top_k vs GraphRAG hops → prompt length impact
  5. GRAPH CHAR         — how retrieval time changes with graph structural
                          properties (nodes, edges, diameter, degree, density,
                          longest path, connected components, relation types)
  6. PHASE-LEVEL HW     — separate prefill vs decode roofline, MFU, bandwidth

HARDWARE CHARACTERISATION METRICS
----------------------------------
  Per-phase (prefill / decode):
    • Latency breakdown            (ms)
    • Arithmetic intensity         (FLOP/byte)
    • Roofline classification      (memory-bound vs compute-bound)
    • HBM bandwidth utilisation    (GB/s, % of peak)
    • Model FLOP Utilisation (MFU) (%)
  Per-experiment:
    • Energy per query             (J/query)
    • KV cache memory pressure     (GB)
    • Throughput                    (tok/s, queries/s)
    • Peak HBM usage               (GB)

FAIRNESS
--------
  Same model, dataset, seed, prompt template, generation config for all configs.
  RAG indexes ALL KG triplets globally (production-style FAISS).
  GraphRAG uses per-query BFS subgraph traversal.
  The only difference is the retrieval mechanism.

Usage
-----
  # Quick smoke test (2 models, 16 samples)
  python profile.py --preset smoke

  # Depth ablation only (default model)
  python profile.py --experiments depth --samples 64

  # Full ISCA suite (all experiments, all models)
  python profile.py --preset isca

  # Single model, all experiments
  python profile.py --llm_model facebook/opt-6.7b --samples 64

  # Specific experiments
  python profile.py --experiments model depth batch --samples 32
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
import time
import traceback
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
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

# ── Optional deps ────────────────────────────────────────────────────────────
try:
    import faiss                         # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore
    _RAG_DEPS = True
except ImportError:
    _RAG_DEPS = False

try:
    import pynvml                        # type: ignore
    pynvml.nvmlInit()
    _NVML = True
except Exception:
    _NVML = False


# ═══════════════════════════════════════════════════════════════════════════════
# §1  Hardware constants
# ═══════════════════════════════════════════════════════════════════════════════

# Detect GPU and set constants; fall back to A100 defaults
def _detect_hw() -> Dict[str, float]:
    hw = {
        "hbm_bw_GBs":  1_935.0,   # A100 PCIe 80GB HBM2e
        "fp16_tflops":  312.0,     # A100 Tensor Core FP16 peak
        "pcie_bw_GBs":  64.0,     # PCIe 4.0 x16
        "gpu_name":     "A100-PCIe-80GB",
    }
    if not torch.cuda.is_available():
        return hw
    name = torch.cuda.get_device_name(0).lower()
    # H100 SXM
    if "h100" in name and "sxm" in name:
        hw.update(hbm_bw_GBs=3_350.0, fp16_tflops=989.0, gpu_name="H100-SXM")
    elif "h100" in name:
        hw.update(hbm_bw_GBs=2_000.0, fp16_tflops=756.0, gpu_name="H100-PCIe")
    # A100 SXM
    elif "a100" in name and "sxm" in name:
        hw.update(hbm_bw_GBs=2_039.0, fp16_tflops=312.0, gpu_name="A100-SXM")
    elif "a100" in name:
        hw.update(hbm_bw_GBs=1_935.0, fp16_tflops=312.0, gpu_name="A100-PCIe")
    # L40S
    elif "l40s" in name:
        hw.update(hbm_bw_GBs=864.0, fp16_tflops=362.0, gpu_name="L40S")
    # A6000
    elif "a6000" in name:
        hw.update(hbm_bw_GBs=768.0, fp16_tflops=155.0, gpu_name="A6000")
    hw["ridge"] = (hw["fp16_tflops"] * 1e12) / (hw["hbm_bw_GBs"] * 1e9)
    return hw

HW = _detect_hw()


# ═══════════════════════════════════════════════════════════════════════════════
# §2  Models registry
# ═══════════════════════════════════════════════════════════════════════════════

# Models from main.py / workload.py — ordered by parameter count
MODELS = {
    "small": [
        "rmanluo/RoG",             # ~2B (LLaMA fine-tune)
        "google/gemma-2-2b",       # 2B
    ],
    "medium": [
        "facebook/opt-6.7b",       # 6.7B
        "Qwen/Qwen3-8B",          # 8B
        "google/gemma-2-9b",       # 9B
    ],
    "large": [
        "facebook/opt-13b",        # 13B
        "google/gemma-2-27b",      # 27B
        "Qwen/Qwen3-32B",         # 32B
    ],
}

ALL_MODELS = MODELS["small"] + MODELS["medium"] + MODELS["large"]

DEFAULT_DATASET = "rmanluo/RoG-cwq"


# ═══════════════════════════════════════════════════════════════════════════════
# §3  Hardware instrumentation (reused from workload.py patterns)
# ═══════════════════════════════════════════════════════════════════════════════

class CUDATimer:
    """GPU-accurate timing via CUDA Events."""
    def __init__(self, dev: int = 0):
        self.dev = dev
        self._s = torch.cuda.Event(enable_timing=True)
        self._e = torch.cuda.Event(enable_timing=True)
        self.elapsed_s: float = 0.0

    def __enter__(self):
        self._s.record()
        return self

    def __exit__(self, *_):
        self._e.record()
        torch.cuda.synchronize(self.dev)
        self.elapsed_s = self._s.elapsed_time(self._e) / 1e3


class HBMSnap:
    """Capture torch.cuda.memory_stats before/after a block."""
    def __init__(self, dev: int = 0):
        self.dev = dev
        self.alloc_delta_B: int = 0
        self.peak_B: int = 0
        self._before: Dict = {}

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


import subprocess

class PowerSampler:
    """GPU power + SM utilisation via nvidia-smi dmon."""
    def __init__(self, dev: int = 0, interval_s: float = 1):
        self.dev = dev
        self._interval = max(1, int(interval_s))
        self._proc: Optional[subprocess.Popen] = None
        self._t0 = 0.0
        self.energy_J = 0.0
        self.mean_W = 0.0
        self.peak_W = 0.0
        self.duration_s = 0.0
        self.mean_sm_pct = 0.0
        self.peak_sm_pct = 0.0

    def __enter__(self):
        try:
            self._proc = subprocess.Popen(
                ["nvidia-smi", "dmon", "-i", str(self.dev),
                 "-s", "pu", "-d", str(self._interval)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
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
        sm_vals, pw_vals = [], []
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            try:
                pw_vals.append(float(parts[1]))
                sm_vals.append(float(parts[4]))
            except (IndexError, ValueError):
                continue
        if not pw_vals:
            return
        self.mean_W = float(np.mean(pw_vals))
        self.peak_W = float(np.max(pw_vals))
        self.mean_sm_pct = float(np.mean(sm_vals))
        self.peak_sm_pct = float(np.max(sm_vals))
        if self.duration_s > 0:
            self.energy_J = self.mean_W * self.duration_s


# ═══════════════════════════════════════════════════════════════════════════════
# §4  Analytical models
# ═══════════════════════════════════════════════════════════════════════════════

def model_info(m: torch.nn.Module) -> Tuple[int, float, str]:
    params = sum(p.numel() for p in m.parameters())
    dt = next(m.parameters()).dtype
    bpp = torch.finfo(dt).bits // 8
    return params, params * bpp / 1e9, str(dt)


def get_arch(gen) -> Dict:
    cfg = gen.model.config
    d   = getattr(cfg, "hidden_size",       getattr(cfg, "d_model", 4096))
    nl  = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 32))
    nh  = getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", 32))
    nkv = getattr(cfg, "num_key_value_heads", nh)
    dh  = d // nh
    fi  = getattr(cfg, "intermediate_size", getattr(cfg, "ffn_dim", 4 * d))
    return {"d": d, "nl": nl, "nh": nh, "nkv": nkv, "dh": dh,
            "ffn_mult": max(1, fi // d)}


def kv_cache_bytes(nl: int, nkv: int, dh: int, seq: int,
                   B: int, dtype_B: int = 2) -> float:
    """HBM occupied by the decode KV cache (K+V per layer)."""
    return 2 * nl * nkv * dh * seq * B * dtype_B


def analytic_prefill_flops(seq: int, d: int, nl: int, nh: int, dh: int,
                           ffn_mult: int, B: int) -> float:
    """FLOPs for the prefill phase (full-sequence forward pass)."""
    ffn   = ffn_mult * d
    qkv   = 2 * B * seq * d * d * 3        # Q, K, V projections
    score = 2 * B * nh * seq * seq * dh     # attention scores (quadratic in seq)
    out   = 2 * B * seq * d * d             # output projection
    ff    = 2 * B * seq * d * ffn * 3       # FFN (gate + up + down)
    return float(nl * (qkv + score + out + ff))


def analytic_decode_flops(d: int, nl: int, nh: int, dh: int,
                          ffn_mult: int, B: int, n_steps: int) -> float:
    """FLOPs for the decode phase (n_steps autoregressive steps, seq_len=1 each)."""
    ffn  = ffn_mult * d
    # Per step: QKV proj + score (1 × cached_seq) + out proj + FFN
    # We approximate cached_seq as constant (ignoring growth per step)
    qkv  = 2 * B * 1 * d * d * 3
    out  = 2 * B * 1 * d * d
    ff   = 2 * B * 1 * d * ffn * 3
    per_step = nl * (qkv + out + ff)  # attention score is small for seq=1
    return float(per_step * n_steps)


def analytic_prefill_hbm_bytes(seq: int, d: int, nl: int, nh: int, nkv: int,
                               dh: int, ffn_mult: int, B: int,
                               use_kv: bool, dtype_B: int = 2) -> float:
    """Estimated HBM traffic for prefill pass (bytes)."""
    ffn = ffn_mult * d
    e = dtype_B
    # Weights read once per layer
    attn_wt  = (3 * d * d + d * d) * e
    attn_act = (3 * B * seq * d + B * nh * seq * seq + B * seq * d) * e
    kv_write = 2 * B * nkv * seq * dh * e
    attn_lay = attn_wt + attn_act + (kv_write if use_kv else 0)

    ffn_wt  = (2 * d * ffn + ffn * d) * e
    ffn_act = (B * seq * d + 2 * B * seq * ffn) * e
    ffn_lay = ffn_wt + ffn_act

    return float(nl * (attn_lay + ffn_lay))


def analytic_decode_hbm_bytes(model_GB: float, nl: int, nkv: int, dh: int,
                              seq: int, B: int, n_steps: int,
                              use_kv: bool, dtype_B: int = 2) -> float:
    """
    Estimated HBM traffic for decode phase.
    KV=ON:  model weights + full KV cache read per step.
    KV=OFF: model weights per step (no cache, but recompute is compute-bound).
    """
    if use_kv:
        kv_per_step = kv_cache_bytes(nl, nkv, dh, seq, B, dtype_B)
        return (model_GB * 1e9 + kv_per_step) * n_steps
    else:
        return model_GB * 1e9 * n_steps


def roofline(flops: float, hbm_bytes: float) -> Dict:
    if hbm_bytes <= 0:
        return {"arith_intensity": float("inf"), "memory_bound": False,
                "attainable_TFLOPs": HW["fp16_tflops"]}
    ai = flops / hbm_bytes
    return {
        "arith_intensity":   ai,
        "memory_bound":      ai < HW["ridge"],
        "ridge_FLOP_byte":   HW["ridge"],
        "attainable_TFLOPs": min(HW["fp16_tflops"], ai * HW["hbm_bw_GBs"] / 1e3),
    }


def compute_mfu(achieved_tflops: float) -> float:
    """Model FLOP Utilisation = achieved / peak."""
    if HW["fp16_tflops"] <= 0:
        return 0.0
    return achieved_tflops / HW["fp16_tflops"]


# ═══════════════════════════════════════════════════════════════════════════════
# §5  RAG Retrieval (global FAISS index)
# ═══════════════════════════════════════════════════════════════════════════════

class RAGRetrieval:
    """Dense-passage retrieval: global FAISS index over all KG triplets."""

    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, embed_model: str = EMBED_MODEL):
        if not _RAG_DEPS:
            raise ImportError("sentence_transformers / faiss not installed")
        self._embedder = SentenceTransformer(embed_model, device="cpu")
        self._dim = self._embedder.get_sentence_embedding_dimension()
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
        embs = self._encode(passages)
        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(embs)
        self._passages = passages
        self.build_time_s = time.perf_counter() - t0
        print(f"[RAG] Index: {len(passages):,} passages in {self.build_time_s:.1f}s")

    def retrieve_batch(self, queries: List[str], top_k: int = 5
                       ) -> Tuple[List[Dict], float, float]:
        assert self._index is not None
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


# ═══════════════════════════════════════════════════════════════════════════════
# §6  Prompt formatters (same template for fairness)
# ═══════════════════════════════════════════════════════════════════════════════

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
# §7  Dataset loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_ds(name: str, split: str, n: int, seed: int):
    short = {"metaqa-1hop": 1, "metaqa-2hop": 2, "metaqa-3hop": 3}
    if name in short:
        hops = short[name]
        kb = SCRIPT_DIR / "metaQA" / "kb.txt"
        if kb.exists():
            ds = load_metaqa(hops=hops, split=split,
                             data_dir=str(SCRIPT_DIR / "metaQA"))
        elif hops == 3:
            ds = load_dataset("mhivanov123/metaqa-3hop",
                              split="validation" if split == "dev" else split)
        else:
            raise FileNotFoundError(
                f"MetaQA local files not found. For metaqa-{hops}hop, "
                "download from https://github.com/yuyuz/MetaQA")
    else:
        ds = load_dataset(name, split=split)
    if len(ds) > n:
        ds = ds.shuffle(seed=seed).select(range(n))
    return ds


# ═══════════════════════════════════════════════════════════════════════════════
# §8  Result dataclass
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProfileResult:
    """All metrics for a single (config, model, dataset, parameters) run."""
    # ── Experiment identity ──
    experiment:     str             # e.g. "model_sweep", "depth_sweep"
    config:         str             # "rag" | "rag_kv" | "graphrag" | "graphrag_kv"
    model_name:     str
    dataset_name:   str
    n_samples:      int
    batch_size:     int
    num_beams:      int
    max_new_tokens: int
    seed:           int
    use_kv:         bool
    hop_constraint: int   = 3
    top_k:          int   = 5

    # ── Model info ──
    model_params:   int   = 0
    model_size_GB:  float = 0.0
    model_dtype:    str   = ""
    arch:           Dict  = field(default_factory=dict)

    # ── GPU info ──
    gpu_name:       str   = ""
    gpu_mem_GB:     float = 0.0

    # ── Offline costs ──
    model_load_s:   float = 0.0
    index_build_s:  float = 0.0

    # ── Per-batch measurements ──
    retrieval_s:       List[float] = field(default_factory=list)
    ttft_s:            List[float] = field(default_factory=list)
    decode_s:          List[float] = field(default_factory=list)
    gen_s:             List[float] = field(default_factory=list)
    gpu_gen_s:         List[float] = field(default_factory=list)
    e2e_s:             List[float] = field(default_factory=list)

    input_tok:         List[int]   = field(default_factory=list)
    output_tok:        List[int]   = field(default_factory=list)
    ctx_tok:           List[int]   = field(default_factory=list)
    n_retrieved:       List[int]   = field(default_factory=list)

    hbm_peak_B:        List[int]   = field(default_factory=list)

    # ── Per-phase analytics ──
    prefill_flops:     List[float] = field(default_factory=list)
    prefill_hbm_B:     List[float] = field(default_factory=list)
    decode_flops:      List[float] = field(default_factory=list)
    decode_hbm_B:      List[float] = field(default_factory=list)

    # ── Hardware metrics ──
    sm_pct:            List[float] = field(default_factory=list)
    power_W:           List[float] = field(default_factory=list)
    energy_J:          List[float] = field(default_factory=list)

    # ── GraphRAG per-query graph characteristics ──
    paths_per_query:       List[int]   = field(default_factory=list)
    graph_nodes:           List[int]   = field(default_factory=list)
    graph_edges:           List[int]   = field(default_factory=list)
    graph_diameter:        List[int]   = field(default_factory=list)
    graph_longest_path:    List[int]   = field(default_factory=list)  # longest BFS path found
    graph_avg_degree:      List[float] = field(default_factory=list)
    graph_max_degree:      List[int]   = field(default_factory=list)
    graph_density:         List[float] = field(default_factory=list)
    graph_components:      List[int]   = field(default_factory=list)
    graph_n_relations:     List[int]   = field(default_factory=list)  # unique relation types
    per_query_retrieval_s: List[float] = field(default_factory=list)  # per-query retrieval time
    per_query_path_len:    List[float] = field(default_factory=list)  # avg hops per path per query

    # ── Quality ──
    hits_at_1:      float = 0.0
    f1:             float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# §9  Core single-config runner
# ═══════════════════════════════════════════════════════════════════════════════

_LABELS = {
    "rag":         "RAG (KV=OFF)",
    "rag_kv":      "RAG (KV=ON)",
    "graphrag":    "GraphRAG (KV=OFF)",
    "graphrag_kv": "GraphRAG (KV=ON)",
}


def run_single(
    *,
    experiment:     str,
    config:         str,
    dataset,
    dataset_name:   str,
    model_name:     str,
    batch_size:     int,
    num_beams:      int,
    max_new_tokens: int,
    seed:           int,
    hop:            int,
    top_k:          int,
    rag_retriever:  Optional[RAGRetrieval],
    device:         int,
) -> ProfileResult:
    """Run one complete profiling pass and return all metrics."""

    is_rag = config.startswith("rag")
    use_kv = config.endswith("_kv")
    label  = _LABELS[config]

    set_reproducible_seed(seed)

    result = ProfileResult(
        experiment=experiment, config=config, model_name=model_name,
        dataset_name=dataset_name, n_samples=len(dataset),
        batch_size=batch_size, num_beams=num_beams,
        max_new_tokens=max_new_tokens, seed=seed, use_kv=use_kv,
        hop_constraint=hop, top_k=top_k,
    )

    if torch.cuda.is_available():
        result.gpu_name = torch.cuda.get_device_name(device)
        result.gpu_mem_GB = torch.cuda.get_device_properties(device).total_memory / 1e9

    # ── Load model ────────────────────────────────────────────────────────
    # Use device_map=None so model loads to CPU, then .to(cuda:N).
    # Do NOT use device_map="auto" — on multi-GPU nodes it spreads the
    # model across GPUs, placing RoPE inv_freq on a different device than
    # the inputs, which causes a CUBLAS crash.
    print(f"\n  [{label}] Loading {model_name}  use_kv={use_kv}  (GPU {device})")
    t0 = time.perf_counter()
    gen = PathFirstGraphRAGGeneration(
        model_name,
        device=f"cuda:{device}",
        use_kv_cache=use_kv,
        use_pfkv=False,          # vanilla only
        device_map=None,         # load to CPU, then .to(cuda:N)
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )
    result.model_load_s = time.perf_counter() - t0

    # Guard: ensure ALL RoPE inv_freq buffers live on the model's device.
    # On multi-GPU nodes (even with device_map=None), persistent=False
    # buffers sometimes fail to migrate during .to().  This explicit pass
    # costs nothing when everything is already correct.
    _target = gen.input_device
    for _m in gen.model.modules():
        if hasattr(_m, "inv_freq") and _m.inv_freq is not None:
            if _m.inv_freq.device != _target:
                _m.inv_freq = _m.inv_freq.to(_target)

    params, size_gb, dtype_s = model_info(gen.model)
    result.model_params = params
    result.model_size_GB = size_gb
    result.model_dtype = dtype_s
    result.arch = get_arch(gen)
    arch = result.arch

    if is_rag and rag_retriever is not None:
        result.index_build_s = rag_retriever.build_time_s

    print(f"  [{label}] {params/1e9:.2f}B params, {size_gb:.2f}GB ({dtype_s})")

    total = len(dataset)
    nbatch = (total + batch_size - 1) // batch_size

    # ── Warmup (1 batch, not measured) ────────────────────────────────────
    wn = min(batch_size, total)
    wbatch = dataset.select(range(wn))
    wqueries = [wbatch[i]["question"] for i in range(len(wbatch))]
    if is_rag and rag_retriever is not None:
        wr, _, _ = rag_retriever.retrieve_batch(wqueries, top_k)
        wp = [_rag_prompt(r["query"], r["passages"], model_name) for r in wr]
    else:
        wret = SimpleGraphRAGRetrieval(wbatch)
        wrr, _ = wret.retrieve_batch(hop)
        wp = [format_prompt(r["query"], r["paths"], model_name) for r in wrr]
    _ = gen.generate_batch(wp, None)
    torch.cuda.synchronize(device)

    # ── Measurement loop ──────────────────────────────────────────────────
    total_hits = total_f1 = 0.0
    total_proc = 0

    for bi in tqdm(range(nbatch), desc=f"  [{label}]", leave=False):
        si = bi * batch_size
        ei = min(si + batch_size, total)
        bds = dataset.select(range(si, ei))
        blist = [bds[i] for i in range(len(bds))]
        queries = [s["question"] for s in blist]
        answers = [s["answer"]   for s in blist]

        e2e_t = time.perf_counter()

        # ── Retrieval ─────────────────────────────────────────────────────
        if is_rag and rag_retriever is not None:
            retr_res, emb_s, srch_s = rag_retriever.retrieve_batch(queries, top_k)
            retr_s = emb_s + srch_s
            prompts = [_rag_prompt(r["query"], r["passages"], model_name)
                       for r in retr_res]
            n_items = [len(r["passages"]) for r in retr_res]
        else:
            t_retr = time.perf_counter()
            retriever = SimpleGraphRAGRetrieval(bds)
            retr_result, _ = retriever.retrieve_batch(hop)
            retr_s = time.perf_counter() - t_retr
            prompts = [format_prompt(r["query"], r["paths"], model_name)
                       for r in retr_result]
            n_items = [len(r["paths"]) for r in retr_result]

            # ── Per-query graph characteristics ───────────────────────────
            for qi in range(len(bds)):
                q_retr_t0 = time.perf_counter()
                g_data = bds[qi].get("graph", [])
                G = build_graph(g_data) if g_data else nx.Graph()
                q_retr_t = time.perf_counter() - q_retr_t0

                n_nodes = G.number_of_nodes()
                n_edges = G.number_of_edges()
                result.graph_nodes.append(n_nodes)
                result.graph_edges.append(n_edges)

                paths = retr_result[qi].get("paths", [])
                result.paths_per_query.append(len(paths))

                # Diameter (longest shortest path) — only for connected graphs
                if n_nodes > 1 and nx.is_connected(G):
                    try:
                        result.graph_diameter.append(nx.diameter(G))
                    except nx.NetworkXError:
                        result.graph_diameter.append(0)
                else:
                    # For disconnected graphs, use max diameter across components
                    max_diam = 0
                    for comp in nx.connected_components(G):
                        if len(comp) > 1:
                            sg = G.subgraph(comp)
                            try:
                                max_diam = max(max_diam, nx.diameter(sg))
                            except nx.NetworkXError:
                                pass
                    result.graph_diameter.append(max_diam)

                # Longest retrieved path (in hops)
                if paths:
                    max_path_len = max(
                        len(p) if isinstance(p, list) else len(p.split(".")) // 2
                        for p in paths)
                    avg_path_len = np.mean([
                        len(p) if isinstance(p, list) else len(p.split(".")) // 2
                        for p in paths])
                    result.graph_longest_path.append(max_path_len)
                    result.per_query_path_len.append(float(avg_path_len))
                else:
                    result.graph_longest_path.append(0)
                    result.per_query_path_len.append(0.0)

                # Degree stats
                if n_nodes > 0:
                    degrees = [d for _, d in G.degree()]
                    result.graph_avg_degree.append(float(np.mean(degrees)))
                    result.graph_max_degree.append(max(degrees))
                else:
                    result.graph_avg_degree.append(0.0)
                    result.graph_max_degree.append(0)

                # Density
                result.graph_density.append(nx.density(G) if n_nodes > 1 else 0.0)

                # Connected components
                result.graph_components.append(nx.number_connected_components(G))

                # Unique relation types
                rels = set()
                for triplet in g_data:
                    rels.add(str(triplet[1]).strip())
                result.graph_n_relations.append(len(rels))

                # Per-query retrieval time (distribute batch time evenly as
                # SimpleGraphRAGRetrieval doesn't track per-query)
                result.per_query_retrieval_s.append(
                    retr_s / max(1, len(bds)))

        result.retrieval_s.append(retr_s)
        result.n_retrieved.extend(n_items)

        # ── Token counts ──────────────────────────────────────────────────
        toks = [gen.tokenizer.encode(p) for p in prompts]
        plen = [len(t) for t in toks]
        base_p = [_rag_prompt(q, [], model_name) if is_rag
                  else format_prompt(q, [], model_name) for q in queries]
        base_len = [len(gen.tokenizer.encode(p)) for p in base_p]
        ctx_len = [max(0, pl - bl) for pl, bl in zip(plen, base_len)]
        result.input_tok.extend(plen)
        result.ctx_tok.extend(ctx_len)

        # ── Generation with HW instrumentation ───────────────────────────
        with PowerSampler(device) as ps:
            with HBMSnap(device) as hbm:
                with CUDATimer(device) as gpu_t:
                    gen_t0 = time.perf_counter()
                    preds = gen.generate_batch(prompts, None)
                    gen_wall = time.perf_counter() - gen_t0

        result.gen_s.append(gen_wall)
        result.gpu_gen_s.append(gpu_t.elapsed_s)
        result.hbm_peak_B.append(hbm.peak_B)

        # TTFT / decode split
        ttft = gen.ttfts[-1] if gen.ttfts else 0.0
        dec = max(0.0, gen_wall - ttft)
        result.ttft_s.append(ttft)
        result.decode_s.append(dec)

        out_lens = [len(gen.tokenizer.encode(p)) for p in preds]
        result.output_tok.extend(out_lens)

        # ── Per-phase analytical estimates ────────────────────────────────
        avg_seq = int(np.mean(plen))
        B = len(prompts)
        avg_out = int(np.mean(out_lens)) if out_lens else 1

        pf_flops = analytic_prefill_flops(
            avg_seq, arch["d"], arch["nl"], arch["nh"], arch["dh"],
            arch["ffn_mult"], B)
        pf_hbm = analytic_prefill_hbm_bytes(
            avg_seq, arch["d"], arch["nl"], arch["nh"], arch["nkv"],
            arch["dh"], arch["ffn_mult"], B, use_kv)
        dc_flops = analytic_decode_flops(
            arch["d"], arch["nl"], arch["nh"], arch["dh"],
            arch["ffn_mult"], B, avg_out)
        dc_hbm = analytic_decode_hbm_bytes(
            size_gb, arch["nl"], arch["nkv"], arch["dh"],
            avg_seq, B, avg_out, use_kv)

        result.prefill_flops.append(pf_flops)
        result.prefill_hbm_B.append(pf_hbm)
        result.decode_flops.append(dc_flops)
        result.decode_hbm_B.append(dc_hbm)

        # ── Power & energy ────────────────────────────────────────────────
        result.sm_pct.append(ps.mean_sm_pct)
        result.power_W.append(ps.mean_W)
        result.energy_J.append(ps.energy_J)

        # ── E2E ───────────────────────────────────────────────────────────
        e2e = time.perf_counter() - e2e_t
        result.e2e_s.append(e2e)

        # ── Quality ───────────────────────────────────────────────────────
        m = evaluate_metrics(preds, answers)
        n = len(preds)
        total_hits += m["hits_at_1"] * n
        total_f1   += m["f1"] * n
        total_proc += n

    nn = total_proc or 1
    result.hits_at_1 = total_hits / nn
    result.f1 = total_f1 / nn

    # Cleanup
    del gen
    torch.cuda.empty_cache()
    gc.collect()

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# §10  Derived metrics computation
# ═══════════════════════════════════════════════════════════════════════════════

def _safe(lst, fn=np.mean, default=0.0):
    if not lst:
        return default
    return float(fn(lst))


def _pct(lst, unit=1e3) -> Dict:
    a = np.array(lst, dtype=float) * unit
    if a.size == 0:
        return {"mean": 0, "med": 0, "p95": 0, "p99": 0, "std": 0}
    return {
        "mean": float(np.mean(a)),
        "med":  float(np.median(a)),
        "p95":  float(np.percentile(a, 95)),
        "p99":  float(np.percentile(a, 99)),
        "std":  float(np.std(a)),
    }


def derive_metrics(r: ProfileResult) -> Dict[str, Any]:
    """Compute all derived ISCA metrics from raw measurements."""
    arch = r.arch
    B = r.batch_size

    # ── Throughput ─────────────────────────────────────────────────────────
    total_in  = sum(r.input_tok)
    total_out = sum(r.output_tok)
    total_gen_s = sum(r.gen_s) or 1e-9
    total_e2e_s = sum(r.e2e_s) or 1e-9

    gen_tps   = (total_in + total_out) / total_gen_s
    queries_s = r.n_samples / total_e2e_s
    prefill_tps = total_in / (sum(r.ttft_s) or 1e-9)
    decode_tps  = total_out / (sum(r.decode_s) or 1e-9)

    # ── Phase-level roofline ──────────────────────────────────────────────
    avg_pf_flops = _safe(r.prefill_flops)
    avg_pf_hbm   = _safe(r.prefill_hbm_B)
    avg_dc_flops = _safe(r.decode_flops)
    avg_dc_hbm   = _safe(r.decode_hbm_B)

    prefill_roofline = roofline(avg_pf_flops, avg_pf_hbm)
    decode_roofline  = roofline(avg_dc_flops, avg_dc_hbm)

    # ── MFU ───────────────────────────────────────────────────────────────
    avg_ttft = _safe(r.ttft_s)
    avg_dec  = _safe(r.decode_s)

    prefill_achieved_tflops = (avg_pf_flops / avg_ttft / 1e12) if avg_ttft > 0 else 0
    decode_achieved_tflops  = (avg_dc_flops / avg_dec / 1e12) if avg_dec > 0 else 0

    prefill_mfu = compute_mfu(prefill_achieved_tflops)
    decode_mfu  = compute_mfu(decode_achieved_tflops)

    # ── Decode bandwidth ──────────────────────────────────────────────────
    avg_out_tok = _safe(r.output_tok, default=1)
    dec_s_per_tok = avg_dec / max(1, avg_out_tok) if avg_dec > 0 else 1e-9
    if r.use_kv:
        avg_seq = _safe(r.input_tok, default=512)
        kv_b = kv_cache_bytes(arch.get("nl", 32), arch.get("nkv", 32),
                              arch.get("dh", 128), int(avg_seq), B)
        bytes_per_step = r.model_size_GB * 1e9 + kv_b
    else:
        bytes_per_step = r.model_size_GB * 1e9
    decode_bw_GBs = bytes_per_step / dec_s_per_tok / 1e9 if dec_s_per_tok > 0 else 0
    decode_bw_pct = 100 * decode_bw_GBs / HW["hbm_bw_GBs"]

    # ── Energy ────────────────────────────────────────────────────────────
    total_energy = sum(r.energy_J) if r.energy_J else 0
    energy_per_query = total_energy / r.n_samples if r.n_samples > 0 else 0

    # ── E2E time fractions ────────────────────────────────────────────────
    retr_frac = 100 * sum(r.retrieval_s) / total_e2e_s
    prefill_frac = 100 * sum(r.ttft_s) / total_e2e_s
    decode_frac = 100 * sum(r.decode_s) / total_e2e_s

    # ── KV cache pressure ─────────────────────────────────────────────────
    avg_seq_int = int(_safe(r.input_tok, default=512))
    kv_size_GB = kv_cache_bytes(
        arch.get("nl", 32), arch.get("nkv", 32), arch.get("dh", 128),
        avg_seq_int, B) / 1e9 if r.use_kv else 0.0

    return {
        "label":       _LABELS.get(r.config, r.config),
        "model":       r.model_name,
        "model_B":     r.model_params / 1e9,
        "model_GB":    r.model_size_GB,
        "dataset":     r.dataset_name,
        "n_samples":   r.n_samples,
        "use_kv":      r.use_kv,
        "hop":         r.hop_constraint,
        "top_k":       r.top_k,
        # Quality
        "hits_at_1":   r.hits_at_1,
        "f1":          r.f1,
        # Latency (ms)
        "retrieval_ms":    _pct(r.retrieval_s),
        "ttft_ms":         _pct(r.ttft_s),
        "decode_ms":       _pct(r.decode_s),
        "gen_ms":          _pct(r.gen_s),
        "e2e_ms":          _pct(r.e2e_s),
        # Throughput
        "gen_tok_per_s":   gen_tps,
        "queries_per_s":   queries_s,
        "prefill_tok_s":   prefill_tps,
        "decode_tok_s":    decode_tps,
        # Token stats
        "avg_input_tok":   _safe(r.input_tok),
        "avg_output_tok":  _safe(r.output_tok),
        "avg_ctx_tok":     _safe(r.ctx_tok),
        "avg_retrieved":   _safe(r.n_retrieved),
        # Phase-level roofline
        "prefill_roofline": prefill_roofline,
        "decode_roofline":  decode_roofline,
        # MFU
        "prefill_mfu":         prefill_mfu,
        "decode_mfu":          decode_mfu,
        "prefill_tflops":      prefill_achieved_tflops,
        "decode_tflops":       decode_achieved_tflops,
        # Decode bandwidth
        "decode_bw_GBs":       decode_bw_GBs,
        "decode_bw_pct_peak":  decode_bw_pct,
        # Memory
        "peak_hbm_GB":     _safe(r.hbm_peak_B) / 1e9,
        "kv_cache_GB":     kv_size_GB,
        "model_footprint_GB": r.model_size_GB,
        # Energy
        "total_energy_J":  total_energy,
        "energy_per_query_J": energy_per_query,
        "mean_power_W":    _safe(r.power_W),
        # Utilisation
        "mean_sm_pct":     _safe(r.sm_pct),
        # Time fractions
        "retrieval_pct":   retr_frac,
        "prefill_pct":     prefill_frac,
        "decode_pct":      decode_frac,
        # GraphRAG graph characteristics
        "avg_paths_per_query": _safe(r.paths_per_query) if r.paths_per_query else None,
        "avg_graph_nodes":     _safe(r.graph_nodes) if r.graph_nodes else None,
        "avg_graph_edges":     _safe(r.graph_edges) if r.graph_edges else None,
        "avg_graph_diameter":  _safe(r.graph_diameter) if r.graph_diameter else None,
        "avg_graph_longest_path": _safe(r.graph_longest_path) if r.graph_longest_path else None,
        "max_graph_longest_path": float(max(r.graph_longest_path)) if r.graph_longest_path else None,
        "avg_graph_avg_degree":   _safe(r.graph_avg_degree) if r.graph_avg_degree else None,
        "avg_graph_max_degree":   _safe(r.graph_max_degree) if r.graph_max_degree else None,
        "max_graph_max_degree":   float(max(r.graph_max_degree)) if r.graph_max_degree else None,
        "avg_graph_density":      _safe(r.graph_density) if r.graph_density else None,
        "avg_graph_components":   _safe(r.graph_components) if r.graph_components else None,
        "avg_graph_n_relations":  _safe(r.graph_n_relations) if r.graph_n_relations else None,
        "avg_per_query_path_len": _safe(r.per_query_path_len) if r.per_query_path_len else None,
        "avg_per_query_retr_ms":  _safe(r.per_query_retrieval_s) * 1e3 if r.per_query_retrieval_s else None,
        # Per-query arrays for correlation analysis
        "graph_nodes_arr":         r.graph_nodes if r.graph_nodes else None,
        "graph_edges_arr":         r.graph_edges if r.graph_edges else None,
        "graph_diameter_arr":      r.graph_diameter if r.graph_diameter else None,
        "per_query_retrieval_ms_arr": [s * 1e3 for s in r.per_query_retrieval_s] if r.per_query_retrieval_s else None,
        # Offline
        "model_load_s":    r.model_load_s,
        "index_build_s":   r.index_build_s,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# §11  Experiment runners
# ═══════════════════════════════════════════════════════════════════════════════

def _build_rag_retriever(dataset) -> Optional[RAGRetrieval]:
    if not _RAG_DEPS:
        print("[WARN] faiss / sentence-transformers not installed. Skipping RAG configs.")
        return None
    ret = RAGRetrieval()
    ret.build_global_index(dataset)
    return ret


def _run_4configs(experiment: str, dataset, dataset_name: str,
                  model_name: str, batch_size: int, num_beams: int,
                  max_new_tokens: int, seed: int, hop: int, top_k: int,
                  device: int, configs: List[str]) -> List[ProfileResult]:
    """Run up to 4 configs (rag, rag_kv, graphrag, graphrag_kv)."""

    rag_retriever = None
    if any(c.startswith("rag") for c in configs):
        rag_retriever = _build_rag_retriever(dataset)
        if rag_retriever is None:
            configs = [c for c in configs if not c.startswith("rag")]

    results = []
    for cfg in configs:
        try:
            r = run_single(
                experiment=experiment, config=cfg,
                dataset=dataset, dataset_name=dataset_name,
                model_name=model_name, batch_size=batch_size,
                num_beams=num_beams, max_new_tokens=max_new_tokens,
                seed=seed, hop=hop, top_k=top_k,
                rag_retriever=rag_retriever, device=device,
            )
            results.append(r)
        except Exception:
            print(f"  [ERROR] {cfg} failed:")
            traceback.print_exc()
        torch.cuda.empty_cache()
        gc.collect()

    return results


# ── Experiment 1: Model sweep ─────────────────────────────────────────────────

def run_model_sweep(
    models: List[str], dataset_name: str, samples: int,
    batch_size: int, num_beams: int, max_tokens: int,
    seed: int, hop: int, top_k: int, device: int,
) -> List[ProfileResult]:
    """Sweep across models; for each model run all 4 configs."""
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: MODEL SWEEP  ({len(models)} models)")
    print(f"{'='*70}")

    dataset = load_ds(dataset_name, "test", samples, seed)
    all_results = []

    for model in models:
        print(f"\n  ── Model: {model} ──")
        configs = ["rag", "rag_kv", "graphrag", "graphrag_kv"]
        rs = _run_4configs(
            "model_sweep", dataset, dataset_name, model,
            batch_size, num_beams, max_tokens, seed, hop, top_k, device,
            configs,
        )
        all_results.extend(rs)

    return all_results


# ── Experiment 2: Depth sweep ─────────────────────────────────────────────────

def run_depth_sweep(
    model_name: str, dataset_name: str, samples: int,
    batch_size: int, num_beams: int, max_tokens: int,
    seed: int, hops: List[int], top_k: int, device: int,
) -> List[ProfileResult]:
    """Sweep hop_constraint for GraphRAG; RAG is depth-independent (run once)."""
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: DEPTH SWEEP  (hops={hops})")
    print(f"{'='*70}")

    dataset = load_ds(dataset_name, "test", samples, seed)

    # RAG is independent of graph depth — run once as baseline
    rag_retriever = _build_rag_retriever(dataset)
    all_results = []

    if rag_retriever is not None:
        for cfg in ["rag", "rag_kv"]:
            try:
                r = run_single(
                    experiment="depth_sweep", config=cfg,
                    dataset=dataset, dataset_name=dataset_name,
                    model_name=model_name, batch_size=batch_size,
                    num_beams=num_beams, max_new_tokens=max_tokens,
                    seed=seed, hop=max(hops), top_k=top_k,
                    rag_retriever=rag_retriever, device=device,
                )
                # Tag with hop=0 to indicate depth-independent
                r.hop_constraint = 0
                all_results.append(r)
            except Exception:
                traceback.print_exc()
            torch.cuda.empty_cache()
            gc.collect()

    # GraphRAG: sweep hops
    for h in hops:
        print(f"\n  ── hop_constraint = {h} ──")
        for cfg in ["graphrag", "graphrag_kv"]:
            try:
                r = run_single(
                    experiment="depth_sweep", config=cfg,
                    dataset=dataset, dataset_name=dataset_name,
                    model_name=model_name, batch_size=batch_size,
                    num_beams=num_beams, max_new_tokens=max_tokens,
                    seed=seed, hop=h, top_k=top_k,
                    rag_retriever=None, device=device,
                )
                all_results.append(r)
            except Exception:
                traceback.print_exc()
            torch.cuda.empty_cache()
            gc.collect()

    return all_results


# ── Experiment 3: Batch size sweep ────────────────────────────────────────────

def run_batch_sweep(
    model_name: str, dataset_name: str, samples: int,
    batch_sizes: List[int], num_beams: int, max_tokens: int,
    seed: int, hop: int, top_k: int, device: int,
) -> List[ProfileResult]:
    """Sweep batch size; show throughput & memory scaling."""
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: BATCH SIZE SWEEP  (bs={batch_sizes})")
    print(f"{'='*70}")

    dataset = load_ds(dataset_name, "test", samples, seed)
    all_results = []

    for bs in batch_sizes:
        print(f"\n  ── batch_size = {bs} ──")
        configs = ["rag", "rag_kv", "graphrag", "graphrag_kv"]
        rs = _run_4configs(
            "batch_sweep", dataset, dataset_name, model_name,
            bs, num_beams, max_tokens, seed, hop, top_k, device,
            configs,
        )
        all_results.extend(rs)

    return all_results


# ── Experiment 4: Context sweep (RAG top_k vs GraphRAG hops) ─────────────────

def run_context_sweep(
    model_name: str, dataset_name: str, samples: int,
    batch_size: int, num_beams: int, max_tokens: int,
    seed: int, device: int,
) -> List[ProfileResult]:
    """Sweep RAG top_k and GraphRAG hops to show context length impact."""
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: CONTEXT SWEEP  (RAG top_k vs GraphRAG hops)")
    print(f"{'='*70}")

    dataset = load_ds(dataset_name, "test", samples, seed)
    rag_retriever = _build_rag_retriever(dataset)
    all_results = []

    # RAG: sweep top_k = 1, 3, 5, 10, 20
    if rag_retriever is not None:
        for tk in [1, 3, 5, 10, 20]:
            print(f"\n  ── RAG top_k = {tk} ──")
            for cfg in ["rag", "rag_kv"]:
                try:
                    r = run_single(
                        experiment="context_sweep", config=cfg,
                        dataset=dataset, dataset_name=dataset_name,
                        model_name=model_name, batch_size=batch_size,
                        num_beams=num_beams, max_new_tokens=max_tokens,
                        seed=seed, hop=3, top_k=tk,
                        rag_retriever=rag_retriever, device=device,
                    )
                    all_results.append(r)
                except Exception:
                    traceback.print_exc()
                torch.cuda.empty_cache()
                gc.collect()

    # GraphRAG: sweep hops 1, 2, 3, 4
    for h in [1, 2, 3, 4]:
        print(f"\n  ── GraphRAG hop = {h} ──")
        for cfg in ["graphrag", "graphrag_kv"]:
            try:
                r = run_single(
                    experiment="context_sweep", config=cfg,
                    dataset=dataset, dataset_name=dataset_name,
                    model_name=model_name, batch_size=batch_size,
                    num_beams=num_beams, max_new_tokens=max_tokens,
                    seed=seed, hop=h, top_k=5,
                    rag_retriever=None, device=device,
                )
                all_results.append(r)
            except Exception:
                traceback.print_exc()
            torch.cuda.empty_cache()
            gc.collect()

    return all_results


# ── Experiment 5: Dataset sweep ───────────────────────────────────────────────

def run_graph_char(
    model_name: str, dataset_name: str, samples: int,
    batch_size: int, num_beams: int, max_tokens: int,
    seed: int, hops: List[int], top_k: int, device: int,
) -> List[ProfileResult]:
    """
    Graph characteristics experiment.

    Measures how retrieval time changes with graph structural properties
    (nodes, edges, diameter, longest path, degree distribution, density,
    connected components, unique relation types) across different hop depths.
    Runs GraphRAG configs at each hop depth and RAG once as baseline.
    Prints per-query graph statistics correlated with retrieval latency.
    """
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: GRAPH CHARACTERISTICS  (hops={hops})")
    print(f"  How retrieval time changes with graph structure")
    print(f"{'='*70}")

    dataset = load_ds(dataset_name, "test", samples, seed)

    # RAG baseline (independent of graph structure — run once)
    rag_retriever = _build_rag_retriever(dataset)
    all_results = []

    if rag_retriever is not None:
        for cfg in ["rag", "rag_kv"]:
            try:
                r = run_single(
                    experiment="graph_char", config=cfg,
                    dataset=dataset, dataset_name=dataset_name,
                    model_name=model_name, batch_size=batch_size,
                    num_beams=num_beams, max_new_tokens=max_tokens,
                    seed=seed, hop=max(hops), top_k=top_k,
                    rag_retriever=rag_retriever, device=device,
                )
                r.hop_constraint = 0  # tag as depth-independent
                all_results.append(r)
            except Exception:
                traceback.print_exc()
            torch.cuda.empty_cache()
            gc.collect()

    # GraphRAG at each hop depth
    for h in hops:
        print(f"\n  ── hop_constraint = {h} ──")
        for cfg in ["graphrag", "graphrag_kv"]:
            try:
                r = run_single(
                    experiment="graph_char", config=cfg,
                    dataset=dataset, dataset_name=dataset_name,
                    model_name=model_name, batch_size=batch_size,
                    num_beams=num_beams, max_new_tokens=max_tokens,
                    seed=seed, hop=h, top_k=top_k,
                    rag_retriever=None, device=device,
                )
                all_results.append(r)
            except Exception:
                traceback.print_exc()
            torch.cuda.empty_cache()
            gc.collect()

    return all_results


def print_graph_char_report(results: List[ProfileResult]):
    """
    Print detailed graph characteristics report showing how retrieval
    time correlates with graph structural properties.
    """
    # Only consider GraphRAG results (they have graph stats)
    grag_results = [r for r in results if r.graph_nodes]
    if not grag_results:
        print("  [No GraphRAG results with graph stats to report]")
        return

    print(f"\n{'='*90}")
    print(f"  GRAPH CHARACTERISTICS vs RETRIEVAL TIME")
    print(f"{'='*90}")

    # ── Aggregate stats per hop depth ─────────────────────────────────────
    hop_groups: Dict[int, List[ProfileResult]] = defaultdict(list)
    for r in grag_results:
        hop_groups[r.hop_constraint].append(r)

    sep = "-" * 90
    print(f"\n  PER-HOP-DEPTH GRAPH STRUCTURE SUMMARY")
    print(sep)
    print(f"  {'hop':<5} {'kv':>4} {'nodes':>8} {'edges':>8} {'diam':>6} "
          f"{'longest':>8} {'avg_deg':>8} {'max_deg':>8} {'density':>8} "
          f"{'comps':>6} {'rels':>6}")
    print(sep)

    for h in sorted(hop_groups.keys()):
        for r in hop_groups[h]:
            kv = "ON" if r.use_kv else "OFF"
            print(f"  {h:<5} {kv:>4} "
                  f"{_safe(r.graph_nodes):>8.1f} "
                  f"{_safe(r.graph_edges):>8.1f} "
                  f"{_safe(r.graph_diameter):>6.1f} "
                  f"{_safe(r.graph_longest_path):>8.1f} "
                  f"{_safe(r.graph_avg_degree):>8.2f} "
                  f"{_safe(r.graph_max_degree):>8.1f} "
                  f"{_safe(r.graph_density):>8.4f} "
                  f"{_safe(r.graph_components):>6.1f} "
                  f"{_safe(r.graph_n_relations):>6.1f}")
    print(sep)

    # ── Retrieval time vs graph characteristics ───────────────────────────
    print(f"\n  RETRIEVAL TIME vs GRAPH STRUCTURE")
    print(sep)
    print(f"  {'hop':<5} {'kv':>4} {'retr_ms':>10} {'paths/q':>8} "
          f"{'avg_hops':>9} {'ttft_ms':>9} {'e2e_ms':>9} "
          f"{'retr_%':>7} {'prefill_%':>10}")
    print(sep)

    for h in sorted(hop_groups.keys()):
        for r in hop_groups[h]:
            m = derive_metrics(r)
            kv = "ON" if r.use_kv else "OFF"
            print(f"  {h:<5} {kv:>4} "
                  f"{m['retrieval_ms']['mean']:>10.1f} "
                  f"{m['avg_paths_per_query'] or 0:>8.1f} "
                  f"{m['avg_per_query_path_len'] or 0:>9.2f} "
                  f"{m['ttft_ms']['mean']:>9.1f} "
                  f"{m['e2e_ms']['mean']:>9.1f} "
                  f"{m['retrieval_pct']:>7.1f} "
                  f"{m['prefill_pct']:>10.1f}")

    # ── RAG baseline for comparison ───────────────────────────────────────
    rag_results = [r for r in results if r.config.startswith("rag")]
    if rag_results:
        print(f"\n  RAG BASELINE (graph-structure independent)")
        print(sep)
        for r in rag_results:
            m = derive_metrics(r)
            kv = "ON" if r.use_kv else "OFF"
            print(f"  RAG  {kv:>4} "
                  f"retr={m['retrieval_ms']['mean']:.1f}ms  "
                  f"ttft={m['ttft_ms']['mean']:.1f}ms  "
                  f"e2e={m['e2e_ms']['mean']:.1f}ms  "
                  f"retr%={m['retrieval_pct']:.1f}%")

    # ── Scaling analysis: how metrics grow with hop depth ─────────────────
    # Use KV=ON results for the scaling analysis (most representative)
    kv_on = [r for r in grag_results if r.use_kv]
    if len(kv_on) >= 2:
        kv_on_sorted = sorted(kv_on, key=lambda r: r.hop_constraint)
        base = kv_on_sorted[0]
        base_m = derive_metrics(base)
        base_retr = base_m["retrieval_ms"]["mean"]
        base_paths = base_m["avg_paths_per_query"] or 1
        base_nodes = _safe(base.graph_nodes) or 1

        print(f"\n  SCALING vs hop=1 BASELINE  (GraphRAG KV=ON)")
        print(sep)
        print(f"  {'hop':<5} {'retr_x':>8} {'paths_x':>8} {'nodes_x':>8} "
              f"{'edges_x':>8} {'diam_x':>8} {'e2e_x':>8}")
        print(sep)
        for r in kv_on_sorted:
            m = derive_metrics(r)
            retr_x = m["retrieval_ms"]["mean"] / max(0.001, base_retr)
            paths_x = (m["avg_paths_per_query"] or 1) / base_paths
            nodes_x = _safe(r.graph_nodes) / base_nodes
            edges_x = _safe(r.graph_edges) / max(1, _safe(base.graph_edges))
            diam_x = _safe(r.graph_diameter) / max(1, _safe(base.graph_diameter))
            e2e_x = m["e2e_ms"]["mean"] / max(0.001, base_m["e2e_ms"]["mean"])
            print(f"  {r.hop_constraint:<5} "
                  f"{retr_x:>8.2f} {paths_x:>8.2f} {nodes_x:>8.2f} "
                  f"{edges_x:>8.2f} {diam_x:>8.2f} {e2e_x:>8.2f}")

    # ── Distribution tails (p50, p95, max) for graph properties ───────────
    print(f"\n  GRAPH PROPERTY DISTRIBUTION  (across all queries)")
    print(sep)
    for r in grag_results:
        kv = "ON" if r.use_kv else "OFF"
        label = f"hop={r.hop_constraint} KV={kv}"
        print(f"\n  {label}")
        for name, arr in [
            ("nodes",     r.graph_nodes),
            ("edges",     r.graph_edges),
            ("diameter",  r.graph_diameter),
            ("longest_path", r.graph_longest_path),
            ("max_degree",   r.graph_max_degree),
            ("n_relations",  r.graph_n_relations),
            ("paths_found",  r.paths_per_query),
        ]:
            if arr:
                a = np.array(arr, dtype=float)
                print(f"    {name:<16}  "
                      f"min={np.min(a):.0f}  "
                      f"p50={np.median(a):.0f}  "
                      f"mean={np.mean(a):.1f}  "
                      f"p95={np.percentile(a, 95):.0f}  "
                      f"max={np.max(a):.0f}")

    print(f"\n{'='*90}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# §12  Reporting
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt(v, f=".2f"):
    if v is None:
        return "    —"
    if isinstance(v, dict):
        v = v.get("mean", 0)
    return format(v, f">{10}{f}")


def print_comparison_table(results: List[ProfileResult], title: str):
    """Print a compact comparison table for a set of results."""
    if not results:
        return

    metrics = [derive_metrics(r) for r in results]

    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")

    # Header
    labels = [m["label"] for m in metrics]
    extra = ""
    if results[0].experiment == "depth_sweep":
        extra = "  hop"
        labels = [f"{m['label']}(h={r.hop_constraint})" for m, r in zip(metrics, results)]
    elif results[0].experiment == "batch_sweep":
        labels = [f"{m['label']}(B={r.batch_size})" for m, r in zip(metrics, results)]

    w = max(16, max(len(l) for l in labels) + 2)
    hdr = f"  {'Metric':<36}" + "".join(f"{l:>{w}}" for l in labels)
    sep = "-" * (38 + w * len(labels))
    print(f"\n{sep}\n{hdr}\n{sep}")

    def row(name, vals, fmt=".3f"):
        s = f"  {name:<36}"
        for v in vals:
            s += _fmt(v, fmt)
        print(s)

    # Quality
    print("  QUALITY")
    row("  Hits@1", [m["hits_at_1"] for m in metrics], ".4f")
    row("  F1",     [m["f1"]        for m in metrics], ".4f")

    # Latency
    print("  LATENCY (ms per batch)")
    row("  Retrieval mean",  [m["retrieval_ms"]["mean"] for m in metrics], ".1f")
    row("  TTFT/Prefill mean", [m["ttft_ms"]["mean"] for m in metrics], ".1f")
    row("  Decode mean",     [m["decode_ms"]["mean"]    for m in metrics], ".1f")
    row("  E2E mean",        [m["e2e_ms"]["mean"]       for m in metrics], ".1f")
    row("  E2E p95",         [m["e2e_ms"]["p95"]        for m in metrics], ".1f")

    # Throughput
    print("  THROUGHPUT")
    row("  Gen tok/s",     [m["gen_tok_per_s"]  for m in metrics], ".0f")
    row("  Queries/s",     [m["queries_per_s"]  for m in metrics], ".2f")
    row("  Prefill tok/s", [m["prefill_tok_s"]  for m in metrics], ".0f")
    row("  Decode tok/s",  [m["decode_tok_s"]   for m in metrics], ".0f")

    # Phase-level hardware characterisation
    print("  PHASE-LEVEL HARDWARE")
    row("  Prefill arith. intensity (F/B)",
        [m["prefill_roofline"]["arith_intensity"] for m in metrics], ".0f")
    print(f"  {'Prefill bound':<36}" +
          "".join(f"{'memory':>{w}}" if m["prefill_roofline"]["memory_bound"]
                  else f"{'compute':>{w}}" for m in metrics))
    row("  Prefill MFU (%)",
        [100 * m["prefill_mfu"] for m in metrics], ".1f")
    row("  Prefill TFLOP/s",
        [m["prefill_tflops"] for m in metrics], ".1f")

    row("  Decode arith. intensity (F/B)",
        [m["decode_roofline"]["arith_intensity"] for m in metrics], ".1f")
    print(f"  {'Decode bound':<36}" +
          "".join(f"{'memory':>{w}}" if m["decode_roofline"]["memory_bound"]
                  else f"{'compute':>{w}}" for m in metrics))
    row("  Decode MFU (%)",
        [100 * m["decode_mfu"] for m in metrics], ".1f")
    row("  Decode BW (GB/s)",
        [m["decode_bw_GBs"] for m in metrics], ".0f")
    row("  Decode BW % peak",
        [m["decode_bw_pct_peak"] for m in metrics], ".1f")

    # Memory
    print("  MEMORY")
    row("  Model footprint (GB)", [m["model_footprint_GB"] for m in metrics], ".2f")
    row("  Peak HBM (GB)",        [m["peak_hbm_GB"]        for m in metrics], ".2f")
    row("  KV cache est (GB)",    [m["kv_cache_GB"]        for m in metrics], ".3f")

    # Energy
    if any(m["mean_power_W"] > 0 for m in metrics):
        print("  ENERGY")
        row("  Mean power (W)",      [m["mean_power_W"]       for m in metrics], ".0f")
        row("  Energy/query (J)",    [m["energy_per_query_J"] for m in metrics], ".1f")
        row("  SM utilisation (%)",  [m["mean_sm_pct"]        for m in metrics], ".0f")

    # Token distribution
    print("  TOKEN DISTRIBUTION")
    row("  Avg prompt tokens", [m["avg_input_tok"]  for m in metrics], ".0f")
    row("  Avg context tokens",[m["avg_ctx_tok"]    for m in metrics], ".0f")
    row("  Avg output tokens", [m["avg_output_tok"] for m in metrics], ".0f")
    row("  Avg items retrieved",[m["avg_retrieved"] for m in metrics], ".1f")

    # Time fractions
    print("  E2E TIME FRACTIONS (%)")
    row("  Retrieval %", [m["retrieval_pct"] for m in metrics], ".1f")
    row("  Prefill %",   [m["prefill_pct"]   for m in metrics], ".1f")
    row("  Decode %",    [m["decode_pct"]    for m in metrics], ".1f")

    # GraphRAG graph characteristics
    if any(m["avg_paths_per_query"] is not None for m in metrics):
        print("  GRAPH CHARACTERISTICS")
        row("  Paths/query",
            [m["avg_paths_per_query"] for m in metrics], ".1f")
        row("  Avg hops/path",
            [m["avg_per_query_path_len"] for m in metrics], ".2f")
        row("  Subgraph nodes",
            [m["avg_graph_nodes"] for m in metrics], ".1f")
        row("  Subgraph edges",
            [m["avg_graph_edges"] for m in metrics], ".1f")
        row("  Graph diameter",
            [m["avg_graph_diameter"] for m in metrics], ".1f")
        row("  Longest path (hops)",
            [m["avg_graph_longest_path"] for m in metrics], ".1f")
        row("  Avg degree",
            [m["avg_graph_avg_degree"] for m in metrics], ".2f")
        row("  Max degree",
            [m["avg_graph_max_degree"] for m in metrics], ".1f")
        row("  Density",
            [m["avg_graph_density"] for m in metrics], ".4f")
        row("  Connected components",
            [m["avg_graph_components"] for m in metrics], ".1f")
        row("  Unique relations",
            [m["avg_graph_n_relations"] for m in metrics], ".1f")

    # KV cache speedup
    kv_off = [m for m in metrics if not m["use_kv"]]
    kv_on  = [m for m in metrics if m["use_kv"]]
    if kv_off and kv_on:
        print("  KV CACHE SPEEDUP (E2E)")
        for off in kv_off:
            matching_on = [o for o in kv_on
                           if off["label"].replace("KV=OFF", "KV=ON") == o["label"]]
            if matching_on:
                on = matching_on[0]
                off_e2e = off["e2e_ms"]["mean"]
                on_e2e = on["e2e_ms"]["mean"]
                speedup = off_e2e / on_e2e if on_e2e > 0 else 0
                base = off["label"].replace(" (KV=OFF)", "")
                print(f"    {base}: {speedup:.2f}x  "
                      f"(KV=OFF {off_e2e:.0f}ms → KV=ON {on_e2e:.0f}ms)")

    print(f"{'='*90}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# §13  JSON output
# ═══════════════════════════════════════════════════════════════════════════════

def save_results(all_results: List[ProfileResult], out_dir: Path, tag: str):
    """Save all results as JSON for post-processing / plotting."""
    out_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "hw": HW,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        "experiments": {},
    }

    for r in all_results:
        exp = r.experiment
        if exp not in data["experiments"]:
            data["experiments"][exp] = []
        data["experiments"][exp].append(derive_metrics(r))

    path = out_dir / f"{tag}_profile.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"[Output] JSON → {path}")

    # Also save a flat CSV for quick plotting
    csv_path = out_dir / f"{tag}_profile.csv"
    _save_flat_csv(all_results, csv_path)
    print(f"[Output] CSV  → {csv_path}")


def _save_flat_csv(results: List[ProfileResult], path: Path):
    """Flat CSV: one row per (experiment, config, model, dataset, params) combo."""
    cols = [
        "experiment", "config", "model", "dataset", "use_kv",
        "hop", "top_k", "batch_size", "n_samples",
        "model_B", "model_GB",
        "hits_at_1", "f1",
        "retrieval_ms_mean", "ttft_ms_mean", "decode_ms_mean",
        "e2e_ms_mean", "e2e_ms_p95",
        "gen_tok_per_s", "queries_per_s",
        "prefill_tok_s", "decode_tok_s",
        "prefill_arith_intensity", "decode_arith_intensity",
        "prefill_mfu", "decode_mfu",
        "decode_bw_GBs", "decode_bw_pct_peak",
        "peak_hbm_GB", "kv_cache_GB",
        "energy_per_query_J", "mean_power_W", "mean_sm_pct",
        "retrieval_pct", "prefill_pct", "decode_pct",
        "avg_input_tok", "avg_ctx_tok", "avg_output_tok",
        "avg_paths_per_query", "avg_graph_nodes", "avg_graph_edges",
        "avg_graph_diameter", "avg_graph_longest_path",
        "avg_graph_avg_degree", "avg_graph_max_degree",
        "avg_graph_density", "avg_graph_components", "avg_graph_n_relations",
        "avg_per_query_path_len",
    ]

    rows = []
    for r in results:
        m = derive_metrics(r)
        row = {
            "experiment": r.experiment,
            "config": r.config,
            "model": r.model_name,
            "dataset": r.dataset_name,
            "use_kv": int(r.use_kv),
            "hop": r.hop_constraint,
            "top_k": r.top_k,
            "batch_size": r.batch_size,
            "n_samples": r.n_samples,
            "model_B": f"{m['model_B']:.2f}",
            "model_GB": f"{m['model_GB']:.2f}",
            "hits_at_1": f"{m['hits_at_1']:.4f}",
            "f1": f"{m['f1']:.4f}",
            "retrieval_ms_mean": f"{m['retrieval_ms']['mean']:.2f}",
            "ttft_ms_mean": f"{m['ttft_ms']['mean']:.2f}",
            "decode_ms_mean": f"{m['decode_ms']['mean']:.2f}",
            "e2e_ms_mean": f"{m['e2e_ms']['mean']:.2f}",
            "e2e_ms_p95": f"{m['e2e_ms']['p95']:.2f}",
            "gen_tok_per_s": f"{m['gen_tok_per_s']:.1f}",
            "queries_per_s": f"{m['queries_per_s']:.3f}",
            "prefill_tok_s": f"{m['prefill_tok_s']:.1f}",
            "decode_tok_s": f"{m['decode_tok_s']:.1f}",
            "prefill_arith_intensity": f"{m['prefill_roofline']['arith_intensity']:.1f}",
            "decode_arith_intensity": f"{m['decode_roofline']['arith_intensity']:.1f}",
            "prefill_mfu": f"{m['prefill_mfu']:.4f}",
            "decode_mfu": f"{m['decode_mfu']:.4f}",
            "decode_bw_GBs": f"{m['decode_bw_GBs']:.1f}",
            "decode_bw_pct_peak": f"{m['decode_bw_pct_peak']:.1f}",
            "peak_hbm_GB": f"{m['peak_hbm_GB']:.3f}",
            "kv_cache_GB": f"{m['kv_cache_GB']:.3f}",
            "energy_per_query_J": f"{m['energy_per_query_J']:.2f}",
            "mean_power_W": f"{m['mean_power_W']:.1f}",
            "mean_sm_pct": f"{m['mean_sm_pct']:.1f}",
            "retrieval_pct": f"{m['retrieval_pct']:.1f}",
            "prefill_pct": f"{m['prefill_pct']:.1f}",
            "decode_pct": f"{m['decode_pct']:.1f}",
            "avg_input_tok": f"{m['avg_input_tok']:.0f}",
            "avg_ctx_tok": f"{m['avg_ctx_tok']:.0f}",
            "avg_output_tok": f"{m['avg_output_tok']:.0f}",
            "avg_paths_per_query": f"{m['avg_paths_per_query']:.1f}" if m['avg_paths_per_query'] else "",
            "avg_graph_nodes": f"{m['avg_graph_nodes']:.1f}" if m['avg_graph_nodes'] else "",
            "avg_graph_edges": f"{m['avg_graph_edges']:.1f}" if m['avg_graph_edges'] else "",
            "avg_graph_diameter": f"{m['avg_graph_diameter']:.1f}" if m['avg_graph_diameter'] else "",
            "avg_graph_longest_path": f"{m['avg_graph_longest_path']:.1f}" if m['avg_graph_longest_path'] else "",
            "avg_graph_avg_degree": f"{m['avg_graph_avg_degree']:.2f}" if m['avg_graph_avg_degree'] else "",
            "avg_graph_max_degree": f"{m['avg_graph_max_degree']:.1f}" if m['avg_graph_max_degree'] else "",
            "avg_graph_density": f"{m['avg_graph_density']:.4f}" if m['avg_graph_density'] else "",
            "avg_graph_components": f"{m['avg_graph_components']:.1f}" if m['avg_graph_components'] else "",
            "avg_graph_n_relations": f"{m['avg_graph_n_relations']:.1f}" if m['avg_graph_n_relations'] else "",
            "avg_per_query_path_len": f"{m['avg_per_query_path_len']:.2f}" if m['avg_per_query_path_len'] else "",
        }
        rows.append(row)

    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(c, "")) for c in cols) + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# §14  Presets
# ═══════════════════════════════════════════════════════════════════════════════

PRESETS = {
    "smoke": {
        "description": "Quick sanity check (~5-10 min)",
        "models": ["rmanluo/RoG", "google/gemma-2-2b"],
        "experiments": ["model"],
        "samples": 16, "batch_size": 8,
    },
    "quick": {
        "description": "Fast development run (~30 min)",
        "models": ["rmanluo/RoG", "facebook/opt-6.7b"],
        "experiments": ["model", "depth"],
        "samples": 32, "batch_size": 8,
    },
    "medium": {
        "description": "Medium validation (~2-4 hrs)",
        "models": MODELS["small"] + MODELS["medium"][:1],
        "experiments": ["model", "depth", "batch"],
        "samples": 64, "batch_size": 16,
    },
    "isca": {
        "description": "Full ISCA suite — all experiments, all models (~12+ hrs)",
        "models": ALL_MODELS,
        "experiments": ["model", "depth", "batch", "context", "graph_char"],
        "samples": 64, "batch_size": 16,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# §15  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        description="RAG vs GraphRAG Deep Profiling (ISCA paper)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--preset", choices=list(PRESETS.keys()), default=None,
                   help="Preset: " + " | ".join(
                       f"{k} ({v['description']})" for k, v in PRESETS.items()))
    p.add_argument("--experiments", nargs="+",
                   choices=["model", "depth", "batch", "context", "graph_char"],
                   default=None,
                   help="Which experiments to run (default: all from preset)")

    # Model
    p.add_argument("--llm_model", default=None,
                   help="Single model (overrides preset model list)")
    p.add_argument("--models", nargs="+", default=None,
                   help="Explicit model list (overrides preset)")

    # Dataset
    p.add_argument("--dataset", default="rmanluo/RoG-cwq")
    # --datasets removed: use --dataset to specify one dataset

    # Size
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--max_tokens", type=int, default=50)

    # Sweep params
    p.add_argument("--hops", nargs="+", type=int, default=[1, 2, 3, 4],
                   help="Hop values for depth sweep (default: 1 2 3 4)")
    p.add_argument("--batch_sizes", nargs="+", type=int,
                   default=[1, 2, 4, 8, 16, 32],
                   help="Batch sizes for batch sweep")
    p.add_argument("--hop", type=int, default=3,
                   help="Default hop constraint (for non-sweep experiments)")
    p.add_argument("--top_k", type=int, default=5,
                   help="RAG top-k passages (for non-sweep experiments)")

    # Output
    p.add_argument("--out_dir", default="profile_results")
    p.add_argument("--device", type=int, default=0)

    return p


def main():
    args = build_parser().parse_args()

    # ── Resolve preset ────────────────────────────────────────────────────
    if args.preset:
        preset = PRESETS[args.preset]
        if args.samples is None:
            args.samples = preset["samples"]
        if args.batch_size is None:
            args.batch_size = preset["batch_size"]
        if args.experiments is None:
            args.experiments = preset["experiments"]
        if args.models is None and args.llm_model is None:
            args.models = preset["models"]

    # Defaults
    if args.samples is None:
        args.samples = 32
    if args.batch_size is None:
        args.batch_size = 8
    if args.experiments is None:
        args.experiments = ["model", "depth"]

    # Model list
    if args.llm_model:
        models = [args.llm_model]
    elif args.models:
        models = args.models
    else:
        models = ["rmanluo/RoG"]

    # ── Header ────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    tag = time.strftime("%Y%m%d_%H%M%S")

    print(f"\n{'#'*70}")
    print(f"#  RAG vs GraphRAG — Deep Profiling (ISCA)")
    print(f"#  Experiments: {args.experiments}")
    print(f"#  Models:      {models}")
    print(f"#  Dataset:     {args.dataset}  N={args.samples}  B={args.batch_size}")
    print(f"#  Params:      beams={args.num_beams}  max_tokens={args.max_tokens}")
    print(f"#  GPU:         {HW['gpu_name']}  "
          f"(HBM {HW['hbm_bw_GBs']:.0f} GB/s, FP16 {HW['fp16_tflops']:.0f} TFLOP/s, "
          f"ridge={HW['ridge']:.0f} F/B)")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"#  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"{'#'*70}\n")

    all_results: List[ProfileResult] = []

    # ── Run experiments ───────────────────────────────────────────────────
    common = dict(
        dataset_name=args.dataset, samples=args.samples,
        num_beams=args.num_beams, max_tokens=args.max_tokens,
        seed=args.seed, hop=args.hop, top_k=args.top_k,
        device=args.device,
    )

    if "model" in args.experiments:
        rs = run_model_sweep(models=models, batch_size=args.batch_size, **common)
        all_results.extend(rs)
        # Print per-model tables
        for model in models:
            model_rs = [r for r in rs if r.model_name == model]
            if model_rs:
                print_comparison_table(model_rs, f"MODEL SWEEP — {model}")

    if "depth" in args.experiments:
        depth_model = models[0]  # Use first model for depth sweep
        rs = run_depth_sweep(
            model_name=depth_model, hops=args.hops,
            batch_size=args.batch_size, **common)
        all_results.extend(rs)
        print_comparison_table(rs, f"DEPTH SWEEP — {depth_model}")

    if "batch" in args.experiments:
        batch_model = models[0]
        rs = run_batch_sweep(
            model_name=batch_model, batch_sizes=args.batch_sizes,
            **common)
        all_results.extend(rs)
        # Print per-batch-size tables
        for bs in args.batch_sizes:
            bs_rs = [r for r in rs if r.batch_size == bs]
            if bs_rs:
                print_comparison_table(bs_rs, f"BATCH SWEEP — B={bs}")

    if "context" in args.experiments:
        ctx_model = models[0]
        rs = run_context_sweep(
            model_name=ctx_model, batch_size=args.batch_size, **common)
        all_results.extend(rs)
        print_comparison_table(rs, f"CONTEXT SWEEP — {ctx_model}")

    if "graph_char" in args.experiments:
        gc_model = models[0]
        rs = run_graph_char(
            model_name=gc_model, dataset_name=args.dataset,
            samples=args.samples, batch_size=args.batch_size,
            num_beams=args.num_beams, max_tokens=args.max_tokens,
            seed=args.seed, hops=args.hops, top_k=args.top_k,
            device=args.device,
        )
        all_results.extend(rs)
        print_comparison_table(rs, f"GRAPH CHAR — {gc_model}")
        print_graph_char_report(rs)

    # ── Save ──────────────────────────────────────────────────────────────
    if all_results:
        save_results(all_results, out_dir, tag)

        # ── Final summary across all experiments ──────────────────────────
        print(f"\n{'#'*70}")
        print(f"#  PROFILING COMPLETE")
        print(f"#  Total runs: {len(all_results)}")
        print(f"#  Experiments: {set(r.experiment for r in all_results)}")
        print(f"#  Models:      {set(r.model_name for r in all_results)}")
        print(f"#  Results:     {out_dir}/")
        print(f"{'#'*70}")
    else:
        print("\n[ERROR] No experiments completed successfully.")


if __name__ == "__main__":
    main()
