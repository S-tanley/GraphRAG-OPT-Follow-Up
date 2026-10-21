"""
GraphRAG batch-size sweep benchmark — real experiment, real numbers.

Copies main.py exactly, adds:
  --condition    : which of the 3 experimental conditions to run
  --sweep_batches: space-separated list of batch sizes to sweep
  GPU busy %     : measured via CUDA events every generate() call
  MFU            : Model FLOP Utilisation (decode estimate, 2×N_params rule)
  JSON output    : benchmark_results/<condition>.json

Three conditions (maps directly to main.py CLI flags):
  graphrag_opt    --retrieve --path_enum --batch_opt --use_pfkv
  baseline_small  --retrieve  (no offloading; batch capped at VRAM_SAFE_BATCH)
  baseline_naive  --retrieve --path_enum  (no --batch_opt, no --use_pfkv)

Why NVML SM% was always 0
--------------------------
nvmlDeviceGetUtilizationRates has a hardware counter window of ~167 ms.
Polling at 100 ms reliably lands between GPU bursts for small models (gemma-2-2b
decode steps finish in microseconds).  We now use CUDA Events instead: they
record timestamps directly into the GPU command stream and are accurate to
microseconds regardless of kernel duration.

MFU (Model FLOP Utilisation)
-----------------------------
  FLOPs per decode token ≈ 2 × N_params   (weight matrix-vector products)
  MFU = (throughput_tok_s × flops_per_token) / (peak_tflops × 1e12)
Reference: Chowdhery et al. "PaLM" (2022), Section 4.

Usage
-----
python main_benchmark.py \\
    --condition graphrag_opt \\
    --sweep_batches 8 16 32 64 128 256 512 \\
    --llm_model Qwen/Qwen3-32B \\
    --dataset rmanluo/RoG-cwq \\
    --max_new_tokens 50 --num_beams 1

python main_benchmark.py --condition baseline_small  --sweep_batches 8 16 32 64 128 256 512 ...
python main_benchmark.py --condition baseline_naive  --sweep_batches 8 16 32 64 128 256 512 ...

# Optional: compile the model for faster decode (PyTorch >= 2.0)
python main_benchmark.py --condition baseline_naive --torch_compile ...

python plot_sweep_results.py
"""

import argparse
import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# NVML background SM-utilisation monitor
# ---------------------------------------------------------------------------
# Uses pynvml (or nvidia-ml-py) to sample SM% every 10 ms in a background
# thread.  10 ms is short enough to catch kernels that are >10 ms long
# (true for batch ≥ 32 on any modern GPU), giving a reliable mean SM%.
# Falls back to a no-op if pynvml is unavailable.
# ---------------------------------------------------------------------------

try:
    import pynvml as _pynvml
    _pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False


class NVMLSMMonitor:
    """Background-thread GPU SM-utilisation sampler (10 ms interval)."""

    def __init__(self, device_idx: int = 0, interval_ms: float = 10.0):
        self._idx        = device_idx
        self._interval   = interval_ms / 1_000.0
        self._samples: List[float] = []
        self._stop       = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, float]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        s = self._samples
        if not s:
            return {"mean": float("nan"), "peak": float("nan"), "p95": float("nan")}
        return {
            "mean": float(np.mean(s)),
            "peak": float(np.max(s)),
            "p95":  float(np.percentile(s, 95)),
        }

    def _run(self):
        if not _NVML_OK:
            return
        handle = _pynvml.nvmlDeviceGetHandleByIndex(self._idx)
        while not self._stop.is_set():
            try:
                util = _pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                self._samples.append(float(util))
            except Exception:
                pass
            time.sleep(self._interval)

from config import QUICK_MODE_CONFIGS, set_reproducible_seed
from generation import PathFirstGraphRAGGeneration, format_prompt
from graph_utils import path_to_str
from metrics import evaluate_metrics
from metaqa_adapter import load_metaqa
from retrieval import BatchOptimizedRetrieval, SimpleGraphRAGRetrieval


# ---------------------------------------------------------------------------
# GPU busy-time measurement via CUDA Events
# ---------------------------------------------------------------------------
# WHY NOT pynvml?
#   nvmlDeviceGetUtilizationRates has a ~167 ms hardware counter window.
#   Polling at 100 ms lands between GPU bursts for small models (2-B decode
#   steps take microseconds) → always reads 0.
#
# CUDA Events record timestamps directly in the GPU stream.
# elapsed_time(start, end) gives the actual GPU kernel execution time in ms,
# accurate to microseconds, with zero blind-spot.
#
# "GPU busy fraction" = cuda_elapsed_ms / wall_ms
#   → fraction of real time the GPU was executing kernels for this call.
# ---------------------------------------------------------------------------

class GPUBusyTimer:
    """
    Tracks actual GPU kernel time via CUDA Events.

    'GPU busy fraction' = CUDA stream elapsed time / wall-clock time.

    Unlike NVML SM%, this metric:
      - Has no blind-spot for short / bursty kernels
      - Requires no background thread or polling
      - Is accurate to microseconds

    Usage:
        timer = GPUBusyTimer()
        timer.reset()          # clear history between batch-size runs
        timer.start()          # record just before generate()
        ... generate() ...
        stats = timer.stop()   # per-call {"gpu_busy_pct", "gpu_ms", "wall_ms"}
        agg   = timer.aggregate()  # {"gpu_busy_mean", "gpu_busy_peak", "gpu_busy_p95"}
    """

    def __init__(self):
        self._gpu_ms:  List[float] = []
        self._wall_ms: List[float] = []
        self._active = torch.cuda.is_available()
        if self._active:
            self._ev_start = torch.cuda.Event(enable_timing=True)
            self._ev_end   = torch.cuda.Event(enable_timing=True)
        self._wall_t0: float = 0.0

    def reset(self):
        self._gpu_ms.clear()
        self._wall_ms.clear()

    def start(self):
        if self._active:
            self._ev_start.record()
        self._wall_t0 = time.perf_counter()

    def stop(self) -> Dict[str, float]:
        """Record one measurement. Returns per-call stats."""
        wall_ms = (time.perf_counter() - self._wall_t0) * 1_000.0
        if self._active:
            self._ev_end.record()
            torch.cuda.synchronize()
            gpu_ms = float(self._ev_start.elapsed_time(self._ev_end))
        else:
            gpu_ms = 0.0
        self._gpu_ms.append(gpu_ms)
        self._wall_ms.append(wall_ms)
        busy_pct = min(gpu_ms / wall_ms * 100.0, 100.0) if wall_ms > 0 else 0.0
        return {"gpu_busy_pct": busy_pct, "gpu_ms": gpu_ms, "wall_ms": wall_ms}

    def aggregate(self) -> Dict[str, float]:
        """Summarise all measurements accumulated since the last reset()."""
        if not self._gpu_ms:
            return {"gpu_busy_mean": -1.0, "gpu_busy_peak": -1.0, "gpu_busy_p95": -1.0}
        busy = np.clip(
            [g / w * 100.0 for g, w in zip(self._gpu_ms, self._wall_ms) if w > 0],
            0.0, 100.0,
        )
        return {
            "gpu_busy_mean": float(np.mean(busy)),
            "gpu_busy_peak": float(np.max(busy)),
            "gpu_busy_p95":  float(np.percentile(busy, 95)),
        }


# ---------------------------------------------------------------------------
# MFU (Model FLOP Utilisation) helpers
# ---------------------------------------------------------------------------

# Peak bf16/fp16 tensor-core TFLOPS for common NVIDIA GPUs.
# Sources: NVIDIA product datasheets.
_GPU_PEAK_TFLOPS: Dict[str, float] = {
    "h200":  989.0,
    "h100":  989.0,   # H100 SXM5 80 GB
    "a100":  312.0,   # A100 SXM4 80 GB
    "l40s":  362.1,
    "l40":   181.1,
    "l4":     30.3,
    "a10g":  125.0,
    "a10":   125.0,
    "a40":   149.7,
    "4090":   82.6,
    "4080":   48.7,
    "3090":   35.6,
    "3080":   29.8,
    "v100":  125.0,   # fp16 tensor core
}


def detect_peak_tflops(device_idx: int = 0) -> Optional[float]:
    """Return peak tensor-core TFLOPS for *device_idx*, or None if unknown."""
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name(device_idx).lower()
    for key, tflops in _GPU_PEAK_TFLOPS.items():
        if key in name:
            return tflops
    return None


def compute_mfu(
    model,
    throughput_tok_s: float,
    peak_tflops: Optional[float],
) -> Optional[float]:
    """
    Model FLOP Utilisation (MFU) for autoregressive decode.

    Standard estimate (Chowdhery et al. PaLM 2022, Section 4):
        FLOPs per decode token ≈ 2 × N_params
        MFU = (throughput × flops_per_token) / (peak_tflops × 1e12)

    Returns a value in [0, 1], or None when peak_tflops is unknown.
    Typical values: small models ~1-5%, large models well-utilised ~40-60%.
    """
    if peak_tflops is None or throughput_tok_s <= 0:
        return None
    try:
        # Unwrap torch.compile wrapper if present
        raw = getattr(model, "_orig_mod", model)
        n_params = sum(p.numel() for p in raw.parameters())
    except Exception:
        return None
    flops_per_token  = 2 * n_params
    achieved_tflops  = throughput_tok_s * flops_per_token / 1e12
    return min(achieved_tflops / peak_tflops, 1.0)


# ---------------------------------------------------------------------------
# torch.profiler snapshot  (runs once per batch-size for kernel-level details)
# ---------------------------------------------------------------------------
#
# WHY RUN ONLY ONCE?
#   The profiler adds ~10-20% overhead per call.  We pay that cost on the
#   *first* batch only to get kernel breakdowns; every other batch uses the
#   zero-overhead CUDA Events timer so throughput numbers stay clean.
#
# WHAT YOU GET:
#   • total_cuda_ms     – total self-CUDA time across all kernels (no double-count)
#   • total_cpu_ms      – total self-CPU time
#   • gpu_kernel_pct    – total_cuda_ms / wall_ms × 100
#                         (should closely match GPUBusyTimer's gpu_busy_mean_pct)
#   • estimated_gflops  – shape-based FLOP estimate via with_flops=True
#   • top5_kernels      – top 5 ops by self-CUDA time; tells you whether you
#                         are compute-bound (big GEMMs) or memory-bound (elementwise)
#   • cupti_available   – whether CUPTI hardware counters were active
#   • sm_occupancy_pct  – mean SM warp occupancy % across kernels (CUPTI only)
#   • trace_path        – chrome://tracing JSON if --save_traces was set;
#                         open in PyTorch profiler UI for per-kernel SM heat-map

def _profile_one_batch(
    generate_fn,
    prompts: List[str],
    batch_paths,
    trace_path: Optional[Path] = None,
) -> Tuple[List[str], Dict]:
    """
    Wrap generate_fn in torch.profiler for one batch; return (preds, stats).

    Tries CUPTI hardware SM occupancy first; silently falls back to
    software-only profiling when CUPTI is unavailable (most cloud machines).
    """
    import torch.profiler as tp

    if not torch.cuda.is_available():
        preds = generate_fn(prompts, batch_paths)
        return preds, {"note": "CUDA not available — profiling skipped"}

    activities = [tp.ProfilerActivity.CPU, tp.ProfilerActivity.CUDA]

    # ── CUPTI hardware SM occupancy (optional) ────────────────────────────
    # Needs: CUDA >= 11.3, CUPTI installed, perf_event_paranoid <= 2.
    # On most cloud/shared machines this will silently fail → fallback below.
    _SM_METRIC = "sm__warps_active.avg.pct_of_peak_sustained_active"
    exp_cfg = None
    try:
        exp_cfg = torch._C._profiler._ExperimentalConfig(
            profiler_metrics=[_SM_METRIC],
            profiler_measure_per_kernel=True,
        )
    except Exception:
        pass

    def _run(use_cupti: bool) -> Tuple[List[str], "tp.profile", float]:
        prof_kwargs: Dict = dict(
            activities=activities,
            record_shapes=False,   # skip to reduce overhead during throughput sweep
            with_flops=True,       # shape-based FLOP estimation; no CUPTI required
            with_stack=False,
        )
        if use_cupti and exp_cfg is not None:
            prof_kwargs["experimental_config"] = exp_cfg
        t0 = time.perf_counter()
        with tp.profile(**prof_kwargs) as prof:
            preds = generate_fn(prompts, batch_paths)
        wall_ms = (time.perf_counter() - t0) * 1_000.0
        return preds, prof, wall_ms

    cupti_ok = exp_cfg is not None
    try:
        preds, prof, wall_ms = _run(use_cupti=True)
    except Exception as exc:
        if exp_cfg is not None:
            print(f"[profiler] CUPTI failed ({type(exc).__name__}: {exc}), "
                  "retrying without hardware counters")
        cupti_ok = False
        preds, prof, wall_ms = _run(use_cupti=False)

    # ── Extract kernel stats ───────────────────────────────────────────────
    key_avgs = prof.key_averages()

    # PyTorch renamed self_cuda_time_total → self_device_time_total in 2.x.
    # Use a safe getter that tries both attribute names.
    def _self_cuda_us(e) -> float:
        for attr in ("self_cuda_time_total", "self_device_time_total"):
            v = getattr(e, attr, None)
            if v is not None:
                return float(v)
        return 0.0

    total_self_cuda_us = sum(_self_cuda_us(e) for e in key_avgs)
    total_self_cpu_us  = sum(e.self_cpu_time_total for e in key_avgs)
    total_flops        = sum(e.flops or 0 for e in key_avgs)

    # Top-5 by self-CUDA time — dominated by actual compute kernels
    top5 = sorted(key_avgs, key=_self_cuda_us, reverse=True)[:5]
    top5_info = [
        {
            "name":          e.key[:80],
            "self_cuda_ms":  round(_self_cuda_us(e) / 1_000.0, 3),
            "cuda_ms_total": round(getattr(e, "cuda_time_total", 0) / 1_000.0, 3),
            "calls":         e.count,
        }
        for e in top5 if _self_cuda_us(e) > 0
    ]

    gpu_kernel_pct = (
        min(total_self_cuda_us / (wall_ms * 1_000.0) * 100.0, 100.0)
        if wall_ms > 0 else 0.0
    )

    snap: Dict = {
        "total_cuda_ms":    round(total_self_cuda_us / 1_000.0, 2),
        "total_cpu_ms":     round(total_self_cpu_us  / 1_000.0, 2),
        "wall_ms":          round(wall_ms, 2),
        "gpu_kernel_pct":   round(gpu_kernel_pct, 2),
        "estimated_gflops": round(total_flops / 1e9, 3) if total_flops else None,
        "top5_kernels":     top5_info,
        "cupti_available":  cupti_ok,
        "sm_occupancy_pct": None,   # filled below if CUPTI succeeded
        "trace_path":       None,
    }

    # ── CUPTI SM occupancy extraction ─────────────────────────────────────
    # PyTorch 2.1+ stores per-kernel CUPTI metrics in the raw event list.
    # Any extraction failure is silently ignored; the trace file (if saved)
    # always contains the full SM data for inspection in chrome://tracing.
    if cupti_ok:
        try:
            occ_vals: List[float] = []
            for evt in prof.events():
                fields = getattr(evt, "extra_fields", None)
                if not fields:
                    continue
                # extra_fields is a dict or iterable depending on PyTorch version
                val = None
                if isinstance(fields, dict):
                    val = fields.get(_SM_METRIC)
                elif hasattr(fields, "__iter__"):
                    for item in fields:
                        if isinstance(item, (int, float)):
                            val = float(item)
                            break
                if val is not None:
                    try:
                        f = float(val)
                        if 0.0 <= f <= 100.0:
                            occ_vals.append(f)
                    except (TypeError, ValueError):
                        pass
            if occ_vals:
                snap["sm_occupancy_pct"] = round(float(np.mean(occ_vals)), 2)
        except Exception:
            pass  # silently ignored; user can inspect Chrome trace

    # ── Export Chrome trace ───────────────────────────────────────────────
    # Opens in chrome://tracing or the PyTorch profiler TensorBoard plugin.
    # When CUPTI is active it contains per-kernel SM occupancy heat-maps.
    if trace_path is not None:
        try:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(str(trace_path))
            snap["trace_path"] = str(trace_path)
            print(f"  [profiler] Chrome trace → {trace_path}")
        except Exception as exc:
            print(f"  [profiler] Could not write trace: {exc}")

    return preds, snap


# ---------------------------------------------------------------------------
# Condition → flag mapping
# ---------------------------------------------------------------------------

# Largest batch that fits without any KV offloading.
VRAM_SAFE_BATCH = 64

CONDITIONS = {
    "graphrag_opt": {
        "retrieve":  True,
        "path_enum": True,
        "batch_opt": True,
        "use_pfkv":  True,
        "description": "GraphRAG-Opt (smart scheduling + PFKV offloading)",
    },
    "baseline_small": {
        "retrieve":  True,
        "path_enum": False,
        "batch_opt": False,
        "use_pfkv":  False,
        "description": f"Baseline: reduced batch (no offloading, cap={VRAM_SAFE_BATCH})",
    },
    "baseline_naive": {
        "retrieve":  True,
        "path_enum": True,
        "batch_opt": False,
        "use_pfkv":  False,
        "description": "Baseline: naive (path_enum, no batch_opt/pfkv)",
    },
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GraphRAG batch-sweep benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Benchmark-specific ───────────────────────────────────────────────
    parser.add_argument(
        "--condition",
        choices=list(CONDITIONS.keys()),
        required=True,
        help="Experimental condition to run",
    )
    parser.add_argument(
        "--sweep_batches", nargs="+", type=int,
        default=[8, 16, 32, 64, 128, 256, 512],
        help="Batch sizes to sweep (default: 8 16 32 64 128 256 512)",
    )
    parser.add_argument(
        "--samples_per_batch_size", type=int, default=128,
        help="Dataset samples used per batch-size run (default: 128)",
    )
    parser.add_argument(
        "--out_dir", default="benchmark_results",
        help="Directory to write JSON results (default: benchmark_results/)",
    )
    parser.add_argument(
        "--gpu_monitor_idx", type=int, default=0,
        help="GPU device index for MFU peak-TFLOPS lookup (default: 0)",
    )

    # ── Profiling ────────────────────────────────────────────────────────
    parser.add_argument(
        "--profile", action="store_true",
        help=(
            "Run torch.profiler on the first batch of each batch-size sweep. "
            "Adds ~10-20%% overhead to that one batch only; all other batches "
            "use the zero-overhead CUDA Events timer. Outputs top-5 GPU kernels, "
            "FLOP estimate, gpu_kernel_pct, and SM occupancy (if CUPTI available) "
            "into the JSON under 'profiler_first_batch'."
        ),
    )
    parser.add_argument(
        "--save_traces", action="store_true",
        help=(
            "Save a Chrome trace file for each profiled batch to "
            "benchmark_results/profiler_traces/<condition>_bs<N>.json. "
            "Open with chrome://tracing or the PyTorch profiler TensorBoard plugin "
            "to inspect per-kernel SM occupancy, memory BW, and kernel timelines."
        ),
    )

    # ── Model compilation ────────────────────────────────────────────────
    parser.add_argument(
        "--torch_compile", action="store_true",
        help=(
            "Compile the model with torch.compile(dynamic=True) before inference. "
            "Requires PyTorch >= 2.0. Warms up on the first batch then accelerates "
            "decode by fusing kernels and reducing launch overhead. "
            "Most effective for larger models; may not help much for 2-B models."
        ),
    )

    # ── Dataset / model (same as main.py) ────────────────────────────────
    parser.add_argument("--dataset", default="rmanluo/RoG-cwq")
    parser.add_argument("--metaqa_dir", default="metaQA")
    parser.add_argument("--metaqa_split", default="test",
                        choices=["train", "dev", "test"])
    parser.add_argument("--llm_model", default="Qwen/Qwen3-32B")
    parser.add_argument("--use_auth_token", default=None)
    parser.add_argument("--seed", type=int, default=42)

    # ── Generation ───────────────────────────────────────────────────────
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--num_beams",      type=int, default=1)
    parser.add_argument("--no_kv_cache",    action="store_true")

    # ── Retrieval ────────────────────────────────────────────────────────
    parser.add_argument("--hop_constraint",      type=int, default=3)
    parser.add_argument("--max_paths_per_query", type=int, default=None)
    parser.add_argument("--pfkv_cache_size",     type=int, default=1000)

    # ── Model loading ────────────────────────────────────────────────────
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--use_fast_tokenizer",    action="store_true")
    parser.add_argument("--no_use_fast_tokenizer",
                        dest="use_fast_tokenizer", action="store_false")

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cond_cfg = CONDITIONS[args.condition]
    print(f"\n{'='*70}")
    print(f"BENCHMARK  condition={args.condition}")
    print(f"  {cond_cfg['description']}")
    print(f"  Batch sizes   : {args.sweep_batches}")
    print(f"  Model         : {args.llm_model}")
    print(f"  torch.compile : {args.torch_compile}")
    print(f"  profiler      : {args.profile}"
          + (" (+ Chrome traces)" if args.profile and args.save_traces else ""))
    print(f"{'='*70}\n")

    set_reproducible_seed(args.seed)

    do_retrieve  = cond_cfg["retrieve"]
    do_path_enum = cond_cfg["path_enum"]
    do_batch_opt = cond_cfg["batch_opt"]
    do_use_pfkv  = cond_cfg["use_pfkv"]

    # ── GPU info + peak TFLOPS for MFU ───────────────────────────────────
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(n)]
        print(f"[GPU] {n} GPU(s): {names}")
    else:
        print("[GPU] No CUDA — running on CPU")

    peak_tflops = detect_peak_tflops(args.gpu_monitor_idx)
    if peak_tflops is not None:
        print(f"[MFU] Detected peak TFLOPS = {peak_tflops:.1f} "
              f"(bf16/fp16 tensor core, GPU {args.gpu_monitor_idx})")
    else:
        gpu_name = (torch.cuda.get_device_name(args.gpu_monitor_idx)
                    if torch.cuda.is_available() else "N/A")
        print(f"[MFU] GPU '{gpu_name}' not in lookup table — "
              f"MFU will be null. Add it to _GPU_PEAK_TFLOPS if needed.")

    # ── Load dataset ──────────────────────────────────────────────────────
    total_needed = max(args.samples_per_batch_size, max(args.sweep_batches))
    print(f"\nLoading dataset: {args.dataset}  ({total_needed} samples needed)")

    _metaqa_shortcuts = {"metaqa-1hop": 1, "metaqa-2hop": 2, "metaqa-3hop": 3}
    if args.dataset in _metaqa_shortcuts:
        hops = _metaqa_shortcuts[args.dataset]
        local_kb = Path(args.metaqa_dir) / "kb.txt"
        if local_kb.exists():
            full_dataset = load_metaqa(
                hops=hops, split=args.metaqa_split, data_dir=args.metaqa_dir)
        elif hops == 3:
            hf_split = "validation" if args.metaqa_split == "dev" else args.metaqa_split
            full_dataset = load_dataset("mhivanov123/metaqa-3hop", split=hf_split)
        else:
            raise FileNotFoundError(
                f"MetaQA local files not found at {args.metaqa_dir}.")
    else:
        full_dataset = load_dataset(args.dataset, split="test")

    full_dataset = full_dataset.shuffle(seed=args.seed).select(
        range(min(total_needed, len(full_dataset))))
    print(f"Dataset loaded: {len(full_dataset)} samples\n")

    # ── Load model once — reused across all batch sizes ───────────────────
    use_kv_cache = not args.no_kv_cache
    generator = PathFirstGraphRAGGeneration(
        args.llm_model,
        use_kv_cache=use_kv_cache,
        use_pfkv=do_use_pfkv,
        pfkv_cache_size=args.pfkv_cache_size,
        use_auth_token=args.use_auth_token,
        device_map=args.device_map,
        use_fast_tokenizer=args.use_fast_tokenizer,
        parallel_mode="none",
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        torch_compile=args.torch_compile,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    timer = GPUBusyTimer()
    sweep_results: List[Dict] = []

    # ── Outer loop: sweep batch sizes ─────────────────────────────────────
    for target_bs in args.sweep_batches:

        # baseline_small caps effective batch at VRAM_SAFE_BATCH
        eff_bs = (min(target_bs, VRAM_SAFE_BATCH)
                  if args.condition == "baseline_small"
                  else target_bs)

        # Round sample count down to a multiple of eff_bs
        n_samples = min(args.samples_per_batch_size, len(full_dataset))
        n_samples = max(eff_bs, (n_samples // eff_bs) * eff_bs)
        dataset   = full_dataset.select(range(n_samples))
        num_batches = n_samples // eff_bs

        print(f"\n{'─'*60}")
        print(f"target_batch={target_bs}  eff_batch={eff_bs}  "
              f"samples={n_samples}  batches={num_batches}")

        retrieve_times, gen_times, throughputs = [], [], []
        vram_peak_gbs: List[float] = []
        nvml_sm_samples: List[float] = []
        total_hits = total_f1 = 0.0
        oom = False
        timer.reset()
        nvml_mon = NVMLSMMonitor(device_idx=args.gpu_monitor_idx, interval_ms=10.0)
        profiler_snap: Optional[Dict] = None

        try:
            for batch_idx in tqdm(range(num_batches),
                                  desc=f"bs={eff_bs}", leave=False):
                s = batch_idx * eff_bs
                e = s + eff_bs
                batch_samples = dataset.select(range(s, e))
                batch_slice   = dataset[s:e]
                queries  = batch_slice["question"]
                answers  = batch_slice["answer"]
                prompts: list     = []
                batch_paths: list = []

                # ── Retrieval ──────────────────────────────────────────
                if do_retrieve:
                    t0 = time.perf_counter()
                    if do_path_enum:
                        retriever = BatchOptimizedRetrieval(
                            batch_samples, device="cuda")
                        results, _ = retriever.retrieve_batch(
                            args.hop_constraint, do_batch_opt,
                            args.max_paths_per_query)
                    else:
                        retriever = SimpleGraphRAGRetrieval(batch_samples)
                        results, _ = retriever.retrieve_batch(
                            args.hop_constraint, args.max_paths_per_query)
                    retrieve_times.append(time.perf_counter() - t0)
                    for r in results:
                        prompts.append(
                            format_prompt(r["query"], r["paths"], args.llm_model))
                        batch_paths.append(r["paths"])
                else:
                    retrieve_times.append(0.0)
                    for q in queries:
                        prompts.append(format_prompt(q, [], args.llm_model))
                        batch_paths.append([])

                # ── Generation + GPU timing ────────────────────────────
                # batch_idx == 0 with --profile: run torch.profiler for kernel
                # breakdown; overhead (~10-20%) only affects this one batch.
                # All other batches use CUDA Events (zero overhead).
                pfkv_paths = batch_paths if (do_use_pfkv and do_retrieve) else None

                # Snapshot VRAM BEFORE generate so we can compute the delta
                # (do NOT reset the peak counter — model weights must be included)
                vram_before_gb = (
                    torch.cuda.memory_allocated() / 1e9
                    if torch.cuda.is_available() else 0.0
                )

                nvml_mon.start()
                timer.start()

                if batch_idx == 0 and args.profile:
                    trace_file = (
                        Path(args.out_dir) / "profiler_traces"
                        / f"{args.condition}_bs{eff_bs}.json"
                        if args.save_traces else None
                    )
                    predictions, profiler_snap = _profile_one_batch(
                        lambda p, bp: generator.generate_batch(p, bp),
                        prompts, pfkv_paths,
                        trace_path=trace_file,
                    )
                else:
                    predictions = generator.generate_batch(prompts, pfkv_paths)

                call_stats = timer.stop()
                sm_stat = nvml_mon.stop()
                if not np.isnan(sm_stat["mean"]):
                    nvml_sm_samples.append(sm_stat["mean"])

                # Total VRAM in use after generate() (model weights + KV cache)
                if torch.cuda.is_available():
                    vram_after_gb = torch.cuda.memory_allocated() / 1e9
                    vram_peak_gbs.append(vram_after_gb)
                gen_time = call_stats["wall_ms"] / 1_000.0
                gen_times.append(gen_time)

                # Token throughput = (input + output tokens) / wall time
                prompt_toks = sum(
                    len(generator.tokenizer.encode(p)) for p in prompts)
                output_toks = sum(
                    len(generator.tokenizer.encode(p)) for p in predictions)
                tput = (prompt_toks + output_toks) / gen_time if gen_time > 0 else 0.0
                throughputs.append(tput)

                m = evaluate_metrics(predictions, answers)
                total_hits += m["hits_at_1"] * len(predictions)
                total_f1   += m["f1"]        * len(predictions)

        except torch.cuda.OutOfMemoryError:
            print(f"  [OOM] eff_batch={eff_bs} — skipping rest")
            oom = True
            torch.cuda.empty_cache()

        # ── Aggregate across batches ──────────────────────────────────
        if throughputs:
            gpu_agg  = timer.aggregate()
            mean_tput = float(np.mean(throughputs))
            mfu_val   = compute_mfu(generator.model, mean_tput, peak_tflops)

            # VRAM peak
            vram_mean = float(np.mean(vram_peak_gbs)) if vram_peak_gbs else float("nan")
            vram_peak = float(np.max(vram_peak_gbs))  if vram_peak_gbs else float("nan")

            # NVML SM%
            sm_mean = float(np.mean(nvml_sm_samples)) if nvml_sm_samples else float("nan")
            sm_peak = float(np.max(nvml_sm_samples))  if nvml_sm_samples else float("nan")

            rec = {
                "condition":            args.condition,
                "target_batch":         target_bs,
                "eff_batch":            eff_bs,
                "n_samples":            n_samples,
                "oom":                  oom,
                # Throughput
                "throughput_mean":      mean_tput,
                "throughput_p95":       float(np.percentile(throughputs, 95)),
                # GPU VRAM (peak allocated during generate(), in GB)
                "vram_peak_gb_mean":    vram_mean,
                "vram_peak_gb_max":     vram_peak,
                # GPU SM utilisation (NVML 10ms polling — primary SM metric for paper)
                "sm_util_mean_pct":     sm_mean,
                "sm_util_peak_pct":     sm_peak,
                # GPU busy fraction (CUDA Events — complementary to SM%)
                "gpu_busy_mean_pct":    gpu_agg["gpu_busy_mean"],
                "gpu_busy_peak_pct":    gpu_agg["gpu_busy_peak"],
                "gpu_busy_p95_pct":     gpu_agg["gpu_busy_p95"],
                # MFU
                "mfu":                  mfu_val,
                "mfu_pct":              round(mfu_val * 100, 2) if mfu_val is not None else None,
                "peak_gpu_tflops":      peak_tflops,
                # Timing
                "gen_time_mean_s":      float(np.mean(gen_times)),
                "retrieve_time_mean_s": float(np.mean(retrieve_times)),
                # Accuracy
                "hits_at_1":            total_hits / n_samples,
                "f1":                   total_f1   / n_samples,
                # Config
                "model":                args.llm_model,
                "dataset":              args.dataset,
                "seed":                 args.seed,
                "max_new_tokens":       args.max_new_tokens,
                "num_beams":            args.num_beams,
                "torch_compile":        args.torch_compile,
                # torch.profiler snapshot from the first batch (None if --profile not set)
                "profiler_first_batch": profiler_snap,
            }
            mfu_str = f"MFU={mfu_val*100:.1f}%" if mfu_val is not None else "MFU=N/A"
            prof_str = ""
            if profiler_snap and "gpu_kernel_pct" in profiler_snap:
                prof_str = (
                    f"  [profiler] gpu_kernel%={profiler_snap['gpu_kernel_pct']:.1f}  "
                    f"gflops={profiler_snap['estimated_gflops']}  "
                    f"SM_occ={profiler_snap['sm_occupancy_pct']}%  "
                    f"CUPTI={'yes' if profiler_snap['cupti_available'] else 'no'}\n"
                    f"  top kernels: " + " | ".join(
                        f"{k['name'].split('::')[-1][:30]} {k['self_cuda_ms']}ms×{k['calls']}"
                        for k in profiler_snap["top5_kernels"]
                    )
                )
            sm_str   = f"{sm_mean:.1f}%" if not np.isnan(sm_mean) else "N/A"
            vram_str = f"{vram_mean:.2f} GB" if not np.isnan(vram_mean) else "N/A"
            print(
                f"  throughput: {mean_tput:.1f} tok/s  |  "
                f"VRAM peak: {vram_str}  |  "
                f"SM util: {sm_str}  |  "
                f"GPU busy: {gpu_agg['gpu_busy_mean']:.1f}%  |  "
                f"{mfu_str}  |  "
                f"Hits@1={rec['hits_at_1']:.3f}"
            )
            if prof_str:
                print(prof_str)
        else:
            rec = {
                "condition":         args.condition,
                "target_batch":      target_bs,
                "eff_batch":         eff_bs,
                "oom":               True,
                "throughput_mean":   None,
                "vram_peak_gb_mean": None,
                "sm_util_mean_pct":  None,
                "gpu_busy_mean_pct": None,
                "gpu_busy_peak_pct": None,
                "mfu":               None,
                "mfu_pct":           None,
            }
            print(f"  OOM at eff_batch={eff_bs}")

        sweep_results.append(rec)

        # Flush after every batch size so partial results survive crashes
        out_path = out_dir / f"{args.condition}.json"
        with open(out_path, "w") as f:
            json.dump(sweep_results, f, indent=2)

    print(f"\n[done] Results saved to {out_path}")
    print("Run all three conditions, then: python plot_sweep_results.py")


if __name__ == "__main__":
    main()
