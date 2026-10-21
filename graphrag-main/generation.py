"""
LLM generation for GraphRAG.

FirstTokenStoppingCriteria   – records time-to-first-token (TTFT)
format_prompt                – builds the few-shot inference prompt
PathFirstGraphRAGGeneration  – loads the model, handles multi-GPU modes,
                               wraps PFKV caching, and runs inference
"""

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers.cache_utils import DynamicCache

from graph_utils import path_to_str
from kv_cache import PathFirstKVCache
from metrics import evaluate_metrics as _evaluate_metrics, normalize_answer

# Optional SubgraphKVCache (paper Section 5)
try:
    from subgraph_kv_cache import SubgraphKVCache, run_all_sanity_checks
    SUBGRAPH_KV_AVAILABLE = True
except ImportError:
    SUBGRAPH_KV_AVAILABLE = False
    print("Warning: subgraph_kv_cache not found — using legacy PathFirstKVCache")

# Faithful subgraph KV reuse (Llama-family): each path is represented ONCE, as
# cached KV at its correct interleaved position, with optional selective
# recompute. Replaces the legacy path that left path text in the prompt while
# also prepending cached KV (double-representation), which hid the reuse error.
try:
    from selective_recompute import LlamaManualPrefill
    from faithful_pfkv import faithful_generate
    FAITHFUL_PFKV_AVAILABLE = True
except ImportError:
    FAITHFUL_PFKV_AVAILABLE = False


# ---------------------------------------------------------------------------
# Stopping criteria (TTFT tracking)
# ---------------------------------------------------------------------------

class FirstTokenStoppingCriteria(StoppingCriteria):
    """Records the wall-clock time when the first new token is generated."""

    def __init__(self):
        super().__init__()
        self.first_token_time: Optional[float] = None
        self.start_time: Optional[float] = None

    def set_start_time(self, start_time: float) -> None:
        self.start_time = start_time
        self.first_token_time = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.first_token_time is None and self.start_time is not None:
            self.first_token_time = time.perf_counter()
        return False  # never actually stop early


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def _path_to_str_rog(path) -> str:
    """Arrow-joined path format expected by rmanluo/RoG fine-tuned models."""
    if isinstance(path, str):
        return path
    if not path:
        return ""
    parts = [path[0][0]]
    for _h, r, t in path:
        parts.append(r)
        parts.append(t)
    return " -> ".join(parts)


def format_prompt(query: str, paths: List, model_name: str = "") -> str:
    """Build the inference prompt from a query and retrieved paths.

    Uses the original RoG fine-tuning template when model_name contains 'rog'
    (e.g. rmanluo/RoG), so the model sees the exact format it was trained on.
    All other models continue to use the existing generic few-shot template.
    """
    if "rog" in model_name.lower():
        # ── RoG-specific prompt (Alpaca format) ──────────────────────────
        # rmanluo/RoG was fine-tuned with the Stanford Alpaca instruction
        # template, NOT the Vicuna "### Human:" template.  Using the wrong
        # template causes near-zero accuracy because the model never
        # recognises the instruction and generates off-topic text.
        #
        # Correct format from the RoG training code:
        #   ### Instruction:\n{task}\n\n### Input:\n{graph_paths}\n\n### Response:
        if paths:
            arrow_paths = [_path_to_str_rog(p) for p in paths[:10]]
            path_str = "\n".join(arrow_paths)
        else:
            path_str = "N/A"
        return (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            f"Please reasoning over the following knowledge graph paths, "
            f"and then answer the question: {query}\n\n"
            "### Input:\n"
            f"Knowledge Graph Paths:\n{path_str}\n\n"
            "### Response:"
        )

    # ── Generic few-shot template (unchanged for all other models) ───────
    sorted_paths = sorted(paths, key=lambda p: len(p) if not isinstance(p, str) else len(p.split(".")))
    path_strings = [path_to_str(p) if not isinstance(p, str) else p for p in sorted_paths]
    return (
        "You are a concise, factual assistant. Based on the following context, "
        "please answer the question.\n\n"
        "Answer concisely with the most relevant short phrase(s) "
        "(e.g., person, place, year, etc.).\n"
        "Do not add extra explanation.\n\n"
        "Q: What is the capital of France?\n"
        "A: Paris\n\n"
        "Q: Who was the first president of the United States?\n"
        "A: George Washington\n\n"
        "Q: Which countries in Europe have a coastline on the Mediterranean Sea?\n"
        "A: Spain, France, Italy, Slovenia, Croatia, Montenegro, Albania, Greece, Turkey.\n\n"
        f"Context (relations relevant to the question and answer):\n{path_strings[:10]}\n\n"
        f"Q: {query}\n"
        "A:"
    )


# ---------------------------------------------------------------------------
# Generation engine
# ---------------------------------------------------------------------------

class PathFirstGraphRAGGeneration:
    """
    GraphRAG generation engine.

    Supports:
      • standard single-GPU inference
      • data-parallel inference (N independent model replicas, one per GPU)
      • model-parallel inference (layers spread across GPUs via device_map)
      • Path-First / Subgraph KV caching (paper Section 5)
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        use_kv_cache: bool = True,
        use_pfkv: bool = False,
        pfkv_cache_size: int = 1000,
        use_auth_token: Optional[str] = None,
        device_map: Optional[str] = "auto",
        use_fast_tokenizer: bool = False,
        run_sanity_checks: bool = False,
        parallel_mode: str = "none",
        num_gpus: Optional[int] = None,
        gpu_ids: Optional[str] = None,
        max_new_tokens: int = 256,
        num_beams: int = 1,
        torch_compile: bool = False,
        recompute_policy: str = "none",
        pfkv_max_paths: int = 8,
        pfkv_template: Optional[str] = None,
    ):
        self.device_str = device
        self.device = torch.device(device)
        self.model_name = model_name
        self.use_kv_cache = use_kv_cache
        self.use_pfkv = use_pfkv
        self.ttfts: List[float] = []
        self.generation_times: List[float] = []
        self._use_auth_token = use_auth_token
        self._pfkv_cache_size = pfkv_cache_size
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

        print(f"Loading model: {model_name}")
        print(f"Standard KV Cache: {'Enabled' if use_kv_cache else 'Disabled'}")
        print(f"Subgraph KV Cache (PFKV): {'Enabled' if use_pfkv else 'Disabled'}")

        is_gemma = "gemma" in model_name.lower()
        is_qwen = "qwen" in model_name.lower()

        if parallel_mode == "data_parallel" and device_map:
            print("[DataParallel] Overriding device_map=None")
            device_map = None

        # ── Model loading ────────────────────────────────────────────────
        if model_name == "Llama-2-7b":
            local = "/home/fa011718/workspace/RAG/GraphRAG_optimize/model/Llama-2-7b"
            self.tokenizer = AutoTokenizer.from_pretrained(local, use_fast=False)
            self.model = AutoModelForCausalLM.from_pretrained(
                local, device_map="auto", torch_dtype=torch.float16)
        elif model_name == "Llama-2-13b":
            local = "/home/fa011718/workspace/RAG/GraphRAG_optimize/model/Llama-2-13b"
            self.tokenizer = AutoTokenizer.from_pretrained(local, use_fast=False)
            self.model = AutoModelForCausalLM.from_pretrained(
                local, device_map="auto", torch_dtype=torch.float16)
        elif is_gemma or is_qwen:
            tok_kwargs: Dict = {"use_fast": True}
            if use_auth_token:
                tok_kwargs["token"] = use_auth_token
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            mdl_kwargs: Dict = {"torch_dtype": torch.bfloat16}
            if device_map:
                mdl_kwargs["device_map"] = device_map
            if use_auth_token:
                mdl_kwargs["token"] = use_auth_token
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **mdl_kwargs)
            if not device_map:
                self.model = self.model.to(self.device)
        else:
            # The tokenizer files bundled with rmanluo/RoG (an old Llama-2-7b
            # SentencePiece export) decode incorrectly under transformers >= 5.x
            # (every sub-word token gets space-joined, e.g. "Shia Islam" ->
            # "Sh ia I sl am"), which silently zeroes out exact-match metrics.
            # RoG shares the Llama-2 vocabulary verbatim, so we load the
            # canonical meta-llama/Llama-2-7b-hf fast tokenizer, which decodes
            # correctly. All other models load their own tokenizer.
            tokenizer_source = model_name
            force_fast = use_fast_tokenizer
            if "rog" in model_name.lower():
                tokenizer_source = "meta-llama/Llama-2-7b-hf"
                force_fast = True
            tok_kwargs = {"use_fast": force_fast}
            if use_auth_token:
                tok_kwargs["token"] = use_auth_token
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tok_kwargs)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            dtype = torch.float16 if self.device.type == "cuda" else None
            mdl_kwargs = {}
            if use_auth_token:
                mdl_kwargs["token"] = use_auth_token
            if device_map:
                mdl_kwargs["device_map"] = device_map
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype, use_safetensors=True, **mdl_kwargs)
            if not device_map:
                self.model = self.model.to(self.device)

        self._model_dtype = next(self.model.parameters()).dtype
        self.model.eval()

        # ── torch.compile (optional) ─────────────────────────────────────
        # Uses TorchInductor to fuse CUDA kernels and reduce launch overhead.
        # dynamic=True handles the variable sequence lengths in LLM generation.
        # The first batch pays a one-time compilation cost (~30-120 s depending
        # on model size); subsequent batches run the compiled, faster version.
        # Most impactful for larger models; marginal for 2-B-param models.
        if torch_compile:
            if not hasattr(torch, "compile"):
                print("[torch.compile] Skipped — requires PyTorch >= 2.0")
            elif not torch.cuda.is_available():
                print("[torch.compile] Skipped — CUDA not available")
            else:
                print("[torch.compile] Compiling model (dynamic=True) … "
                      "first batch will be slow while kernels are compiled.")
                try:
                    self.model = torch.compile(self.model, dynamic=True)
                    print("[torch.compile] Done.")
                except Exception as exc:
                    print(f"[torch.compile] Failed: {exc} — running uncompiled.")

        # ── PFKV cache ───────────────────────────────────────────────────
        if self.use_pfkv:
            if SUBGRAPH_KV_AVAILABLE:
                self.pfkv_cache = SubgraphKVCache(self.model, self.tokenizer, device, pfkv_cache_size)
                if run_sanity_checks:
                    all_passed, _ = run_all_sanity_checks(self.model, self.tokenizer, device)
                    if not all_passed:
                        print("[PFKV] WARNING: some sanity checks failed!")
            else:
                self.pfkv_cache = PathFirstKVCache(self.model, self.tokenizer, device, pfkv_cache_size)
        else:
            self.pfkv_cache = None

        self.using_subgraph_cache = self.use_pfkv and SUBGRAPH_KV_AVAILABLE

        # ── Faithful subgraph KV reuse (Llama-family) ────────────────────
        # Routes --use_pfkv through the weight-level engine so reuse is
        # measured honestly (path appears once, as cached KV at its correct
        # position; no double-representation), with selective recompute.
        self.recompute_policy = recompute_policy
        self.pfkv_max_paths = pfkv_max_paths
        self.pfkv_template = pfkv_template
        self.faithful_engine = None
        self.using_faithful_pfkv = False
        if self.use_pfkv and FAITHFUL_PFKV_AVAILABLE:
            is_llama = getattr(self.model.config, "model_type", "") == "llama"
            single_device = not (getattr(self.model, "hf_device_map", None)
                                 and len(set(self.model.hf_device_map.values())) > 1)
            if is_llama and single_device:
                self.faithful_engine = LlamaManualPrefill(self.model)
                self.using_faithful_pfkv = True
                self._pfkv_path_cache = {}   # global cross-query path-KV reuse
                tmpl = self.pfkv_template or ("rog" if "rog" in model_name.lower() else "fewshot")
                self._pfkv_template = tmpl
                print(f"[PFKV] Faithful subgraph KV reuse enabled "
                      f"(recompute_policy={recompute_policy}, template={tmpl})")
            else:
                print("[PFKV] Faithful reuse unavailable (non-Llama or multi-GPU) — "
                      "use_pfkv falls back to full in-context prefill (== GraphRAG-R).")

        # ── Multi-GPU data-parallel setup ────────────────────────────────
        self._parallel_mode = parallel_mode
        self._replica_models: List[Tuple] = []

        if parallel_mode == "data_parallel" and torch.cuda.is_available():
            if gpu_ids is not None:
                dp_ids = [int(x.strip()) for x in gpu_ids.split(",")]
            elif num_gpus is not None:
                dp_ids = list(range(min(num_gpus, torch.cuda.device_count())))
            else:
                dp_ids = list(range(torch.cuda.device_count()))

            self._replica_models.append((self.model, self.device))

            if len(dp_ids) > 1:
                rep_kwargs: Dict = {}
                if self._use_auth_token:
                    rep_kwargs["token"] = self._use_auth_token
                for gpu_id in dp_ids[1:]:
                    rep_dev = torch.device(f"cuda:{gpu_id}")
                    print(f"[DataParallel] Loading replica on cuda:{gpu_id} ...")
                    replica = AutoModelForCausalLM.from_pretrained(
                        self.model_name, torch_dtype=self._model_dtype,
                        use_safetensors=True, **rep_kwargs,
                    ).to(rep_dev)
                    replica.eval()
                    self._replica_models.append((replica, rep_dev))
                print(f"[DataParallel] {len(self._replica_models)} replica(s) ready on GPUs {dp_ids}")
            else:
                print("[DataParallel] Only 1 GPU — running single-GPU")
        elif parallel_mode == "model_parallel":
            if hasattr(self.model, "hf_device_map"):
                print(f"[ModelParallel] Layer map: {self.model.hf_device_map}")

        # ── Per-replica PFKV caches ──────────────────────────────────────
        self._replica_pfkv_caches: List = []
        if self.use_pfkv and len(self._replica_models) > 1:
            self._replica_pfkv_caches.append(self.pfkv_cache)
            for r_model, r_device in self._replica_models[1:]:
                r_dev_str = str(r_device)
                if SUBGRAPH_KV_AVAILABLE:
                    r_cache = SubgraphKVCache(r_model, self.tokenizer, r_dev_str, self._pfkv_cache_size)
                else:
                    r_cache = PathFirstKVCache(r_model, self.tokenizer, r_dev_str, self._pfkv_cache_size)
                self._replica_pfkv_caches.append(r_cache)
        elif self.use_pfkv:
            self._replica_pfkv_caches = [self.pfkv_cache]

        self.input_device = self._get_input_device()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def _base_model(self):
        return self.model

    def _layer_device(self, layer_idx: int) -> torch.device:
        """Return the device that hosts layer `layer_idx`.

        With device_map='auto' on a large model (e.g. Qwen3-32B) the layers
        are spread across GPUs.  Every (k, v) pair inserted into DynamicCache
        must live on the same device as the layer that will consume it —
        otherwise torch.cat inside the attention forward raises a device-
        mismatch error.
        """
        if hasattr(self.model, "hf_device_map") and self.model.hf_device_map:
            dm = self.model.hf_device_map
            # Try common key patterns: "model.layers.N", "layers.N", etc.
            for pattern in (
                f"model.layers.{layer_idx}",
                f"layers.{layer_idx}",
                f"decoder.layers.{layer_idx}",
                f"transformer.h.{layer_idx}",
                f"h.{layer_idx}",
            ):
                if pattern in dm:
                    dev = dm[pattern]
                    return torch.device(f"cuda:{dev}") if isinstance(dev, int) else torch.device(dev)
            # Fallback: scan for any key that contains the layer index
            for key, dev in dm.items():
                if f".{layer_idx}" in key or f"/{layer_idx}" in key:
                    return torch.device(f"cuda:{dev}") if isinstance(dev, int) else torch.device(dev)
        return self.input_device

    def _get_input_device(self) -> torch.device:
        if hasattr(self.model, "hf_device_map") and self.model.hf_device_map:
            embed_keys = ["embed_tokens", "embed", "wte", "word_embedding", "decoder.embed"]
            for key, dev in self.model.hf_device_map.items():
                for ek in embed_keys:
                    if ek in key:
                        return torch.device(f"cuda:{dev}") if isinstance(dev, int) else torch.device(dev)
            first = list(self.model.hf_device_map.values())[0]
            return torch.device(f"cuda:{first}") if isinstance(first, int) else torch.device(first)
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self.device

    # Prefixes the Alpaca-fine-tuned RoG model sometimes prepends before the answer.
    _ROG_STRIP_PREFIXES = (
        "the answer is:",
        "the answer is",
        "answer:",
        "answer is:",
        "answer is",
        "based on the knowledge graph paths,",
        "based on the knowledge graph paths",
        "based on the paths,",
        "based on the paths",
    )

    @staticmethod
    def _extract_answer(generated_text: str, prompt: str, tokenizer, outputs,
                        model_name: str = "") -> str:
        """Parse the model's raw generated text into a clean answer string."""
        if "rog" in model_name.lower():
            # RoG (Alpaca fine-tune) generates the answer right after
            # ### Response: with no extra scaffolding in the well-trained case.
            # Take the first non-empty, non-header line, then strip any common
            # verbose prefix the model might add before the actual entity name.
            for line in generated_text.split("\n"):
                line = line.strip()
                if not line or line.startswith("###"):
                    continue
                low = line.lower()
                for prefix in PathFirstGraphRAGGeneration._ROG_STRIP_PREFIXES:
                    if low.startswith(prefix):
                        line = line[len(prefix):].strip(" .,")
                        break
                if line:
                    return line
            # Nothing usable found — fall through to generic extraction below.

        answer = re.split(r"\n|Final Answer:|Answer:|A:|Q:", generated_text)[0].strip()
        if not answer:
            full = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = (full[len(prompt):] if full.startswith(prompt) else full).strip()
            answer = answer[:200].strip()
        return answer

    # ------------------------------------------------------------------
    # Standard single-GPU generation
    # ------------------------------------------------------------------

    def generate_batch_standard(self, prompts: List[str]) -> List[str]:
        """Standard greedy / beam generation (no PFKV). Supports data-parallel.

        Prompts are processed as a single padded batch so the GPU sees all
        sequences in parallel — this is what drives throughput scaling with
        batch size.  Falls back to sequential if tokenizer has no pad token.
        """
        if self._parallel_mode == "data_parallel" and len(self._replica_models) > 1:
            return self._generate_data_parallel(prompts)

        # ── True GPU batching ─────────────────────────────────────────────
        if len(prompts) > 1 and self.tokenizer.pad_token_id is not None:
            # Left-pad so all sequences end at the same position (decoder-only)
            self.tokenizer.padding_side = "left"
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.input_device)

            input_len = inputs["input_ids"].shape[1]
            stopping = FirstTokenStoppingCriteria()

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            stopping.set_start_time(t0)

            with torch.no_grad():
                outputs = self._base_model.generate(
                    **inputs,
                    use_cache=self.use_kv_cache,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=self.num_beams,
                    do_sample=False,
                    stopping_criteria=StoppingCriteriaList([stopping]),
                )

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            self.generation_times.append(time.perf_counter() - t0)
            self.ttfts.append(
                stopping.first_token_time - t0 if stopping.first_token_time else 0
            )

            predictions = []
            _debug_printed = False
            for i, prompt in enumerate(prompts):
                generated_text = self.tokenizer.decode(
                    outputs[i, input_len:], skip_special_tokens=True
                ).strip()
                if not _debug_printed:
                    _debug_printed = True
                    print("\n[DEBUG] First prompt (last 300 chars):")
                    print(f"  ...{prompt[-300:]!r}")
                    print(f"[DEBUG] Raw generated text: {generated_text!r}\n")
                predictions.append(self._extract_answer(
                    generated_text, prompt, self.tokenizer, outputs, self.model_name))
            return predictions

        # ── Fallback: sequential (single prompt or no pad token) ──────────
        predictions = []
        _debug_printed = False
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.input_device)
            stopping = FirstTokenStoppingCriteria()

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            stopping.set_start_time(t0)

            with torch.no_grad():
                outputs = self._base_model.generate(
                    **inputs,
                    use_cache=self.use_kv_cache,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=self.num_beams,
                    do_sample=False,
                    stopping_criteria=StoppingCriteriaList([stopping]),
                )

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            self.generation_times.append(time.perf_counter() - t0)
            self.ttfts.append(
                stopping.first_token_time - t0 if stopping.first_token_time else 0
            )

            generated_text = self.tokenizer.decode(
                outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()

            if not _debug_printed:
                _debug_printed = True
                print("\n[DEBUG] First prompt (last 300 chars):")
                print(f"  ...{prompt[-300:]!r}")
                print(f"[DEBUG] Raw generated text: {generated_text!r}\n")

            predictions.append(self._extract_answer(
                generated_text, prompt, self.tokenizer, outputs, self.model_name))

        return predictions

    # ------------------------------------------------------------------
    # Data-parallel generation (no PFKV)
    # ------------------------------------------------------------------

    def _generate_data_parallel(self, prompts: List[str]) -> List[str]:
        n_gpus = len(self._replica_models)
        splits: List[List[Tuple[int, str]]] = [[] for _ in range(n_gpus)]
        for i, p in enumerate(prompts):
            splits[i % n_gpus].append((i, p))

        timing_lock = threading.Lock()

        def run_on_gpu(pairs, model, device):
            local = []
            for orig_idx, prompt in pairs:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
                input_len = inputs["input_ids"].size(1)
                stopping = FirstTokenStoppingCriteria()
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                stopping.set_start_time(t0)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        use_cache=self.use_kv_cache,
                        max_new_tokens=self.max_new_tokens,
                        num_beams=self.num_beams,
                        do_sample=False,
                        stopping_criteria=StoppingCriteriaList([stopping]),
                    )
                torch.cuda.synchronize(device)
                gen_time = time.perf_counter() - t0
                with timing_lock:
                    self.generation_times.append(gen_time)
                    self.ttfts.append(stopping.first_token_time - t0 if stopping.first_token_time else 0)
                text = self.tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True).strip()
                local.append((orig_idx, self._extract_answer(
                    text, prompt, self.tokenizer, outputs, self.model_name)))
            return local

        all_results: List[Tuple[int, str]] = []
        with ThreadPoolExecutor(max_workers=n_gpus) as pool:
            futures = [pool.submit(run_on_gpu, splits[i], m, d)
                       for i, (m, d) in enumerate(self._replica_models)]
            for f in as_completed(futures):
                all_results.extend(f.result())

        all_results.sort(key=lambda x: x[0])
        return [ans for _, ans in all_results]

    # ------------------------------------------------------------------
    # PFKV-aware generation
    # ------------------------------------------------------------------

    def _generate_single_with_pfkv(
        self,
        prompt: str,
        query_paths: List,
        query_idx: int,
    ) -> str:
        """Run one PFKV-assisted generate() call for a single query.

        Called only for queries that have a cache hit — all other queries are
        batched together by generate_batch_with_pfkv.
        """
        cached_paths = [
            p for p in query_paths
            if self.pfkv_cache.path_to_str(p) in self.pfkv_cache.batch_shared_cache
        ]

        inputs    = self.tokenizer(prompt, return_tensors="pt").to(self.input_device)
        input_len = inputs["input_ids"].size(1)

        cached_kv_states, cached_tokens = [], 0
        if cached_paths and self.using_subgraph_cache:
            cached_kv_states, cached_tokens, _ = self.pfkv_cache.get_realigned_kv_for_paths(
                cached_paths, start_position=0, max_paths=10)
        elif cached_paths:
            cached_kv_states, cached_tokens = self.pfkv_cache.get_cached_kv_for_paths(
                cached_paths, current_position=0)

        stopping = FirstTokenStoppingCriteria()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        stopping.set_start_time(t0)

        with torch.no_grad():
            try:
                gen_kwargs: dict = {
                    "input_ids":        inputs["input_ids"],
                    "attention_mask":   inputs["attention_mask"],
                    "use_cache":        True,
                    "max_new_tokens":   self.max_new_tokens,
                    "num_beams":        self.num_beams,
                    "do_sample":        False,
                    "stopping_criteria": StoppingCriteriaList([stopping]),
                }
                if cached_kv_states and cached_tokens > 0:
                    n_layers = self.pfkv_cache.num_layers
                    if len(cached_kv_states) == n_layers:
                        all_valid = all(
                            k is not None and v is not None
                            and k.dim() == 4 and v.dim() == 4
                            for k, v in cached_kv_states
                        )
                        if all_valid:
                            cached_seq_len = cached_kv_states[0][0].shape[2]
                            ext_mask = torch.ones(
                                (inputs["input_ids"].shape[0], cached_seq_len + input_len),
                                dtype=torch.long, device=self.input_device,
                            )
                            dyn = DynamicCache()
                            for li, (k, v) in enumerate(cached_kv_states):
                                ld = self._layer_device(li)
                                dyn.update(k.to(ld).contiguous(), v.to(ld).contiguous(), layer_idx=li)
                            gen_kwargs["past_key_values"] = dyn
                            gen_kwargs["attention_mask"]  = ext_mask
                            print(f"[PFKV] Query {query_idx+1}: reusing {cached_tokens} cached tokens")
                outputs = self._base_model.generate(**gen_kwargs)
            except Exception as e:
                print(f"[PFKV] Generation error: {e} — falling back")
                outputs = self._base_model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    use_cache=self.use_kv_cache,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=self.num_beams,
                    do_sample=False,
                    stopping_criteria=StoppingCriteriaList([stopping]),
                )

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.generation_times.append(time.perf_counter() - t0)
        self.ttfts.append(stopping.first_token_time - t0 if stopping.first_token_time else 0)

        generated_text = self.tokenizer.decode(
            outputs[0, input_len:], skip_special_tokens=True).strip()
        return self._extract_answer(generated_text, prompt, self.tokenizer, outputs, self.model_name)

    def generate_batch_with_pfkv(
        self,
        prompts: List[str],
        batch_paths: List[List[List[Tuple[str, str, str]]]],
    ) -> List[str]:
        """Generation with Subgraph KV caching (paper Section 5).

        Key insight for throughput:
          - Queries WITH a PFKV cache hit carry heterogeneous past_key_values
            (different lengths per query) → cannot be batched → sequential.
          - Queries WITHOUT a cache hit have no past_key_values and can be
            padded + batched into a single generate() call just like the
            standard path → true GPU batching → throughput scales with N.

        This split is what makes the graphrag_opt condition genuinely faster
        than the baselines at large batch sizes.
        """
        if not self.use_pfkv or not self.pfkv_cache:
            return self.generate_batch_standard(prompts)
        if self._parallel_mode == "data_parallel" and len(self._replica_models) > 1:
            return self._generate_pfkv_data_parallel(prompts, batch_paths)

        self.pfkv_cache.prepare_batch_cache(batch_paths)

        # ── Partition into PFKV-cached vs plain ──────────────────────────
        pfkv_indices:  List[int] = []
        plain_indices: List[int] = []

        for i, query_paths in enumerate(batch_paths):
            has_cache_hit = any(
                self.pfkv_cache.path_to_str(p) in self.pfkv_cache.batch_shared_cache
                for p in query_paths
            )
            if has_cache_hit:
                pfkv_indices.append(i)
            else:
                plain_indices.append(i)

        predictions: List[str] = [""] * len(prompts)

        # ── Batch-generate the plain queries (no cache hit) ───────────────
        if plain_indices:
            plain_preds = self.generate_batch_standard(
                [prompts[i] for i in plain_indices])
            for i, pred in zip(plain_indices, plain_preds):
                predictions[i] = pred

        # ── Sequential PFKV-assisted generation for cache-hit queries ─────
        for i in pfkv_indices:
            predictions[i] = self._generate_single_with_pfkv(
                prompts[i], batch_paths[i], i)

        self.pfkv_cache.clear_batch_cache()
        return predictions

    def _generate_pfkv_data_parallel(
        self,
        prompts: List[str],
        batch_paths: List[List[List[Tuple[str, str, str]]]],
    ) -> List[str]:
        """Data-parallel PFKV generation (one cache per GPU replica)."""
        n_gpus = len(self._replica_models)
        splits: List[List[Tuple[int, str, list]]] = [[] for _ in range(n_gpus)]
        for i, (p, paths) in enumerate(zip(prompts, batch_paths)):
            splits[i % n_gpus].append((i, p, paths))

        timing_lock = threading.Lock()

        def run_pfkv_on_gpu(triples, model, device, pfkv_cache):
            pfkv_cache.prepare_batch_cache([qp for _, _, qp in triples])
            results = []
            for orig_idx, prompt, query_paths in triples:
                cached_paths = [
                    p for p in query_paths
                    if pfkv_cache.path_to_str(p) in pfkv_cache.batch_shared_cache
                ]
                inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
                input_len = inputs["input_ids"].size(1)

                cached_kv_states, cached_tokens = [], 0
                if cached_paths and self.using_subgraph_cache:
                    cached_kv_states, cached_tokens, _ = pfkv_cache.get_realigned_kv_for_paths(
                        cached_paths, start_position=0, max_paths=10)
                elif cached_paths:
                    cached_kv_states, cached_tokens = pfkv_cache.get_cached_kv_for_paths(
                        cached_paths, current_position=0)

                stopping = FirstTokenStoppingCriteria()
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                stopping.set_start_time(t0)

                with torch.no_grad():
                    try:
                        gen_kwargs = {
                            "input_ids": inputs["input_ids"],
                            "attention_mask": inputs["attention_mask"],
                            "use_cache": True,
                            "max_new_tokens": self.max_new_tokens,
                            "num_beams": self.num_beams,
                            "do_sample": False,
                            "stopping_criteria": StoppingCriteriaList([stopping]),
                        }
                        if cached_kv_states and cached_tokens > 0:
                            n_layers = pfkv_cache.num_layers
                            if len(cached_kv_states) == n_layers:
                                all_valid = all(
                                    k is not None and v is not None and k.dim() == 4 and v.dim() == 4
                                    for k, v in cached_kv_states
                                )
                                if all_valid:
                                    cached_seq_len = cached_kv_states[0][0].shape[2]
                                    ext_mask = torch.ones(
                                        (inputs["input_ids"].shape[0], cached_seq_len + input_len),
                                        dtype=torch.long, device=device,
                                    )
                                    dyn = DynamicCache()
                                    for li, (k, v) in enumerate(cached_kv_states):
                                        ld = self._layer_device(li)
                                        dyn.update(k.to(ld).contiguous(), v.to(ld).contiguous(), layer_idx=li)
                                    gen_kwargs["past_key_values"] = dyn
                                    gen_kwargs["attention_mask"] = ext_mask
                        outputs = model.generate(**gen_kwargs)
                    except Exception as e:
                        print(f"[PFKV+DP] Error on {device}: {e}")
                        outputs = model.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            use_cache=self.use_kv_cache,
                            max_new_tokens=self.max_new_tokens,
                            num_beams=self.num_beams,
                            do_sample=False,
                            stopping_criteria=StoppingCriteriaList([stopping]),
                        )

                torch.cuda.synchronize(device)
                gen_time = time.perf_counter() - t0
                with timing_lock:
                    self.generation_times.append(gen_time)
                    self.ttfts.append(stopping.first_token_time - t0 if stopping.first_token_time else 0)

                text = self.tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True).strip()
                results.append((orig_idx, self._extract_answer(
                    text, prompt, self.tokenizer, outputs, self.model_name)))

            pfkv_cache.clear_batch_cache()
            return results

        all_results: List[Tuple[int, str]] = []
        with ThreadPoolExecutor(max_workers=n_gpus) as pool:
            futures = [
                pool.submit(run_pfkv_on_gpu, splits[i], m, d, self._replica_pfkv_caches[i])
                for i, (m, d) in enumerate(self._replica_models)
            ]
            for f in as_completed(futures):
                all_results.extend(f.result())

        all_results.sort(key=lambda x: x[0])
        return [ans for _, ans in all_results]

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def generate_batch_faithful(self, queries: List[str], batch_paths: List) -> List[str]:
        """Faithful subgraph KV reuse (Llama-family): each path is represented
        once, as cached KV at its correct interleaved position, with selective
        recompute (self.recompute_policy). No double-representation."""
        eos = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []
        preds: List[str] = []
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for q, paths in zip(queries, batch_paths):
            ans, _ = faithful_generate(
                self.faithful_engine, self.tokenizer, q, paths,
                policy=self.recompute_policy, max_paths=self.pfkv_max_paths,
                max_new_tokens=min(self.max_new_tokens, 32), eos_ids=eos,
                template=self._pfkv_template, path_cache=self._pfkv_path_cache)
            preds.append(ans)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.generation_times.append(time.perf_counter() - t0)
        return preds

    def generate_batch(
        self,
        prompts: List[str],
        batch_paths: Optional[List[List[List[Tuple[str, str, str]]]]] = None,
        queries: Optional[List[str]] = None,
    ) -> List[str]:
        # Faithful reuse (honest, no double-representation) for Llama-family.
        if self.using_faithful_pfkv and batch_paths is not None and queries is not None:
            return self.generate_batch_faithful(queries, batch_paths)
        # use_pfkv requested but faithful unavailable (non-Llama / multi-GPU):
        # honest fallback is full in-context prefill (== GraphRAG-R), NOT the
        # legacy double-representation path.
        if self.use_pfkv and not self.using_faithful_pfkv and batch_paths is not None:
            return self.generate_batch_standard(prompts)
        return self.generate_batch_standard(prompts)

    # ------------------------------------------------------------------
    # Evaluation (delegates to metrics.py)
    # ------------------------------------------------------------------

    def evaluate_metrics(
        self,
        predictions: List[str],
        ground_truths: List[Union[str, List[str]]],
    ) -> Dict[str, float]:
        """Standard NeurIPS KGQA metrics: Hits@1, F1, legacy Acc."""
        return _evaluate_metrics(predictions, ground_truths)

    def evaluate_accuracy(
        self,
        predictions: List[str],
        ground_truths: List[Union[str, List[str]]],
    ) -> float:
        """Legacy wrapper — returns Hits@1 (the correct primary metric)."""
        return _evaluate_metrics(predictions, ground_truths)["hits_at_1"]

    # Keep old normalization alias so any calling code still works
    def normalize_text(self, s: str) -> str:
        return normalize_answer(s)

    normalize_answer = staticmethod(normalize_answer)
