"""
Path-First KV Cache implementation (legacy / fallback).

Used when subgraph_kv_cache.SubgraphKVCache is not available.
Inspired by the KVLink paper; optimized for GPU memory efficiency.

Classes:
  CachedPath       – dataclass holding one cached path's KV states
  PathFirstKVCache – cache manager for path-level KV reuse
"""

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from graph_utils import path_to_str as _path_to_str


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class CachedPath:
    """One cached path with its KV states and metadata."""
    path: List[Tuple[str, str, str]]          # (head, relation, tail) triples
    tokens: List[int]                          # tokenized path
    kv_states: List[Tuple[torch.Tensor, torch.Tensor]]  # per-layer (K, V)
    original_positions: List[int]              # original positional indices
    path_str: str                              # dot-joined string (cache key)


# ---------------------------------------------------------------------------
# Cache manager
# ---------------------------------------------------------------------------

class PathFirstKVCache:
    """
    Path-First KV Cache.
    Caches KV states per unique graph path so repeated paths share prefill work.
    """

    def __init__(self, model, tokenizer, device: str = "cuda", max_cache_size: int = 1000):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.max_cache_size = max_cache_size

        self.path_cache: Dict[str, CachedPath] = {}
        self.batch_shared_cache: Dict[str, CachedPath] = {}

        self.static_cache_size = 512
        self.static_kv_cache: Optional[List] = None
        self.cache_position = 0

        self.stats: Dict = {
            "cache_hits": 0,
            "cache_misses": 0,
            "paths_cached": 0,
            "batch_cache_reuse": 0,
            "memory_saved_tokens": 0,
            "gpu_memory_allocated": 0,
            "cache_overhead_time": 0.0,
        }

        self.num_layers = (
            len(self.model.transformer.h)
            if hasattr(self.model, "transformer")
            else len(self.model.model.layers)
        )
        self.num_heads = self.model.config.num_attention_heads
        self.head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        self._initialize_static_cache()
        print(f"PathFirstKVCache initialized: {self.num_layers} layers, "
              f"{self.num_heads} heads, {self.head_dim} head_dim")

    # ------------------------------------------------------------------
    # Static cache
    # ------------------------------------------------------------------

    def _initialize_static_cache(self) -> None:
        try:
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            self.static_kv_cache = []
            for _ in range(self.num_layers):
                self.static_kv_cache.append((
                    torch.zeros(1, self.num_heads, self.static_cache_size, self.head_dim,
                                device=self.device, dtype=dtype),
                    torch.zeros(1, self.num_heads, self.static_cache_size, self.head_dim,
                                device=self.device, dtype=dtype),
                ))
            print(f"[PFKV] Static KV cache: {self.num_layers} layers × {self.static_cache_size} tokens")
            if torch.cuda.is_available():
                self.stats["gpu_memory_allocated"] = (
                    torch.cuda.memory_allocated(self.device) / 1024 ** 3
                )
        except Exception as e:
            print(f"[PFKV] Static cache init failed: {e}")
            self.static_kv_cache = None

    def _update_static_cache(self, kv_states: List[Tuple[torch.Tensor, torch.Tensor]],
                              tokens: List[int]) -> None:
        if not self.static_kv_cache or not kv_states:
            return
        seq_len = kv_states[0][0].shape[2] if kv_states[0][0].dim() == 4 else 0
        if seq_len == 0 or seq_len > self.static_cache_size:
            return
        if self.cache_position + seq_len > self.static_cache_size:
            self.cache_position = 0
        start_pos, end_pos = self.cache_position, self.cache_position + seq_len
        try:
            for layer_idx, (key, value) in enumerate(kv_states):
                if layer_idx < len(self.static_kv_cache):
                    ck, cv = self.static_kv_cache[layer_idx]
                    ck[:, :, start_pos:end_pos, :] = key[:, :, :seq_len, :]
                    cv[:, :, start_pos:end_pos, :] = value[:, :, :seq_len, :]
            self.cache_position = end_pos
        except Exception as e:
            print(f"[PFKV] Static cache update failed: {e}")

    # ------------------------------------------------------------------
    # Path ↔ string helpers (delegate to graph_utils for consistency)
    # ------------------------------------------------------------------

    def path_to_str(self, path: List[Tuple[str, str, str]]) -> str:
        return _path_to_str(path)

    def tokenize_path(self, path: List[Tuple[str, str, str]]) -> List[int]:
        s = self.path_to_str(path)
        return self.tokenizer.encode(s, add_special_tokens=False) if s else []

    # ------------------------------------------------------------------
    # KV extraction
    # ------------------------------------------------------------------

    def extract_kv_states_for_path(self, path_tokens: List[int]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        if not path_tokens:
            return []
        ids = torch.tensor([path_tokens], device=self.device, dtype=torch.long)
        mask = torch.ones_like(ids, device=self.device)
        try:
            with torch.no_grad():
                outputs = self.model(input_ids=ids, attention_mask=mask,
                                     use_cache=True, return_dict=True)
            if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
                kv_states = []
                for layer_idx, layer_kv in enumerate(outputs.past_key_values):
                    if isinstance(layer_kv, (tuple, list)) and len(layer_kv) == 2:
                        k, v = layer_kv
                        if k.dim() == 4 and v.dim() == 4:
                            kv_states.append((k.clone().detach(), v.clone().detach()))
                        else:
                            return []
                    else:
                        return []
                return kv_states if len(kv_states) == self.num_layers else []
        except Exception as e:
            print(f"[PFKV] KV extraction failed: {e}")
        return []

    def _apply_rope_delta(self, key_tensor: torch.Tensor, position_delta: int) -> torch.Tensor:
        """Apply an incremental RoPE rotation Δ to already-rotated keys."""
        try:
            if position_delta == 0:
                return key_tensor
            batch_size, num_heads, seq_len, head_dim = key_tensor.shape
            theta = float(getattr(self.model.config, "rope_theta", 10000.0))
            device, dtype = key_tensor.device, key_tensor.dtype
            inv_freq = 1.0 / (theta ** (
                torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
            ))
            delta = torch.tensor(float(position_delta), device=device, dtype=torch.float32)
            t = delta.unsqueeze(0) * inv_freq.unsqueeze(0)
            cos_t = torch.cos(t).to(dtype).view(1, 1, 1, -1).expand(batch_size, num_heads, seq_len, -1)
            sin_t = torch.sin(t).to(dtype).view(1, 1, 1, -1).expand(batch_size, num_heads, seq_len, -1)
            x1, x2 = key_tensor[..., :head_dim:2], key_tensor[..., 1:head_dim:2]
            rotated = torch.empty_like(key_tensor)
            rotated[..., :head_dim:2] = x1 * cos_t - x2 * sin_t
            rotated[..., 1:head_dim:2] = x1 * sin_t + x2 * cos_t
            return rotated
        except Exception as e:
            print(f"[PFKV] RoPE Δ failed: {e}")
            return key_tensor

    # ------------------------------------------------------------------
    # Cache population
    # ------------------------------------------------------------------

    def cache_path(self, path: List[Tuple[str, str, str]]) -> str:
        path_str = self.path_to_str(path)
        if not path_str or path_str in self.path_cache:
            return path_str
        path_tokens = self.tokenize_path(path)
        if not path_tokens:
            return path_str
        try:
            kv_states = self.extract_kv_states_for_path(path_tokens)
            if not kv_states:
                return path_str
            if self.static_kv_cache and len(path_tokens) < 100:
                self._update_static_cache(kv_states, path_tokens)
            cached = CachedPath(
                path=path, tokens=path_tokens, kv_states=kv_states,
                original_positions=list(range(len(path_tokens))), path_str=path_str,
            )
            self.path_cache[path_str] = cached
            self.stats["paths_cached"] += 1
            if len(self.path_cache) > self.max_cache_size:
                del self.path_cache[next(iter(self.path_cache))]
        except Exception as e:
            print(f"[PFKV] Failed to cache path {path_str}: {e}")
        return path_str

    def _store_cached_path(self, path_str: str, path, kv_states) -> None:
        path_tokens = self.tokenize_path(path)
        cached = CachedPath(
            path=path, tokens=path_tokens, kv_states=kv_states,
            original_positions=list(range(len(path_tokens))), path_str=path_str,
        )
        self.path_cache[path_str] = cached
        self.stats["paths_cached"] += 1
        if torch.cuda.is_available():
            self.stats["gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1024 ** 3

    def _cache_path_individual(self, path_str: str, path) -> bool:
        try:
            tokens = self.tokenize_path(path)
            if not tokens:
                return False
            kv_states = self.extract_kv_states_for_path(tokens)
            if kv_states:
                self._store_cached_path(path_str, path, kv_states)
                return True
        except Exception:
            pass
        return False

    def _lightweight_cache_paths(self, paths_to_process: List[Tuple[str, list]]) -> int:
        cached_count = 0
        for path_str, path in paths_to_process:
            try:
                tokens = self.tokenize_path(path)
                if not tokens:
                    continue
                kv_states = self.extract_kv_states_for_path(tokens)
                if kv_states:
                    cached = CachedPath(
                        path=path, tokens=tokens, kv_states=kv_states,
                        original_positions=list(range(len(tokens))), path_str=path_str,
                    )
                    self.path_cache[path_str] = cached
                    self.stats["paths_cached"] += 1
                    cached_count += 1
            except Exception as e:
                print(f"[PFKV] Failed to cache {path_str[:30]}: {e}")
        return cached_count

    # ------------------------------------------------------------------
    # Batch-level cache preparation
    # ------------------------------------------------------------------

    def prepare_batch_cache(self, all_paths: List[List[List[Tuple[str, str, str]]]]) -> None:
        t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        path_frequency: Dict[str, int] = defaultdict(int)
        unique_paths: Dict[str, list] = {}

        for query_paths in all_paths:
            seen = set()
            for path in query_paths:
                ps = self.path_to_str(path)
                if ps and ps not in seen:
                    seen.add(ps)
                    path_frequency[ps] += 1
                    unique_paths[ps] = path

        shared = {ps: cnt for ps, cnt in path_frequency.items() if cnt > 1}
        redundancy = len(shared) / len(path_frequency) if path_frequency else 0

        if redundancy < 0.01:
            self.stats["cache_overhead_time"] += time.time() - t0
            return

        high_value = {ps: cnt for ps, cnt in shared.items() if cnt >= 2}
        if not high_value:
            self.stats["cache_overhead_time"] += time.time() - t0
            return

        to_process = [
            (ps, unique_paths[ps])
            for ps in high_value
            if ps not in self.path_cache and ps in unique_paths
        ]

        if to_process:
            self._lightweight_cache_paths(to_process)

        for ps, cnt in high_value.items():
            if ps in self.path_cache:
                self.stats["batch_cache_reuse"] += cnt - 1

        self.batch_shared_cache = {
            ps: self.path_cache[ps]
            for ps in high_value
            if ps in self.path_cache
        }

        self.stats["cache_overhead_time"] += time.time() - t0

    # ------------------------------------------------------------------
    # KV retrieval for generation
    # ------------------------------------------------------------------

    def get_cached_kv_for_paths(
        self,
        paths: List[List[Tuple[str, str, str]]],
        current_position: int = 0,
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], int]:
        if not paths:
            return [], 0

        layer_keys: List[List[torch.Tensor]] = [[] for _ in range(self.num_layers)]
        layer_values: List[List[torch.Tensor]] = [[] for _ in range(self.num_layers)]
        total_cached = 0

        for path in paths[:5]:
            ps = self.path_to_str(path)
            if ps not in self.path_cache:
                self.stats["cache_misses"] += 1
                continue
            cached = self.path_cache[ps]
            self.stats["cache_hits"] += 1
            self.stats["memory_saved_tokens"] += len(cached.tokens)
            total_cached += len(cached.tokens)

            if not cached.kv_states or len(cached.kv_states) != self.num_layers:
                continue
            for li, (k, v) in enumerate(cached.kv_states):
                if k is None or v is None:
                    continue
                k, v = k.to(self.device), v.to(self.device)
                if k.dim() == 4 and v.dim() == 4:
                    layer_keys[li].append(k)
                    layer_values[li].append(v)

        final_kv: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for li in range(self.num_layers):
            if layer_keys[li] and layer_values[li]:
                try:
                    k_cat = torch.cat(layer_keys[li], dim=2)
                    v_cat = torch.cat(layer_values[li], dim=2)
                    if k_cat.shape[0] == 1 and v_cat.shape[0] == 1:
                        final_kv.append((k_cat, v_cat))
                    else:
                        return [], 0
                except Exception:
                    return [], 0
            else:
                return [], 0

        return final_kv, total_cached

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def clear_batch_cache(self) -> None:
        self.batch_shared_cache.clear()

    def get_cache_statistics(self) -> Dict:
        total = self.stats["cache_hits"] + self.stats["cache_misses"]
        return {
            **self.stats,
            "cache_size": len(self.path_cache),
            "batch_cache_size": len(self.batch_shared_cache),
            "hit_rate": self.stats["cache_hits"] / total if total > 0 else 0.0,
            "total_requests": total,
        }
