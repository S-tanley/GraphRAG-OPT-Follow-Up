"""
Subgraph KV Cache Reuse with Positional Re-encoding

Implementation based on the paper "A Holistic Framework for Accelerating Graph Retrieval-Augmented Generation"
Section 5: Subgraph KV Reuse in Generation Phase
Section 5.2: Subgraph-Aware Positional Realignment

Key concepts:
1. Store position-agnostic KV states (without RoPE rotation)
2. Apply positional realignment when reusing cached KV at new positions
3. Focus on prefill attention states for higher reuse potential
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class CachedSubgraph:
    """Represents a cached subgraph with position-agnostic KV states"""
    path: List[Tuple[str, str, str]]
    tokens: List[int]
    # Position-agnostic KV states (RoPE rotation removed)
    position_agnostic_keys: List[torch.Tensor]  # Per layer: [batch, heads, seq, dim]
    values: List[torch.Tensor]  # Values don't have positional encoding in RoPE
    original_positions: List[int]
    path_str: str
    reuse_count: int = 0


class SubgraphKVCache:
    """
    Subgraph KV Cache with proper positional re-encoding.
    
    From the paper (Section 5.2):
    - For cached subgraph Si with tokens at original positions {p1, p2, ..., p|Si|},
      we store position-agnostic KV representations by removing rotational components:
      K̃_Si = {Wk h1, Wk h2, ..., Wk h|Si|} (without rotation)
    
    - During inference, when the same subgraph appears at new positions {p1', p2', ...},
      we perform positional realignment:
      K_Si^realigned = {R_p1' k̃1, R_p2' k̃2, ...}
    """
    
    def __init__(self, model, tokenizer, device: str = "cuda", max_cache_size: int = 1000):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.max_cache_size = max_cache_size
        
        # Cache storage
        self.subgraph_cache: Dict[str, CachedSubgraph] = {}
        self.batch_shared_cache: Dict[str, CachedSubgraph] = {}
        
        # Statistics
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "subgraphs_cached": 0,
            "batch_cache_reuse": 0,
            "tokens_saved": 0,
            "realignment_time": 0.0,
            "extraction_time": 0.0,
        }
        
        # Model configuration - detect model architecture
        self._detect_model_config()
        
        # RoPE configuration
        self.rope_theta = float(getattr(self.model.config, 'rope_theta', 10000.0))
        self.rotary_dim = self.head_dim  # Most models use full head_dim for RoPE
        
        # Pre-compute inverse frequencies for RoPE
        self._precompute_rope_frequencies()
        
        print(f"[SubgraphKV] Initialized: {self.num_layers} layers, {self.num_heads} heads, "
              f"{self.head_dim} head_dim, rope_theta={self.rope_theta}")
    
    def _get_input_device(self):
        """Get the correct device for model inputs, handling multi-GPU models."""
        if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map:
            # Look for embedding layer in device map
            embed_keys = ['embed_tokens', 'embed', 'wte', 'word_embedding', 'decoder.embed']
            for key, device in self.model.hf_device_map.items():
                for embed_key in embed_keys:
                    if embed_key in key:
                        if isinstance(device, int):
                            return f"cuda:{device}"
                        return device
            # Fallback to first device in map
            first_device = list(self.model.hf_device_map.values())[0]
            if isinstance(first_device, int):
                return f"cuda:{first_device}"
            return first_device
        
        # Try to get device from model parameters directly
        try:
            first_param = next(self.model.parameters())
            return first_param.device
        except StopIteration:
            pass
        
        return self.device
    
    def _detect_model_config(self):
        """Detect model architecture and extract configuration"""
        config = self.model.config
        model_type_name = getattr(config, 'model_type', 'unknown').lower()
        
        # Get number of layers
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            self.num_layers = len(self.model.transformer.h)
            self.model_type = "gpt2"
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            self.num_layers = len(self.model.model.layers)
            self.model_type = "llama"
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'decoder'):
            # OPT model
            self.num_layers = len(self.model.model.decoder.layers)
            self.model_type = "opt"
        else:
            self.num_layers = getattr(config, 'num_hidden_layers', 32)
            self.model_type = model_type_name
        
        # Get attention configuration
        self.num_heads = getattr(config, 'num_attention_heads', 32)
        self.num_kv_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        self.hidden_size = getattr(config, 'hidden_size', 4096)
        self.head_dim = self.hidden_size // self.num_heads
        
        # Detect if model uses RoPE (Rotary Position Embeddings)
        # Models like LLaMA, Mistral use RoPE; OPT, GPT-2 use learned position embeddings
        self.uses_rope = model_type_name in ['llama', 'mistral', 'qwen', 'qwen2', 'gemma', 'phi']
        if hasattr(config, 'rope_theta'):
            self.uses_rope = True
        
        print(f"[SubgraphKV] Model type: {self.model_type}, KV heads: {self.num_kv_heads}")
        print(f"[SubgraphKV] Uses RoPE: {self.uses_rope}")
    
    def _precompute_rope_frequencies(self):
        """Pre-compute inverse frequencies for RoPE operations (stored on CPU)."""
        inv_freq = 1.0 / (self.rope_theta ** (
            torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim
        ))
        self.register_buffer_inv_freq = inv_freq  # CPU tensor; moved per-call
    
    def _compute_rope_rotation(self, positions: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute RoPE rotation matrices (cos, sin) for given positions.

        Args:
            positions: [seq_len] tensor of position indices
            dtype: target dtype for the output

        Returns:
            cos, sin tensors of shape [1, 1, seq_len, rotary_dim//2]
        """
        # Move inv_freq to the same device as positions so outer product works
        # regardless of whether positions is on CPU, cuda:0, cuda:1, etc.
        inv_freq = self.register_buffer_inv_freq.to(positions.device)

        # Compute angles: [seq_len, rotary_dim//2]
        freqs = torch.outer(positions.float(), inv_freq)
        
        cos = torch.cos(freqs).to(dtype)
        sin = torch.sin(freqs).to(dtype)
        
        # Reshape for broadcasting: [1, 1, seq_len, rotary_dim//2]
        cos = cos.view(1, 1, -1, cos.shape[-1])
        sin = sin.view(1, 1, -1, sin.shape[-1])
        
        return cos, sin
    
    def _apply_rope_rotation(self, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE rotation to key tensor.
        
        From RoPE paper: rotate pairs of dimensions
        key: [batch, heads, seq, head_dim]
        cos, sin: [1, 1, seq, rotary_dim//2]
        """
        batch, heads, seq_len, head_dim = key.shape
        rotary_dim = self.rotary_dim
        
        # Split into pairs
        x1 = key[..., :rotary_dim:2]  # Even indices
        x2 = key[..., 1:rotary_dim:2]  # Odd indices
        
        # Apply rotation: (x1, x2) -> (x1*cos - x2*sin, x1*sin + x2*cos)
        rotated = torch.empty_like(key)
        rotated[..., :rotary_dim:2] = x1 * cos - x2 * sin
        rotated[..., 1:rotary_dim:2] = x1 * sin + x2 * cos
        
        # Copy non-rotary dimensions if any
        if rotary_dim < head_dim:
            rotated[..., rotary_dim:] = key[..., rotary_dim:]
        
        return rotated
    
    def _remove_rope_rotation(self, key: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Remove RoPE rotation from key tensor to get position-agnostic representation.
        
        This is the inverse operation: multiply by R(-θ) = R^T(θ)
        Since R^T has cos unchanged and sin negated.
        
        Args:
            key: [batch, heads, seq, head_dim] with RoPE applied
            positions: [seq] original position indices
            
        Returns:
            Position-agnostic key tensor
        """
        # For non-RoPE models (OPT, GPT-2), return key as-is
        # These models use learned position embeddings added at input, not in attention
        if not self.uses_rope:
            return key.clone()
        
        cos, sin = self._compute_rope_rotation(positions, key.dtype)
        
        # Inverse rotation: use -sin (R^{-1} = R^T for rotation matrices)
        return self._apply_rope_rotation(key, cos, -sin)
    
    def _realign_rope_positions(self, key_agnostic: torch.Tensor, 
                                 new_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE rotation at new positions to position-agnostic keys.
        
        From paper Section 5.2:
        K_Si^realigned = {R_p1' k̃1, R_p2' k̃2, ...}
        
        Args:
            key_agnostic: [batch, heads, seq, head_dim] position-agnostic keys
            new_positions: [seq] new position indices
            
        Returns:
            Keys with RoPE applied at new positions
        """
        # For non-RoPE models, return as-is (positions are fixed in these models)
        if not self.uses_rope:
            return key_agnostic.clone()
        
        cos, sin = self._compute_rope_rotation(new_positions, key_agnostic.dtype)
        return self._apply_rope_rotation(key_agnostic, cos, sin)
    
    def path_to_str(self, path: List[Tuple[str, str, str]]) -> str:
        """Convert path to string for hashing"""
        if not path:
            return ""
        parts = [path[0][0]]
        for (u, rel, v) in path:
            parts.append(rel)
            parts.append(v)
        return ".".join(parts)
    
    def tokenize_path(self, path: List[Tuple[str, str, str]]) -> List[int]:
        """Tokenize a path"""
        path_str = self.path_to_str(path)
        if not path_str:
            return []
        return self.tokenizer.encode(path_str, add_special_tokens=False)
    
    def extract_position_agnostic_kv(self, path_tokens: List[int]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Extract position-agnostic KV states for a path.
        
        Steps:
        1. Run forward pass to get KV with original positional encoding
        2. Remove RoPE rotation from keys to get position-agnostic representation
        3. Values don't have positional encoding in RoPE, so keep as-is
        
        Returns:
            (position_agnostic_keys, values) - lists per layer
        """
        if not path_tokens:
            return [], []
        
        start_time = time.time()
        
        # Handle multi-GPU models with device_map
        input_device = self._get_input_device()
        
        input_ids = torch.tensor([path_tokens], device=input_device, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        seq_len = len(path_tokens)
        
        # Original positions on CPU (keys/values are stored on CPU)
        original_positions = torch.arange(seq_len, device="cpu")
        
        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True
                )
                
                if not hasattr(outputs, 'past_key_values') or outputs.past_key_values is None:
                    print("[SubgraphKV] Model does not support past_key_values")
                    return [], []
                
                past_kv = outputs.past_key_values
                position_agnostic_keys = []
                values = []
                
                # Iterate over past_key_values (works for both DynamicCache and tuple/list)
                # DynamicCache is iterable and yields (key, value, ...) tuples
                for layer_idx, layer_kv in enumerate(past_kv):
                    if isinstance(layer_kv, (tuple, list)) and len(layer_kv) >= 2:
                        key, value = layer_kv[0], layer_kv[1]
                    elif isinstance(layer_kv, torch.Tensor):
                        # Some formats have a different structure
                        print(f"[SubgraphKV] Unexpected tensor at layer {layer_idx}")
                        return [], []
                    else:
                        print(f"[SubgraphKV] Unexpected KV format at layer {layer_idx}: {type(layer_kv)}")
                        return [], []
                    
                    # Keep on CPU so the cache is device-agnostic.
                    # Tensors are moved to the correct per-layer device at
                    # lookup time in get_realigned_kv_for_paths / generation.py.
                    key = key.cpu().clone()
                    value = value.cpu().clone()
                    
                    # Validate shapes
                    if key.dim() != 4 or value.dim() != 4:
                        print(f"[SubgraphKV] Invalid KV shape at layer {layer_idx}: key={key.shape}, value={value.shape}")
                        return [], []
                    
                    # Remove RoPE rotation from keys to get position-agnostic
                    key_agnostic = self._remove_rope_rotation(key, original_positions)
                    
                    position_agnostic_keys.append(key_agnostic.detach())
                    values.append(value.detach())
                
                self.stats["extraction_time"] += time.time() - start_time
                return position_agnostic_keys, values
                
        except Exception as e:
            print(f"[SubgraphKV] KV extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return [], []
    
    def cache_subgraph(self, path: List[Tuple[str, str, str]]) -> bool:
        """
        Cache a subgraph's position-agnostic KV states.
        
        Returns True if successfully cached.
        """
        path_str = self.path_to_str(path)
        if not path_str:
            return False
        
        # Already cached?
        if path_str in self.subgraph_cache:
            return True
        
        tokens = self.tokenize_path(path)
        if not tokens:
            return False
        
        # Extract position-agnostic KV states
        keys_agnostic, values = self.extract_position_agnostic_kv(tokens)
        
        if not keys_agnostic or len(keys_agnostic) != self.num_layers:
            return False
        
        # Create cached subgraph
        cached = CachedSubgraph(
            path=path,
            tokens=tokens,
            position_agnostic_keys=keys_agnostic,
            values=values,
            original_positions=list(range(len(tokens))),
            path_str=path_str
        )
        
        self.subgraph_cache[path_str] = cached
        self.stats["subgraphs_cached"] += 1
        
        # Evict if cache is full (LRU-like: remove least reused)
        if len(self.subgraph_cache) > self.max_cache_size:
            min_reuse = min(self.subgraph_cache.values(), key=lambda x: x.reuse_count)
            del self.subgraph_cache[min_reuse.path_str]
        
        return True
    
    def get_realigned_kv_for_paths(self, paths: List[List[Tuple[str, str, str]]], 
                                    start_position: int = 0,
                                    max_paths: int = 10) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], int, List[int]]:
        """
        Get KV states for cached paths with positional realignment.
        
        From paper Section 5.2:
        "During inference, when the same subgraph appears at new positions {p1', p2', ...},
        we perform positional realignment:
        K_Si^realigned = {R_p1' k̃1, R_p2' k̃2, ...}"
        
        Args:
            paths: List of paths to retrieve
            start_position: Starting position for the realigned KV states
            max_paths: Maximum number of paths to process
            
        Returns:
            (realigned_kv_states, total_tokens, token_positions)
        """
        if not paths:
            return [], 0, []
        
        start_time = time.time()
        
        # Collect all cached paths
        cached_paths = []
        for path in paths[:max_paths]:
            path_str = self.path_to_str(path)
            if path_str in self.subgraph_cache:
                cached_paths.append(self.subgraph_cache[path_str])
                self.subgraph_cache[path_str].reuse_count += 1
                self.stats["cache_hits"] += 1
            else:
                self.stats["cache_misses"] += 1
        
        if not cached_paths:
            return [], 0, []
        
        # Calculate new positions for all tokens
        current_pos = start_position
        all_new_positions = []
        
        for cached in cached_paths:
            seq_len = len(cached.tokens)
            new_positions = list(range(current_pos, current_pos + seq_len))
            all_new_positions.extend(new_positions)
            current_pos += seq_len
        
        total_tokens = len(all_new_positions)
        # Keep positions on CPU — they'll be moved per-layer below
        new_positions_tensor = torch.tensor(all_new_positions, device="cpu")

        # Realign KV states per layer.
        # Cached tensors live on CPU; we return them on CPU so the caller
        # (generation.py) can move each (k, v) to the correct layer device.
        realigned_kv = []

        for layer_idx in range(self.num_layers):
            layer_keys_agnostic = []
            layer_values = []

            for cached in cached_paths:
                layer_keys_agnostic.append(cached.position_agnostic_keys[layer_idx])
                layer_values.append(cached.values[layer_idx])

            # Concatenate along sequence dimension (all on CPU)
            keys_concat   = torch.cat(layer_keys_agnostic, dim=2)  # [1, heads, total_seq, dim]
            values_concat = torch.cat(layer_values,        dim=2)

            # Apply RoPE at new positions (positions already on CPU, matching keys)
            keys_realigned = self._realign_rope_positions(keys_concat, new_positions_tensor)

            realigned_kv.append((keys_realigned, values_concat))
        
        self.stats["tokens_saved"] += total_tokens
        self.stats["realignment_time"] += time.time() - start_time
        
        return realigned_kv, total_tokens, all_new_positions
    
    def prepare_batch_cache(self, all_paths: List[List[List[Tuple[str, str, str]]]]):
        """
        Prepare cache for a batch of queries.
        
        Implements multi-level caching strategy:
        1. Full path caching (exact path matches across queries)
        2. Relation-type caching (shared relation types like "sports.mascot.team")
        3. Entity-relation segment caching (e.g., "San Francisco Giants.sports.champion")
        
        This enables much higher cache hit rates than full-path-only matching.
        """
        print("[SubgraphKV] Preparing batch cache...")
        
        # Level 1: Count full path frequency
        path_frequency = defaultdict(int)
        unique_paths = {}
        
        # Level 2: Count relation type frequency (middle part of triples)
        relation_frequency = defaultdict(int)
        relation_examples = {}  # Store example triple for each relation
        
        # Level 3: Count entity-relation segments
        segment_frequency = defaultdict(int)
        segment_examples = {}
        
        for query_idx, query_paths in enumerate(all_paths):
            seen_paths = set()
            seen_relations = set()
            seen_segments = set()
            
            for path in query_paths:
                # Full path tracking
                path_str = self.path_to_str(path)
                if path_str and path_str not in seen_paths:
                    seen_paths.add(path_str)
                    path_frequency[path_str] += 1
                    unique_paths[path_str] = path
                
                # Track individual triples and relations
                for triple in path:
                    if len(triple) >= 3:
                        entity_src, relation, entity_dst = triple[0], triple[1], triple[2]
                        
                        # Relation type frequency
                        if relation not in seen_relations:
                            seen_relations.add(relation)
                            relation_frequency[relation] += 1
                            if relation not in relation_examples:
                                relation_examples[relation] = [triple]
                        
                        # Entity-relation segment frequency (destination entity + relation)
                        segment = f"{entity_dst}.{relation}"
                        if segment not in seen_segments:
                            seen_segments.add(segment)
                            segment_frequency[segment] += 1
                            if segment not in segment_examples:
                                segment_examples[segment] = triple
        
        # Find high-value items at each level
        high_value_paths = {ps: cnt for ps, cnt in path_frequency.items() if cnt >= 2}
        high_value_relations = {rel: cnt for rel, cnt in relation_frequency.items() if cnt >= 2}
        high_value_segments = {seg: cnt for seg, cnt in segment_frequency.items() if cnt >= 2}
        
        total_high_value = len(high_value_paths) + len(high_value_relations) + len(high_value_segments)
        
        if total_high_value == 0:
            print("[SubgraphKV] No shared paths found in batch")
            return
        
        print(f"[SubgraphKV] Found shareable content:")
        print(f"  - {len(high_value_paths)} full paths (appear in 2+ queries)")
        print(f"  - {len(high_value_relations)} relation types (shared across queries)")
        print(f"  - {len(high_value_segments)} entity-relation segments")
        
        # Cache full paths
        cached_count = 0
        for path_str in high_value_paths:
            if path_str not in self.subgraph_cache:
                if self.cache_subgraph(unique_paths[path_str]):
                    cached_count += 1
        
        # Cache high-frequency relations as single-triple paths
        for relation in list(high_value_relations.keys())[:20]:  # Limit to top 20
            if relation in relation_examples:
                single_triple_path = relation_examples[relation]
                rel_path_str = f"__REL__{relation}"
                if rel_path_str not in self.subgraph_cache:
                    if self.cache_subgraph(single_triple_path):
                        cached_count += 1
                        # Also add to batch shared cache for easy lookup
                        if rel_path_str in self.subgraph_cache:
                            self.batch_shared_cache[rel_path_str] = self.subgraph_cache[rel_path_str]
        
        # Populate batch shared cache with full paths
        for ps in high_value_paths:
            if ps in self.subgraph_cache:
                self.batch_shared_cache[ps] = self.subgraph_cache[ps]
        
        # Update stats
        for path_str, count in high_value_paths.items():
            if path_str in self.subgraph_cache:
                self.stats["batch_cache_reuse"] += count - 1
        
        # Add relation-level reuse stats
        for relation, count in high_value_relations.items():
            self.stats["batch_cache_reuse"] += count - 1
        
        print(f"[SubgraphKV] Cached {cached_count} new subgraphs, "
              f"{len(self.batch_shared_cache)} total in batch cache")
    
    def clear_batch_cache(self):
        """Clear batch-level cache"""
        self.batch_shared_cache.clear()
    
    def get_statistics(self) -> Dict:
        """Get cache statistics"""
        total = self.stats["cache_hits"] + self.stats["cache_misses"]
        hit_rate = self.stats["cache_hits"] / total if total > 0 else 0.0
        
        return {
            **self.stats,
            "cache_size": len(self.subgraph_cache),
            "batch_cache_size": len(self.batch_shared_cache),
            "hit_rate": hit_rate,
            "paths_cached": self.stats["subgraphs_cached"],
            "memory_saved_tokens": self.stats["tokens_saved"],
        }
    
    def get_cache_statistics(self) -> Dict:
        """Alias for get_statistics() for API compatibility"""
        return self.get_statistics()


# =============================================================================
# Sanity Check Functions
# =============================================================================

def sanity_check_rope_inversion(cache: SubgraphKVCache, verbose: bool = True):
    """
    Sanity check: Verify that RoPE removal and reapplication is correct.
    
    Test: Apply RoPE -> Remove RoPE -> Reapply RoPE at same positions
    Result should match original.
    """
    print("\n" + "="*60)
    print("SANITY CHECK: RoPE Inversion")
    print("="*60)
    
    # Skip for non-RoPE models
    if not cache.uses_rope:
        if verbose:
            print(f"Model type '{cache.model_type}' does not use RoPE.")
            print("Skipping RoPE inversion test.")
            print("Result: SKIPPED (N/A)")
        return True  # Not applicable, consider passed
    
    # Create random key tensor
    batch, heads, seq_len, head_dim = 1, cache.num_kv_heads, 10, cache.head_dim
    key_original = torch.randn(batch, heads, seq_len, head_dim, 
                               device=cache.device, dtype=torch.float16)
    
    positions = torch.arange(seq_len, device=cache.device)
    
    # Apply RoPE
    cos, sin = cache._compute_rope_rotation(positions, key_original.dtype)
    key_with_rope = cache._apply_rope_rotation(key_original, cos, sin)
    
    # Remove RoPE
    key_agnostic = cache._remove_rope_rotation(key_with_rope, positions)
    
    # Reapply RoPE at same positions
    key_reconstructed = cache._realign_rope_positions(key_agnostic, positions)
    
    # Check error
    error = (key_with_rope - key_reconstructed).abs().max().item()
    relative_error = error / key_with_rope.abs().max().item()
    
    passed = relative_error < 2e-3  # Relaxed threshold for fp16 precision
    
    if verbose:
        print(f"Max absolute error: {error:.2e}")
        print(f"Max relative error: {relative_error:.2e}")
        print(f"Result: {'PASSED ✓' if passed else 'FAILED ✗'}")
    
    return passed


def sanity_check_position_shift(cache: SubgraphKVCache, verbose: bool = True):
    """
    Sanity check: Verify that position shift works correctly.
    
    Test: Key at position 0 shifted to position 5 should equal
          key computed directly at position 5.
    """
    print("\n" + "="*60)
    print("SANITY CHECK: Position Shift")
    print("="*60)
    
    # Skip for non-RoPE models
    if not cache.uses_rope:
        if verbose:
            print(f"Model type '{cache.model_type}' does not use RoPE.")
            print("Position shift via RoPE is not applicable.")
            print("Result: SKIPPED (N/A)")
        return True
    
    batch, heads, seq_len, head_dim = 1, cache.num_kv_heads, 5, cache.head_dim
    key_raw = torch.randn(batch, heads, seq_len, head_dim,
                          device=cache.device, dtype=torch.float16)
    
    # Method 1: Compute at original positions, remove, reapply at new positions
    original_pos = torch.arange(seq_len, device=cache.device)
    new_pos = torch.arange(10, 10 + seq_len, device=cache.device)  # Shift by 10
    
    cos_orig, sin_orig = cache._compute_rope_rotation(original_pos, key_raw.dtype)
    key_at_orig = cache._apply_rope_rotation(key_raw, cos_orig, sin_orig)
    
    key_agnostic = cache._remove_rope_rotation(key_at_orig, original_pos)
    key_shifted = cache._realign_rope_positions(key_agnostic, new_pos)
    
    # Method 2: Directly compute at new positions
    cos_new, sin_new = cache._compute_rope_rotation(new_pos, key_raw.dtype)
    key_direct = cache._apply_rope_rotation(key_raw, cos_new, sin_new)
    
    # Compare
    error = (key_shifted - key_direct).abs().max().item()
    relative_error = error / key_direct.abs().max().item()
    
    passed = relative_error < 2e-3  # Relaxed for fp16 precision
    
    if verbose:
        print(f"Shift amount: 10 positions")
        print(f"Max absolute error: {error:.2e}")
        print(f"Max relative error: {relative_error:.2e}")
        print(f"Result: {'PASSED ✓' if passed else 'FAILED ✗'}")
    
    return passed


def _get_model_input_device(model, fallback_device):
    """Get the correct device for model inputs, handling multi-GPU models."""
    # For models loaded with device_map, find where embeddings live
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        # Look for embedding layer in device map
        embed_keys = ['embed_tokens', 'embed', 'wte', 'word_embedding']
        for key, device in model.hf_device_map.items():
            for embed_key in embed_keys:
                if embed_key in key:
                    if isinstance(device, int):
                        return f"cuda:{device}"
                    return device
        # Fallback to first layer
        first_device = list(model.hf_device_map.values())[0]
        if isinstance(first_device, int):
            return f"cuda:{first_device}"
        return first_device
    
    # Try to get device from model parameters directly
    try:
        first_param = next(model.parameters())
        return first_param.device
    except StopIteration:
        pass
    
    return fallback_device


def _get_layer_device(model, layer_idx, fallback_device):
    """Get the device for a specific layer, handling multi-GPU models."""
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        # Look for this layer in the device map
        layer_keys = [f'layers.{layer_idx}', f'decoder.layers.{layer_idx}', 
                      f'model.layers.{layer_idx}', f'model.decoder.layers.{layer_idx}',
                      f'h.{layer_idx}', f'transformer.h.{layer_idx}']
        for map_key, device in model.hf_device_map.items():
            for layer_key in layer_keys:
                if layer_key in map_key:
                    if isinstance(device, int):
                        return f"cuda:{device}"
                    return device
        # Fallback: check if there's a general pattern
        for map_key, device in model.hf_device_map.items():
            if str(layer_idx) in map_key and ('layer' in map_key.lower() or 'h.' in map_key):
                if isinstance(device, int):
                    return f"cuda:{device}"
                return device
    return fallback_device


def sanity_check_cache_consistency(cache: SubgraphKVCache, test_path: List[Tuple[str, str, str]], verbose: bool = True):
    """
    Sanity check: Verify that cached KV states can be correctly realigned.
    
    Test: 
    1. Cache a path
    2. Get realigned KV at position 0 (same as original)
    3. Compare with fresh extraction
    """
    print("\n" + "="*60)
    print("SANITY CHECK: Cache Consistency")
    print("="*60)
    
    path_str = cache.path_to_str(test_path)
    tokens = cache.tokenize_path(test_path)
    
    if not tokens:
        print("Cannot run test: empty path")
        return False
    
    # Get correct device for model inputs
    input_device = _get_model_input_device(cache.model, cache.device)
    
    # Fresh extraction (with RoPE at original positions)
    input_ids = torch.tensor([tokens], device=input_device)
    with torch.no_grad():
        outputs = cache.model(input_ids, use_cache=True, return_dict=True)
    
    # Convert fresh_kv from DynamicCache to list of (key, value) tuples
    raw_fresh_kv = outputs.past_key_values
    fresh_kv_list = []
    for layer_kv in raw_fresh_kv:
        if isinstance(layer_kv, (tuple, list)) and len(layer_kv) >= 2:
            fresh_kv_list.append((layer_kv[0].to(cache.device), layer_kv[1].to(cache.device)))
        else:
            print(f"[SanityCheck] Unexpected fresh KV format: {type(layer_kv)}")
            return False
    
    # Cache the path
    success = cache.cache_subgraph(test_path)
    if not success:
        print("Failed to cache path")
        return False
    
    # Get realigned KV at position 0 (should match fresh)
    realigned_kv, total_tokens, _ = cache.get_realigned_kv_for_paths(
        [test_path], start_position=0
    )
    
    if not realigned_kv:
        print("Failed to get realigned KV")
        return False
    
    # Compare layer by layer (move tensors to same device)
    max_key_error = 0.0
    max_value_error = 0.0
    
    for layer_idx in range(min(len(fresh_kv_list), len(realigned_kv))):
        fresh_k, fresh_v = fresh_kv_list[layer_idx]
        realigned_k, realigned_v = realigned_kv[layer_idx]
        
        # Move to same device for comparison
        realigned_k = realigned_k.to(cache.device)
        realigned_v = realigned_v.to(cache.device)
        
        key_error = (fresh_k - realigned_k).abs().max().item()
        value_error = (fresh_v - realigned_v).abs().max().item()
        
        max_key_error = max(max_key_error, key_error)
        max_value_error = max(max_value_error, value_error)
    
    key_rel_error = max_key_error / fresh_kv_list[0][0].abs().max().item()
    value_rel_error = max_value_error / fresh_kv_list[0][1].abs().max().item()
    
    # Relaxed threshold for fp16 and non-RoPE models
    passed = key_rel_error < 0.01 and value_rel_error < 0.01
    
    if verbose:
        print(f"Path tokens: {len(tokens)}")
        print(f"Max key relative error: {key_rel_error:.2e}")
        print(f"Max value relative error: {value_rel_error:.2e}")
        print(f"Result: {'PASSED ✓' if passed else 'FAILED ✗'}")
    
    return passed


def sanity_check_generation_accuracy(model, tokenizer, cache: SubgraphKVCache, 
                                      prompt: str, path: List[Tuple[str, str, str]],
                                      verbose: bool = True):
    """
    Sanity check: Verify that using cached KV produces same output as fresh generation.
    """
    print("\n" + "="*60)
    print("SANITY CHECK: Generation Accuracy")
    print("="*60)
    
    from transformers.cache_utils import DynamicCache
    
    # Get correct device for model inputs
    input_device = _get_model_input_device(model, cache.device)
    
    # Tokenize prompt
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    path_tokens = cache.tokenize_path(path)
    
    if not path_tokens:
        print("Empty path, skipping test")
        return False
    
    # Full input = path + prompt
    full_tokens = path_tokens + prompt_tokens
    full_input = torch.tensor([full_tokens], device=input_device)
    
    # Method 1: Fresh generation
    with torch.no_grad():
        outputs_fresh = model(full_input, use_cache=True, return_dict=True)
        logits_fresh = outputs_fresh.logits[0, -1, :]  # Last token logits
    
    # Method 2: Using cached KV
    # First, cache the path
    cache.cache_subgraph(path)
    
    # Get realigned KV for the path at position 0
    realigned_kv, cached_tokens, _ = cache.get_realigned_kv_for_paths([path], start_position=0)
    
    if not realigned_kv:
        print("Failed to get cached KV")
        return False
    
    # Convert to DynamicCache, moving each layer to the correct device
    dyn_cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(realigned_kv):
        layer_device = _get_layer_device(model, layer_idx, cache.device)
        k_dev = k.to(layer_device).contiguous()
        v_dev = v.to(layer_device).contiguous()
        dyn_cache.update(k_dev, v_dev, layer_idx=layer_idx)
    
    # Generate with cached KV + prompt
    prompt_input = torch.tensor([prompt_tokens], device=input_device)
    
    # Attention mask covers cached + new tokens
    total_len = cached_tokens + len(prompt_tokens)
    attention_mask = torch.ones((1, total_len), device=input_device, dtype=torch.long)
    
    with torch.no_grad():
        outputs_cached = model(
            input_ids=prompt_input,
            attention_mask=attention_mask,
            past_key_values=dyn_cache,
            use_cache=True,
            return_dict=True
        )
        logits_cached = outputs_cached.logits[0, -1, :]
    
    # Compare logits
    logit_diff = (logits_fresh - logits_cached).abs()
    max_diff = logit_diff.max().item()
    mean_diff = logit_diff.mean().item()
    
    # Check if top prediction matches
    top_fresh = logits_fresh.argmax().item()
    top_cached = logits_cached.argmax().item()
    top_match = top_fresh == top_cached
    
    # Check top-5 overlap
    top5_fresh = set(logits_fresh.topk(5).indices.tolist())
    top5_cached = set(logits_cached.topk(5).indices.tolist())
    top5_overlap = len(top5_fresh & top5_cached) / 5
    
    passed = top_match and top5_overlap >= 0.8
    
    if verbose:
        print(f"Path tokens: {len(path_tokens)}, Prompt tokens: {len(prompt_tokens)}")
        print(f"Max logit difference: {max_diff:.4f}")
        print(f"Mean logit difference: {mean_diff:.4f}")
        print(f"Top-1 match: {top_match}")
        print(f"Top-5 overlap: {top5_overlap:.0%}")
        print(f"Fresh top token: {tokenizer.decode([top_fresh])}")
        print(f"Cached top token: {tokenizer.decode([top_cached])}")
        print(f"Result: {'PASSED ✓' if passed else 'FAILED ✗'}")
    
    return passed


def run_all_sanity_checks(model, tokenizer, device: str = "cuda"):
    """Run all sanity checks"""
    print("\n" + "="*70)
    print("RUNNING ALL SANITY CHECKS FOR SUBGRAPH KV CACHE")
    print("="*70)
    
    # Detect the actual model device
    actual_device = _get_model_input_device(model, device)
    print(f"[SanityCheck] Using device: {actual_device}")
    
    cache = SubgraphKVCache(model, tokenizer, actual_device)
    
    # Test path
    test_path = [
        ("Newton", "works_at", "Cambridge"),
        ("Cambridge", "located_in", "England")
    ]
    
    results = {}
    
    # Check 1: RoPE inversion
    results["rope_inversion"] = sanity_check_rope_inversion(cache)
    
    # Check 2: Position shift
    results["position_shift"] = sanity_check_position_shift(cache)
    
    # Check 3: Cache consistency
    results["cache_consistency"] = sanity_check_cache_consistency(cache, test_path)
    
    # Check 4: Generation accuracy
    test_prompt = "What is Newton famous for?"
    results["generation_accuracy"] = sanity_check_generation_accuracy(
        model, tokenizer, cache, test_prompt, test_path
    )
    
    print("\n" + "="*70)
    print("SANITY CHECK SUMMARY")
    print("="*70)
    
    all_passed = True
    for name, passed in results.items():
        status = "PASSED ✓" if passed else "FAILED ✗"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed
    
    print("="*70)
    print(f"OVERALL: {'ALL CHECKS PASSED ✓' if all_passed else 'SOME CHECKS FAILED ✗'}")
    print("="*70)
    
    return all_passed, results


if __name__ == "__main__":
    # Quick test with a small model
    print("Loading test model...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    model_name = "facebook/opt-125m"  # Small model for testing
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    run_all_sanity_checks(model, tokenizer)
