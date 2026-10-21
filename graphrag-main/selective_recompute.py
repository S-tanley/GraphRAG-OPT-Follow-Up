"""
Selective KV recompute for cross-query path KV reuse (reviewer response).

Background
----------
The subgraph KV reuse in this repo caches each path's key/value tensors
*computed in isolation* (the path alone, from position 0) and, at reuse time,
only re-applies the RoPE rotation for the path's new absolute position
(`subgraph_kv_cache.py`).  RoPE realignment fixes the *positional* component of
the keys but NOT the *context-dependent* component of K and V: in a real prompt
the path's tokens attend to the system preamble, the query, and the other
paths, none of which are present when the path is encoded alone.  This is the
exact issue CacheBlend (arXiv:2405.16444) and EPIC/LegoLink (arXiv:2410.15332)
identify; both recover accuracy by *recomputing a small fraction of tokens
in-context*.

This module implements that missing piece in a self-contained, version-robust
way (it reimplements the Llama decoder forward from the model's own weight
tensors, so it does not depend on private transformers internals), and exposes
three reuse policies so the accuracy-vs-recompute trade-off can be measured:

    * RECOMPUTE_NONE   - reuse every path token's cached KV (this repo's current
                         method == CacheBlend's broken "full reuse" baseline).
    * EPIC first-k     - additionally recompute the first k tokens of each path
                         in-context (static positional rule; kills the
                         attention-sink mis-formation).
    * CacheBlend top-r - additionally recompute the r% path tokens whose cached
                         KV deviates most from the in-context KV (dynamic rule).

`recompute=full` recomputes everything and must match a plain `model()` forward;
that equality is the module's correctness gate.

Only standard Llama-family models (RoPE, RMSNorm, SiLU MLP, GQA) are supported,
which covers RoG/Llama-2/TinyLlama used in the paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Llama math primitives (reimplemented from weights for version robustness)
# ---------------------------------------------------------------------------

def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dt = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (weight * x.to(dt))


def _rope_cos_sin(positions: torch.Tensor, head_dim: int, theta: float,
                  device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """HF-Llama RoPE tables (rotate_half convention). Returns [seq, head_dim]."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.outer(positions.float().to(device), inv_freq)      # [seq, hd/2]
    emb = torch.cat([freqs, freqs], dim=-1)                          # [seq, hd]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [seq, heads, hd]; cos/sin: [seq, hd]."""
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    return x * cos + _rotate_half(x) * sin


# ---------------------------------------------------------------------------
# Cached, position-agnostic per-path KV
# ---------------------------------------------------------------------------

@dataclass
class PathKV:
    """Per-path KV computed in isolation. Keys are stored *pre-RoPE* so they can
    be realigned to any absolute position at reuse time."""
    token_ids: List[int]
    k_pre: List[torch.Tensor]   # per-layer [len, n_kv, hd], no RoPE applied
    v: List[torch.Tensor]       # per-layer [len, n_kv, hd]


class LlamaManualPrefill:
    """Manual Llama prefill supporting interleaved cached-KV reuse + recompute."""

    def __init__(self, model):
        cfg = model.config
        assert cfg.model_type == "llama", f"only llama supported, got {cfg.model_type}"
        self.model = model
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.L = cfg.num_hidden_layers
        self.H = cfg.num_attention_heads
        self.KVH = cfg.num_key_value_heads
        self.hd = cfg.hidden_size // cfg.num_attention_heads
        self.eps = getattr(cfg, "rms_norm_eps", 1e-5)
        self.theta = float(getattr(cfg, "rope_theta", None) or 10000.0)
        self.n_rep = self.H // self.KVH
        self._m = model.model

    # -- per-layer projections from weights --------------------------------
    def _attn_proj(self, layer_idx: int, x_norm: torch.Tensor):
        a = self._m.layers[layer_idx].self_attn
        n = x_norm.shape[0]
        q = (x_norm @ a.q_proj.weight.T).view(n, self.H, self.hd)
        k = (x_norm @ a.k_proj.weight.T).view(n, self.KVH, self.hd)
        v = (x_norm @ a.v_proj.weight.T).view(n, self.KVH, self.hd)
        return q, k, v

    def _mlp(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        mlp = self._m.layers[layer_idx].mlp
        return (F.silu(x @ mlp.gate_proj.weight.T) * (x @ mlp.up_proj.weight.T)) @ mlp.down_proj.weight.T

    def embed(self, token_ids: List[int]) -> torch.Tensor:
        ids = torch.tensor(token_ids, device=self.device)
        return self._m.embed_tokens(ids).to(self.dtype)

    # ----------------------------------------------------------------------
    @torch.no_grad()
    def encode_path_isolated(self, token_ids: List[int], prefix_ids=None) -> PathKV:
        """Encode a path and capture its pre-RoPE K, V.

        prefix_ids=None : path alone (positions 0..len-1), reproducing exactly how
                          `subgraph_kv_cache.py` builds its cache.
        prefix_ids given: the path is encoded with that (query-independent) prefix
                          as left-context, occupying positions 0..P-1; only the
                          path's KV slice (positions P..P+L-1) is stored. The
                          path's KV then reflects attention to the prefix and its
                          first token is no longer a position-0 attention sink —
                          a one-time, cross-query-shared improvement to reuse
                          accuracy that needs no per-query recompute."""
        pref = list(prefix_ids) if prefix_ids else []
        P = len(pref)
        all_ids = pref + list(token_ids)
        n = len(all_ids)
        positions = torch.arange(n, device=self.device)
        H = self.embed(all_ids)
        k_pre_all, v_all = [], []
        for li in range(self.L):
            lyr = self._m.layers[li]
            xn = _rms_norm(H, lyr.input_layernorm.weight, self.eps)
            q, k, v = self._attn_proj(li, xn)
            k_pre_all.append(k[P:].clone())   # store path slice only
            v_all.append(v[P:].clone())
            cos, sin = _rope_cos_sin(positions, self.hd, self.theta, self.device, self.dtype)
            qr = _apply_rope(q, cos, sin)
            kr = _apply_rope(k, cos, sin)
            attn = self._attention(qr, kr, v, positions, positions)
            H = H + (attn.reshape(n, -1) @ lyr.self_attn.o_proj.weight.T)
            H = H + self._mlp(li, _rms_norm(H, lyr.post_attention_layernorm.weight, self.eps))
        return PathKV(token_ids=list(token_ids), k_pre=k_pre_all, v=v_all)

    def _attention(self, q, k, v, q_pos, k_pos):
        """q:[nq,H,hd] k,v:[nk,KVH,hd]; causal by absolute position."""
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        q = q.transpose(0, 1)               # [H, nq, hd]
        k = k.transpose(0, 1)               # [H, nk, hd]
        v = v.transpose(0, 1)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.hd)   # [H, nq, nk]
        mask = (k_pos[None, :] <= q_pos[:, None]).to(scores.dtype)  # [nq, nk]
        scores = scores.masked_fill(mask[None] == 0, float("-inf"))
        a = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        out = a @ v                          # [H, nq, hd]
        return out.transpose(0, 1)           # [nq, H, hd]

    # ----------------------------------------------------------------------
    @torch.no_grad()
    def prefill(
        self,
        carried_ids: List[int],
        carried_pos: List[int],
        reuse: Optional[Dict[int, PathKV]] = None,
        reuse_index: Optional[Dict[int, Tuple[int, int]]] = None,
    ) -> torch.Tensor:
        """Run a prefill where `carried` tokens are computed in-context and
        `reuse` positions are served from cached PathKV (RoPE-realigned).

        carried_ids/carried_pos : tokens computed normally (preamble, query,
                                  and any recomputed path tokens).
        reuse                   : maps absolute position -> PathKV holding it.
        reuse_index             : maps absolute position -> (path_local_idx)
                                  via (PathKV, local token index).

        Returns logits for every carried position: [n_carried, vocab].
        """
        reuse = reuse or {}
        reuse_index = reuse_index or {}
        nc = len(carried_ids)
        c_pos = torch.tensor(carried_pos, device=self.device)
        r_pos_sorted = sorted(reuse.keys())
        r_pos = torch.tensor(r_pos_sorted, device=self.device) if r_pos_sorted else None

        H = self.embed(carried_ids)
        for li in range(self.L):
            lyr = self._m.layers[li]
            xn = _rms_norm(H, lyr.input_layernorm.weight, self.eps)
            q, k_c, v_c = self._attn_proj(li, xn)
            cos_c, sin_c = _rope_cos_sin(c_pos, self.hd, self.theta, self.device, self.dtype)
            qr = _apply_rope(q, cos_c, sin_c)
            kr_c = _apply_rope(k_c, cos_c, sin_c)

            if r_pos is not None:
                # Gather cached pre-RoPE K and V for reuse positions, realign K.
                k_pre = torch.stack([reuse[p].k_pre[li][reuse_index[p]] for p in r_pos_sorted])  # [nr,KVH,hd]
                v_r = torch.stack([reuse[p].v[li][reuse_index[p]] for p in r_pos_sorted])
                cos_r, sin_r = _rope_cos_sin(r_pos, self.hd, self.theta, self.device, self.dtype)
                kr_r = _apply_rope(k_pre, cos_r, sin_r)
                all_pos = torch.cat([c_pos, r_pos])
                K = torch.cat([kr_c, kr_r], dim=0)
                V = torch.cat([v_c, v_r], dim=0)
            else:
                all_pos = c_pos
                K, V = kr_c, v_c

            attn = self._attention(qr, K, V, c_pos, all_pos)        # queries = carried only
            H = H + (attn.reshape(nc, -1) @ lyr.self_attn.o_proj.weight.T)
            H = H + self._mlp(li, _rms_norm(H, lyr.post_attention_layernorm.weight, self.eps))

        H = _rms_norm(H, self._m.norm.weight, self.eps)
        return H @ self.model.lm_head.weight.T

    # ----------------------------------------------------------------------
    @torch.no_grad()
    def prefill_full_kv(
        self,
        carried_ids: List[int],
        carried_pos: List[int],
        reuse: Optional[Dict[int, PathKV]] = None,
        reuse_index: Optional[Dict[int, int]] = None,
    ):
        """Like `prefill`, but also returns the ordered full-sequence KV cache so
        decoding can continue. Returns (K_full, V_full, last_logits, N) where
        K_full[l]/V_full[l] are [N, n_kv, hd] ordered by absolute position and
        last_logits is the next-token distribution at the final position."""
        reuse = reuse or {}
        reuse_index = reuse_index or {}
        nc = len(carried_ids)
        c_pos = torch.tensor(carried_pos, device=self.device)
        r_pos_sorted = sorted(reuse.keys())
        r_pos = torch.tensor(r_pos_sorted, device=self.device) if r_pos_sorted else None
        N = max(carried_pos + r_pos_sorted) + 1

        H = self.embed(carried_ids)
        K_full, V_full = [], []
        for li in range(self.L):
            lyr = self._m.layers[li]
            xn = _rms_norm(H, lyr.input_layernorm.weight, self.eps)
            q, k_c, v_c = self._attn_proj(li, xn)
            cos_c, sin_c = _rope_cos_sin(c_pos, self.hd, self.theta, self.device, self.dtype)
            qr = _apply_rope(q, cos_c, sin_c)
            kr_c = _apply_rope(k_c, cos_c, sin_c)

            Kf = torch.zeros(N, self.KVH, self.hd, device=self.device, dtype=self.dtype)
            Vf = torch.zeros(N, self.KVH, self.hd, device=self.device, dtype=self.dtype)
            Kf[c_pos] = kr_c
            Vf[c_pos] = v_c
            if r_pos is not None:
                k_pre = torch.stack([reuse[p].k_pre[li][reuse_index[p]] for p in r_pos_sorted]).to(self.device)
                v_r = torch.stack([reuse[p].v[li][reuse_index[p]] for p in r_pos_sorted]).to(self.device)
                cos_r, sin_r = _rope_cos_sin(r_pos, self.hd, self.theta, self.device, self.dtype)
                kr_r = _apply_rope(k_pre, cos_r, sin_r)
                Kf[r_pos] = kr_r
                Vf[r_pos] = v_r
                all_pos = torch.cat([c_pos, r_pos])
                K = torch.cat([kr_c, kr_r], 0)
                V = torch.cat([v_c, v_r], 0)
            else:
                all_pos = c_pos
                K, V = kr_c, v_c

            attn = self._attention(qr, K, V, c_pos, all_pos)
            H = H + (attn.reshape(nc, -1) @ lyr.self_attn.o_proj.weight.T)
            H = H + self._mlp(li, _rms_norm(H, lyr.post_attention_layernorm.weight, self.eps))
            K_full.append(Kf)
            V_full.append(Vf)

        Hn = _rms_norm(H, self._m.norm.weight, self.eps)
        logits = Hn @ self.model.lm_head.weight.T
        last_local = int(torch.tensor(carried_pos).argmax().item())
        return K_full, V_full, logits[last_local], N

    @torch.no_grad()
    def decode_greedy(self, K_full, V_full, first_token_id, start_pos,
                      max_new_tokens=24, eos_ids=()):
        """Greedy decode appending to an existing ordered KV cache (in place)."""
        eos = {int(e) for e in eos_ids}
        out = [int(first_token_id)]
        tid = int(first_token_id)
        pos = start_pos
        for _ in range(max_new_tokens):
            if tid in eos:
                break
            h = self.embed([tid])                  # [1, hidden]
            p = torch.tensor([pos], device=self.device)
            cos, sin = _rope_cos_sin(p, self.hd, self.theta, self.device, self.dtype)
            for li in range(self.L):
                lyr = self._m.layers[li]
                xn = _rms_norm(h, lyr.input_layernorm.weight, self.eps)
                q, k, v = self._attn_proj(li, xn)
                qr = _apply_rope(q, cos, sin)
                kr = _apply_rope(k, cos, sin)
                K_full[li] = torch.cat([K_full[li], kr], 0)
                V_full[li] = torch.cat([V_full[li], v], 0)
                kpos = torch.arange(pos + 1, device=self.device)
                attn = self._attention(qr, K_full[li], V_full[li], p, kpos)
                h = h + (attn.reshape(1, -1) @ lyr.self_attn.o_proj.weight.T)
                h = h + self._mlp(li, _rms_norm(h, lyr.post_attention_layernorm.weight, self.eps))
            hn = _rms_norm(h, self._m.norm.weight, self.eps)
            tid = int((hn @ self.model.lm_head.weight.T)[0].argmax().item())
            pos += 1
            out.append(tid)
        return out


# ---------------------------------------------------------------------------
# Recompute-policy selection
# ---------------------------------------------------------------------------

def select_recompute_tokens(
    policy: str,
    path_local_lens: List[int],
    k: int = 2,
    ratio: float = 0.15,
    deviation: Optional[List[torch.Tensor]] = None,
) -> List[List[int]]:
    """Return, per path, the list of local token indices to recompute in-context.

    policy:
      'none'       -> [] for every path           (current repo method)
      'epic'       -> first-k indices per path     (EPIC/LegoLink static rule)
      'cacheblend' -> top-`ratio` by `deviation`   (CacheBlend dynamic rule)
      'full'       -> all indices (correctness gate)
    """
    out: List[List[int]] = []
    for pi, n in enumerate(path_local_lens):
        if policy == "none":
            out.append([])
        elif policy == "full":
            out.append(list(range(n)))
        elif policy == "epic":
            out.append(list(range(min(k, n))))
        elif policy == "cacheblend":
            m = max(0, int(round(n * ratio)))
            if m == 0 or deviation is None:
                out.append([])
            else:
                dev = deviation[pi]
                idx = torch.topk(dev, k=min(m, n)).indices.tolist()
                out.append(sorted(idx))
        else:
            raise ValueError(f"unknown policy {policy}")
    return out
