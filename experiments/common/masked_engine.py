"""Masked-prefill engine: arbitrary attention mask + decoupled positions over the context.

The prompt is split into
  context  : preamble P + retrieved-path region (flat lines, or a deduplicated trie)
  suffix   : question + generated tokens (always attend to every context token)

For the context the caller supplies
  pos_int  : positions used for the context's OWN forward pass (q/k rotation inside the context)
  pos_out  : positions at which the context keys are READ by the suffix / decode tokens
  mask     : bool [n, n], mask[i, j] = token i may attend to token j
so that M0 (flat causal), M1-* (block-diagonal), M2-* (trie) and GraphRAG-Opt's
"hybrid" reuse (computed at one set of positions, read at another) are all one engine.

Optional `full_layers`: set of layer indices in which the context instead uses `full_mask`
with positions `pos_out` (used for the "top-l layers full attention" variants).

Works with HF Llama-family and Qwen3 modules (uses each layer's own projections, norms,
rotary embedding and MLP; Qwen3 q_norm/k_norm are applied when present).
"""

import torch
import torch.nn.functional as F


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


class MaskedEngine:
    def __init__(self, model):
        self.model = model
        self.core = model.model
        self.layers = self.core.layers
        self.rot = self.core.rotary_emb
        cfg = model.config
        self.hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.dev = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.qk_norm = hasattr(self.layers[0].self_attn, "q_norm")

    # ---------------------------------------------------------------- helpers
    def _cos_sin(self, pos):
        pos = torch.as_tensor(pos, device=self.dev, dtype=torch.long)
        dummy = torch.zeros(1, device=self.dev, dtype=self.dtype)
        cos, sin = self.rot(dummy, pos[None])
        return cos[0], sin[0]                                   # [n, hd]

    @staticmethod
    def _rope(x, cos, sin):                                     # x [H, n, hd]
        return x * cos[None] + _rotate_half(x) * sin[None]

    def _qkv(self, layer, xn):
        a = layer.self_attn
        n = xn.shape[0]
        q = a.q_proj(xn).view(n, -1, self.hd).transpose(0, 1)   # [H, n, hd]
        k = a.k_proj(xn).view(n, -1, self.hd).transpose(0, 1)   # [KVH, n, hd]
        v = a.v_proj(xn).view(n, -1, self.hd).transpose(0, 1)
        if self.qk_norm:
            q, k = a.q_norm(q), a.k_norm(k)
        return q, k, v

    def _attn(self, layer, q, k, v, mask):
        out = F.scaled_dot_product_attention(q[None], k[None], v[None],
                                             attn_mask=mask[None, None], enable_gqa=True)
        return layer.self_attn.o_proj(out[0].transpose(0, 1).reshape(q.shape[1], -1))

    # ---------------------------------------------------------------- context
    @torch.no_grad()
    def encode_context(self, ids, pos_int, mask, pos_out=None, full_layers=None, full_mask=None,
                       ctx_rows=None):
        """Returns per-layer (K read-out keys rotated at pos_out, V) lists.

        ctx_rows (optional bool [n]): rows computed "in context" — their attention uses
        q and ALL keys rotated at pos_out, while the other rows use pos_int. Lets cached
        pieces (local positions) and recomputed tokens (sequential positions) coexist."""
        pos_out = pos_int if pos_out is None else pos_out
        mask = torch.as_tensor(mask, device=self.dev, dtype=torch.bool)
        if full_mask is not None:
            full_mask = torch.as_tensor(full_mask, device=self.dev, dtype=torch.bool)
        if ctx_rows is not None:
            ctx_rows = torch.as_tensor(ctx_rows, device=self.dev, dtype=torch.bool)
        ci, si = self._cos_sin(pos_int)
        co, so = self._cos_sin(pos_out)
        H = self.core.embed_tokens(torch.as_tensor(ids, device=self.dev))
        Ks, Vs = [], []
        for li, layer in enumerate(self.layers):
            xn = layer.input_layernorm(H)
            q, k, v = self._qkv(layer, xn)
            if full_layers and li in full_layers:
                c, s, m = co, so, full_mask
            else:
                c, s, m = ci, si, mask
            a = self._attn(layer, self._rope(q, c, s), self._rope(k, c, s), v, m)
            if ctx_rows is not None and bool(ctx_rows.any()):
                a_ctx = self._attn(layer, self._rope(q, co, so), self._rope(k, co, so), v, m)
                a = torch.where(ctx_rows[:, None], a_ctx, a)
            H = H + a
            H = H + layer.mlp(layer.post_attention_layernorm(H))
            Ks.append(self._rope(k, co, so))
            Vs.append(v)
        return Ks, Vs

    # ---------------------------------------------------------------- suffix / decode
    @torch.no_grad()
    def _step(self, ids, pos, Ks, Vs):
        n = len(ids)
        c, s = self._cos_sin(pos)
        H = self.core.embed_tokens(torch.as_tensor(ids, device=self.dev))
        for li, layer in enumerate(self.layers):
            xn = layer.input_layernorm(H)
            q, k, v = self._qkv(layer, xn)
            Ks[li] = torch.cat([Ks[li], self._rope(k, c, s)], dim=1)
            Vs[li] = torch.cat([Vs[li], v], dim=1)
            nk = Ks[li].shape[1]
            m = torch.ones(n, nk, device=self.dev, dtype=torch.bool)
            m[:, nk - n:] = torch.tril(m[:, nk - n:])
            H = H + self._attn(layer, self._rope(q, c, s), Ks[li], Vs[li], m)
            H = H + layer.mlp(layer.post_attention_layernorm(H))
        return self.model.lm_head(self.core.norm(H[-1:]))[0]

    @torch.no_grad()
    def generate(self, ctx, suf_ids, suf_pos0, max_new_tokens=16, eos_id=None, stop_fn=None, **ctx_kw):
        """ctx: dict(ids, pos_int, mask, [pos_out]) -> greedy token ids.
        stop_fn(ids) -> True stops early (must not change what the caller extracts)."""
        Ks, Vs = self.encode_context(ctx["ids"], ctx["pos_int"], ctx["mask"],
                                     ctx.get("pos_out"), **ctx_kw)
        pos = list(range(suf_pos0, suf_pos0 + len(suf_ids)))
        logits = self._step(suf_ids, pos, Ks, Vs)
        out, cur = [], suf_pos0 + len(suf_ids)
        for _ in range(max_new_tokens):
            tid = int(logits.argmax())
            if eos_id is not None and tid == eos_id:
                break
            out.append(tid)
            if stop_fn is not None and stop_fn(out):
                break
            logits = self._step([tid], [cur], Ks, Vs)
            cur += 1
        return out
