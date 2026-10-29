"""Tests of hf_offload.py on tiny random Qwen3-MoE / GLM-4.5 models (one GPU, < 1 GB):
  - no substitution: the offloaded model's logits / greedy tokens equal the in-memory model's, with pools smaller than
    a prefill needs (staging path) and large enough for decode;
  - substitution: the change rate stays within the budget, and nothing changes in prefill.

  CUDA_VISIBLE_DEVICES=2 .venv/bin/python experiments/moe/test_hf_offload.py
"""
import copy
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hf_offload as ho  # noqa: E402


def tiny(kind):
    torch.manual_seed(0)
    if kind == "qwen3":
        from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
        cfg = Qwen3MoeConfig(vocab_size=256, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
                             num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                             num_experts=16, num_experts_per_tok=4, norm_topk_prob=True, decoder_sparse_step=1,
                             mlp_only_layers=[])
        m = Qwen3MoeForCausalLM(cfg)
    else:
        from transformers import Glm4MoeConfig, Glm4MoeForCausalLM
        cfg = Glm4MoeConfig(vocab_size=256, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
                            num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                            n_routed_experts=16, num_experts_per_tok=4, n_shared_experts=1, n_group=1, topk_group=1,
                            first_k_dense_replace=1, norm_topk_prob=True, routed_scaling_factor=1.0,
                            partial_rotary_factor=0.5)
        m = Glm4MoeForCausalLM(cfg)
        for layer in m.model.layers:                         # a non-trivial correction bias
            g = getattr(getattr(layer, "mlp", None), "gate", None)
            if g is not None and hasattr(g, "e_score_correction_bias"):
                g.e_score_correction_bias.data = torch.randn_like(g.e_score_correction_bias) * 0.1
    return m.to(torch.bfloat16).eval()


def forced(model, ids, toks, dev):
    """Decode logits [len(toks), V] feeding the same tokens (prefill, then one token per step with the cache)."""
    out = model(torch.tensor([ids], device=dev), use_cache=True)
    cache, res = out.past_key_values, [out.logits[0, -1].float()]
    for t in toks[:-1]:
        out = model(torch.tensor([[t]], device=dev), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        res.append(out.logits[0, -1].float())
    return torch.stack(res)


def greedy(model, ids, n, dev):
    out = model.generate(torch.tensor([ids], device=dev), max_new_tokens=n, do_sample=False, min_new_tokens=n)
    return out[0, len(ids):].tolist()


def check(kind, impl):
    dev = "cuda:0"
    ref = tiny(kind)
    off = copy.deepcopy(ref)
    ref = ref.to(dev)
    ref.config._experts_implementation = impl
    blocks = ho.moe_blocks(off)
    L, E = len(blocks), blocks[0].experts.gate_up_proj.shape[0]
    rng = np.random.default_rng(1)
    counts = rng.random((L, E))
    caps = np.full(L, 6)                                     # < 16 experts: prefill uses the staging path
    eng = ho.HFOffload.attach(off, caps, dev)
    off.config._experts_implementation = impl
    eng.warm(counts)
    ids = rng.integers(0, 256, 24).tolist()
    with torch.no_grad():
        a = ref(torch.tensor([ids], device=dev)).logits.float()
        b = off(torch.tensor([ids], device=dev)).logits.float()
    err = (a - b).abs().max().item()
    assert err < 1e-2, (kind, impl, "prefill logits", err)
    assert eng.stats["staged"] > 0, "staging path not exercised"
    # greedy decode: prefill then decode steps; the engine runs decode on its pools (<= 4 experts per step)
    eng.subst = ho.Subst("none", L, E)                     # routing patch active in decode; none never changes
    g_ref = greedy(ref, ids, 16, dev)
    if impl == "grouped_mm":                              # sums each token's k outputs in k order: bit-identical
        g_off = greedy(off, ids, 16, dev)
        assert g_ref == g_off, (kind, impl, g_ref, g_off)
    else:                                                  # eager adds experts in id order: slot ids reorder the sum
        with torch.no_grad():
            d = (forced(ref, ids, g_ref, dev) - forced(off, ids, g_ref, dev)).abs().max().item()
        assert d < 5e-2, (kind, impl, "teacher-forced decode logits", d)
    assert eng.subst.changed == 0 and eng.subst.late > 0
    # substitution: change rate within budget; late experts exist (pools of 6 of 16)
    eng.warm(counts)
    eng.subst = ho.Subst("rate:0.1", L, E)
    greedy(off, ids, 32, dev)
    sm = eng.subst.summary()
    assert eng.subst.uses == 31 * L * 4, eng.subst.uses     # decode steps only (the prefill is never changed)
    assert 0 < sm["change_rate"] <= 0.1 + 1 / eng.subst.uses, sm     # random rounding: at most one change ahead
    # fast decode path (plan 4.2.6): no substitution equals the in-memory model; with rate:0.1 it makes the same
    # decisions as the per-row path, so the generation is identical
    if impl == "grouped_mm":
        eng.fast = True
        eng.warm(counts)
        eng.subst = ho.Subst("none", L, E)
        assert greedy(off, ids, 16, dev) == g_ref, (kind, "fast path, no substitution")
        outs = []
        for fast in (False, True):
            eng.fast = fast
            eng.warm(counts)
            eng.subst = ho.Subst("rate:0.1", L, E)
            outs.append((greedy(off, ids, 32, dev), eng.subst.summary()["change_rate"], eng.stats["loads"]))
        assert outs[0] == outs[1], (kind, "fast vs per-row path with substitution", outs)
        eng.fast = False
    print(f"ok {kind} {impl}: prefill logits max err {err:.2e}, greedy equal, subst rate:0.1 -> changed "
          f"{sm['change_rate']:.3f} (late {sm['late_rate']:.3f}), loads {eng.stats['loads']}")


if __name__ == "__main__":
    for k in ("qwen3", "glm"):
        for impl in ("eager", "grouped_mm"):
            check(k, impl)
