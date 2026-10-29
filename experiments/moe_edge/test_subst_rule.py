"""The three implementations of the use-time substitution rule make the same decisions on the same inputs:
tsim_mu.Subst (simulator), subst_offload.Subst (Mac prototype), hf_offload.Subst (H200 prototype).
Random tokens, random cache state (cached and ready / cached but still loading / not cached), budgets rate / tok.

  .venv-mac/bin/python experiments/moe_edge/test_subst_rule.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "moe"))
import hf_offload  # noqa: E402
import subst_offload  # noqa: E402
import tsim_mu as mu  # noqa: E402


class FakeMacEngine:
    def __init__(self, ready):
        self.ready = ready

    def ready_mask(self, j):
        return self.ready[j].copy()


def check(spec, seed):
    L, E, k, T = 4, 32, 8, 300
    rng = np.random.default_rng(seed)
    res = rng.random((L, E)) < 0.6
    loading = res & (rng.random((L, E)) < 0.15)             # cached but not ready at gate time
    ready = res & ~loading
    eng = mu.Engine(L, E, L * E, mu.HW["h200_pcie"])
    eng.res[:] = res.ravel()
    eng.ready[:] = np.where(loading.ravel(), 1.0, 0.0)
    a = mu.Subst(spec, L, E)
    b = subst_offload.Subst(spec, L, E, seed=0)
    c = hf_offload.Subst(spec, L, E, seed=0)
    mac = FakeMacEngine(ready)
    for d in range(T):
        sc = rng.random((L, E)).astype(np.float32)
        r = dict(w=np.zeros((L, T, k), np.float32), score=np.zeros((L, T, E), np.float32))
        for l in range(L):
            inds = np.argsort(-sc[l], kind="stable")[:k]
            w = sc[l, inds] / sc[l, inds].sum()
            r["w"][l, d], r["score"][l, d] = w, sc[l]
            out_a = a.apply(eng, r, d, l, l * E + inds, 0.0) - l * E
            out_b = b.apply(mac, l, inds[None], w[None], sc[l][None])[0]
            out_c = c.apply(ready[l], l, inds[None], w[None], sc[l][None])[0]
            assert (out_a == out_b).all() and (out_a == out_c).all(), (spec, d, l, out_a, out_b, out_c)
    sa, sb, sc_ = a.summary(), b.summary(), c.summary()
    assert sa["subst_change_rate"] == sb["change_rate"] == sc_["change_rate"]
    assert sa["subst_late_rate"] == sb["late_rate"] == sc_["late_rate"]
    return sa["subst_change_rate"], sa["subst_late_rate"]


def check_fast(spec, seed, N=16):
    """hf_offload's fast path (late order + top-k candidates computed in bulk, Subst.decide for the budget) makes the
    same decisions as Subst.apply on whole batches."""
    L, E, k, T = 4, 32, 8, 120
    rng = np.random.default_rng(seed)
    ready = rng.random((L, E)) < 0.6
    a = hf_offload.Subst(spec, L, E, seed=0)
    b = hf_offload.Subst(spec, L, E, seed=0)
    for _ in range(T):
        for l in range(L):
            sel = rng.random((N, E)).astype(np.float32)
            inds = np.argsort(-sel, 1, kind="stable")[:, :k]
            w = np.take_along_axis(sel, inds, 1)
            w = w / w.sum(1, keepdims=True)
            out_a = a.apply(ready[l], l, inds, w, sel)
            late = ~ready[l][inds]
            order = np.argsort(np.where(late, w, np.inf), 1, kind="stable")
            ok = np.repeat(ready[l][None], N, 0)
            np.put_along_axis(ok, inds, False, 1)
            ms = np.where(ok, sel, -np.inf)
            ci = np.argsort(-ms, 1, kind="stable")[:, :k]
            cand = np.where(np.isfinite(np.take_along_axis(ms, ci, 1)), ci, -1)
            m = b.decide(l, late.sum(1), order, cand, k)
            out_b = inds.copy()
            for n in np.flatnonzero(m):
                out_b[n, order[n, :m[n]]] = cand[n, :m[n]]
            assert (out_a == out_b).all(), (spec, l)
    assert a.summary() == b.summary()


if __name__ == "__main__":
    for spec in ("rate:0.05", "rate:0.1", "tok:1", "none"):
        for seed in range(2):
            check_fast(spec, seed)
        print(f"ok fast path {spec}: same decisions as Subst.apply (batches of 16)")
    for spec in ("rate:0.05", "rate:0.1", "tok:1", "tok:2", "tok:1:rate:0.05"):
        for seed in range(3):
            ch, late = check(spec, seed)
        print(f"ok {spec}: identical decisions (last: changed {ch:.3f}, late {late:.3f})")
