"""Op-level profile of the first 3 layers of Qwen3-30B-A3B (MLX 4bit) on the Mac, for the profiling slides.

Configurations: phase {prefill, decode} x mode {resident, offload} x batch {1, 8, 16, 32, 64}.
  prefill : B requests x 256 prompt tokens (real KGQA prompts, last 256 tokens of each)
  decode  : one new token per request, KV cache of 256 tokens per request
  resident: every expert in unified memory (standard mlx-lm model)
  offload : routed experts on the internal SSD (offload_engine.py: experts.bin, preadv + F_NOCACHE into
            per-layer cache slots); 50% of each layer's experts cached, warmed with the globally most used
            experts (LFU, steady state); missing experts are read on demand.
Each op is evaluated and synchronised on its own (mx.eval) and timed with perf_counter, so the timeline is
op-resolved; the unsynchronised end-to-end time of the same 3 layers is also recorded (sync overhead).
For each op: wall time, bytes it must read, FLOPs -> achieved memory bandwidth and FLOP/s, and their ratio to
the measured peaks ("GPU utilisation"; no hardware counters without root). Every SSD read (offload) is logged
with start / end / bytes -> SSD read bandwidth over time.

  .venv-mac/bin/python experiments/moe_edge/profile_layers.py --mode resident --out outputs/moe_edge/profile/resident.json
  .venv-mac/bin/python experiments/moe_edge/profile_layers.py --mode offload  --out outputs/moe_edge/profile/offload.json
"""

import argparse
import glob
import json
import os
import sys
import time

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models import switch_layers
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.cache import KVCache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from offload_engine import OffloadEngine  # noqa: E402

REPO = "mlx-community/Qwen3-30B-A3B-4bit"
STORE = os.path.expanduser("~/mlx_offload/Qwen3-30B-A3B-4bit")
TRACES = "outputs/moe_edge/traces/Qwen3-30B-A3B-4bit-cf"
NL, T_PROMPT = 3, 256
QB = 0.5625                       # bytes per parameter of MLX 4bit (group 64: 4 bits + scale/bias per 64)


# ----------------------------------------------------------------------------- peaks
def measure_peaks():
    def t(f, n=20, trials=5):
        """Best of several trials (the GPU clocks up during the first ones)."""
        best = 1e9
        for _ in range(trials):
            f(); mx.synchronize()
            s = time.perf_counter()
            for _ in range(n):
                f()
            mx.synchronize()
            best = min(best, (time.perf_counter() - s) / n)
        return best
    a = mx.random.normal((1 << 28,)); mx.eval(a)
    bw = a.nbytes / t(lambda: mx.eval(mx.sum(a)))
    x = mx.random.normal((4096, 4096)).astype(mx.bfloat16); w = mx.random.normal((4096, 4096)).astype(mx.bfloat16)
    mx.eval(x, w)
    fl = 2 * 4096 ** 3 / t(lambda: mx.eval(x @ w))
    wq = mx.quantize(w, group_size=64, bits=4); mx.eval(wq)
    flq = 2 * 4096 ** 3 / t(lambda: mx.eval(mx.quantized_matmul(x, *wq, transpose=True, group_size=64, bits=4)))
    return dict(mem_bw=bw, flops_bf16=fl, flops_q4=flq)


# ----------------------------------------------------------------------------- inputs
def load_prompts(n):
    out = []
    for f in sorted(glob.glob(os.path.join(TRACES + "_cwq_M0_cap8", "q*.npz"))):
        z = np.load(f)
        ids = z["prompt_ids"]
        if len(ids) >= T_PROMPT and len(z["gen_ids"]):
            out.append((ids[-T_PROMPT:].astype(np.int32), int(z["gen_ids"][0])))
        if len(out) == n:
            break
    return out


def lfu_hot(L, E):
    """Global expert use counts of layers 0..L-1 over the Wtrain traces (prefill + decode)."""
    c = np.zeros((L, E))
    for f in sorted(glob.glob(os.path.join(TRACES + "_train_M0_cap8", "q*.npz")))[:1000]:
        z = np.load(f)
        for l in range(L):
            np.add.at(c[l], z["pre_idx"][l].ravel(), 1)
            np.add.at(c[l], z["dec_idx"][l].ravel(), 1)
    return c


# ----------------------------------------------------------------------------- profiler
class Prof:
    """Records ops in execution order. A GPU op's duration is its amortised throughput time: the op is
    re-run back-to-back R times with a single synchronisation at the end (per-op synchronisation costs
    ~0.1 ms, which would swamp decode ops). I/O waits and the host round-trip that offloading needs
    (routing -> CPU) are real wall times. The timeline lays ops end to end with these durations."""

    def __init__(self):
        self.ev = []
        self._big = mx.random.normal((4096, 4096)).astype(mx.bfloat16)
        mx.eval(self._big)

    def _clock_up(self, ms=20):
        """Keep the GPU busy for a moment so the op is timed at full clock: the GPU down-clocks after idle
        periods (I/O waits), which would otherwise inflate small ops 2x. That real slowdown is reported
        separately as the gap between the measured end-to-end time and the timeline."""
        t = time.perf_counter()
        while (time.perf_counter() - t) * 1e3 < ms:
            mx.eval(self._big @ self._big)

    def start(self):
        self.ev = []
        self.t0 = time.perf_counter()

    def op(self, name, layer, fn, nbytes=0.0, flops=0.0, kind="gpu", flop_peak="q4", reset=None, restore=None):
        s = time.perf_counter()
        out = fn()
        if out is not None:
            mx.eval(out)
        e = time.perf_counter()
        dur = e - s
        if kind == "gpu":
            R = int(min(40, max(3, 3e-3 / max(dur, 1e-6))))
            outs = []
            if reset:
                reset()
            self._clock_up()
            fn(); mx.synchronize()                      # warm
            ts = time.perf_counter()
            for _ in range(R):
                if reset:
                    reset()
                outs.append(fn())
            mx.eval(outs)
            dur = (time.perf_counter() - ts) / R
            if restore:
                restore()
        self.ev.append(dict(name=name, layer=layer, dur=dur, wall_t0=s - self.t0, wall_t1=e - self.t0,
                            bytes=nbytes, flops=flops, kind=kind, flop_peak=flop_peak))
        return out


def layer_forward(P, li, layer, x, cache, mask, cfg, eng, io_log):
    B, L, D = x.shape
    attn = layer.self_attn
    H, KVH = attn.n_heads, attn.n_kv_heads
    hd = attn.q_proj.weight.shape[0] // H                     # QuantizedLinear weight: [out, packed in]
    T = B * L
    ctx = cache.offset + L
    act = 2 * T * D                                           # bf16 activation bytes in / out
    h = P.op("input_norm", li, lambda: layer.input_layernorm(x), 2 * act)
    p_qkv = D * (H + 2 * KVH) * hd
    qkv = P.op("qkv_proj", li, lambda: (attn.q_proj(h), attn.k_proj(h), attn.v_proj(h)),
               p_qkv * QB + act, 2 * T * p_qkv)

    def rope_cache():
        q, k, v = qkv
        q = attn.q_norm(q.reshape(B, L, H, -1)).transpose(0, 2, 1, 3)
        k = attn.k_norm(k.reshape(B, L, KVH, -1)).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, KVH, -1).transpose(0, 2, 1, 3)
        q = attn.rope(q, offset=cache.offset)
        k = attn.rope(k, offset=cache.offset)
        k, v = cache.update_and_fetch(k, v)
        return q, k, v
    off = cache.offset
    q, k, v = P.op("qk_norm_rope_kv", li, rope_cache, 2 * T * (H + 2 * KVH) * hd * 2,
                   reset=lambda: setattr(cache, "offset", off), restore=lambda: setattr(cache, "offset", off + L))
    kv_bytes = 2 * B * KVH * ctx * hd * 2
    att_flops = 2 * 2 * B * H * L * ctx * hd
    o = P.op("attention", li, lambda: scaled_dot_product_attention(q, k, v, cache=cache, scale=attn.scale, mask=mask)
             .transpose(0, 2, 1, 3).reshape(B, L, -1), kv_bytes + 2 * T * H * hd * 2, att_flops, flop_peak="bf16")
    p_o = H * hd * D
    h1 = P.op("o_proj_residual", li, lambda: x + attn.o_proj(o), p_o * QB + 2 * act, 2 * T * p_o)
    h2 = P.op("post_norm", li, lambda: layer.post_attention_layernorm(h1), 2 * act)
    moe = layer.mlp
    E, K = moe.num_experts, moe.top_k

    def router():
        g = mx.softmax(moe.gate(h2), axis=-1, precise=True)
        inds = mx.argpartition(g, kth=-K, axis=-1)[..., -K:]
        sc = mx.take_along_axis(g, inds, axis=-1)
        sc = sc / mx.sum(sc, axis=-1, keepdims=True)
        return inds, sc
    inds, sc = P.op("router", li, router, D * E * 2 + act, 2 * T * D * E, flop_peak="bf16")
    if cfg["mode"] == "offload":                           # the engine must see the routing on the CPU
        hold = {}
        P.op("route_to_host", li, lambda: hold.setdefault("i", np.array(inds)) is None or None, kind="host")
        idx = hold["i"]
    else:
        idx = np.array(inds)
    U = len(np.unique(idx))
    inter = eng.mode_cfg["hidden_dims"] if eng is not None else moe.switch_mlp.gate_proj.weight.shape[1]
    p_exp = 3 * D * inter                                     # gate + up + down of one expert
    e_bytes, e_flops = U * p_exp * QB + 2 * T * K * D * 2, 2 * T * K * p_exp
    if cfg["mode"] == "resident":
        y = P.op("experts", li, lambda: moe.switch_mlp(h2, inds), e_bytes, e_flops)
    else:
        j = li
        uniq = np.unique(idx)
        n_io0 = len(io_log)
        if len(uniq) <= eng.S:                              # pool path (decode)
            def load_wait():
                prot = set(int(e) for e in uniq)
                for e in uniq:
                    if eng.slot_of[j, e] < 0 and (j, int(e)) not in eng.pending:
                        eng._load_async(j, int(e), prot)
                eng._wait(j, uniq)
            P.op("expert_load_wait", li, lambda: load_wait(), kind="io")
            remap = eng.slot_of[j][idx].astype(np.uint32)
            y = P.op("experts", li, lambda: switch_layers.SwitchGLU.__call__(eng._bind(eng.pool[j]), h2, mx.array(remap)),
                     e_bytes, e_flops)
        else:                                               # staging path (prefill)
            res = np.array([eng.slot_of[j, e] >= 0 for e in uniq])
            miss = uniq[~res]

            def load_wait():
                futs = [eng.pool_exec.submit(eng._read_into, j, int(e), [v[int(e)] for v in eng.stage_np]) for e in miss]
                for f in futs:
                    f.result()
            P.op("expert_load_wait", li, lambda: load_wait(), kind="io")
            in_pool = eng.slot_of[j][idx] >= 0
            pool_idx = np.where(in_pool, eng.slot_of[j][idx], 0).astype(np.uint32)

            def compute():
                yp = switch_layers.SwitchGLU.__call__(eng._bind(eng.pool[j]), h2, mx.array(pool_idx))
                if len(miss):
                    ys = switch_layers.SwitchGLU.__call__(eng._bind(eng.stage), h2, mx.array(idx.astype(np.uint32)))
                    return mx.where(mx.array(in_pool)[..., None], yp, ys)
                return yp
            y = P.op("experts", li, compute, e_bytes, e_flops)
        P.ev[-1]["loaded"] = len(io_log) - n_io0
    out = P.op("combine_residual", li, lambda: h1 + (y * sc[..., None]).sum(axis=-2).astype(y.dtype), 2 * T * K * D * 2 + act)
    P.ev[-1]["distinct_experts"] = U
    return out


def run_config(model, eng, phase, B, prompts, mode, P, io_log, warm, reps):
    inner = model.model
    toks = mx.array(np.stack([p[0] for p in prompts[:B]]))
    nxt = mx.array(np.array([[p[1]] for p in prompts[:B]], dtype=np.int32))
    results = []
    for rep in range(reps + 1):                               # rep 0 = warm-up
        if eng is not None:
            warm()
        caches = [KVCache() for _ in range(NL)]
        if phase == "decode":                                 # build the 256-token context (not profiled)
            h = inner.embed_tokens(toks)
            for li in range(NL):
                h = inner.layers[li](h, "causal", caches[li])
            mx.eval(h, [c.keys for c in caches])
            if eng is not None:
                warm()
            x0 = inner.embed_tokens(nxt)
            mask = None
        else:
            x0 = inner.embed_tokens(toks)
            mask = "causal"
        mx.eval(x0)
        io_log.clear()
        P.start()
        x = x0
        for li in range(NL):
            x = layer_forward(P, li, inner.layers[li], x, caches[li], mask, dict(mode=mode), eng, io_log)
        total = sum(e["dur"] for e in P.ev)
        io = [dict(wall_t0=a - P.t0, wall_t1=b - P.t0, bytes=n) for a, b, n in io_log]
        # unsynchronised end-to-end time of the same 3 layers (same inputs, fresh caches)
        if eng is not None:
            warm()
        caches2 = [KVCache() for _ in range(NL)]
        if phase == "decode":
            h = inner.embed_tokens(toks)
            for li in range(NL):
                h = inner.layers[li](h, "causal", caches2[li])
            mx.eval(h, [c.keys for c in caches2])
            if eng is not None:
                warm()
        s = time.perf_counter()
        x = x0
        for li in range(NL):
            x = inner.layers[li](x, mask, caches2[li])
        mx.eval(x)
        nosync = time.perf_counter() - s
        if rep:
            results.append(dict(events=P.ev, io=io, total_s=total, nosync_total_s=nosync))
    results.sort(key=lambda r: r["total_s"])
    med = results[len(results) // 2]
    med["all_totals_s"] = [r["total_s"] for r in results]
    med["all_nosync_s"] = [r["nosync_total_s"] for r in results]
    return med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["resident", "offload"], required=True)
    ap.add_argument("--batches", default="1,8,16,32,64")
    ap.add_argument("--phases", default="decode,prefill")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--cache_frac", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    peaks = measure_peaks()
    print("peaks:", {k: f"{v / 1e9:.0f} G" for k, v in peaks.items()}, flush=True)
    prompts = load_prompts(64)
    io_log = []
    eng, warm = None, None
    if args.mode == "resident":
        model, _ = load(REPO)
    else:
        model, _ = load(REPO, lazy=True)
        meta = json.load(open(os.path.join(STORE, "meta.json")))
        S = int(args.cache_frac * meta["E"])
        eng = OffloadEngine.attach(model, STORE, S, policy="sw-moeinf")
        orig = eng._read_into

        def logged(j, e, views):
            s = time.perf_counter()
            n = orig(j, e, views)
            io_log.append((s, time.perf_counter(), n))
            return n
        eng._read_into = logged
        hot = lfu_hot(NL, meta["E"])
        eng.freq[:NL] = hot

        def warm():
            """Steady state: the S globally most used experts of each profiled layer are resident."""
            for j in range(NL):
                eng.slot_of[j][:] = -1
                eng.exp_in[j][:] = -1
                for e in np.argsort(-hot[j])[:S]:
                    eng._load_async(j, int(e), protect=set())
                eng._wait(j, np.arange(meta["E"]))
            eng.pending.clear()
            io_log.clear()
    out = dict(model=REPO, mode=args.mode, layers=NL, prompt_tokens=T_PROMPT, peaks=peaks,
               cache_frac=args.cache_frac if eng else 1.0, expert_bytes=3 * 2048 * 768 * QB,
               ssd_record_bytes=eng.rec if eng else None, configs=[])
    for phase in args.phases.split(","):
        for B in [int(b) for b in args.batches.split(",")]:
            r = run_config(model, eng, phase, B, prompts, args.mode, Prof(), io_log, warm, args.reps)
            r.update(phase=phase, batch=B)
            out["configs"].append(r)
            gpu = sum(e["dur"] for e in r["events"] if e["kind"] == "gpu")
            io_t = sum(e["dur"] for e in r["events"] if e["kind"] == "io")
            print(f"{args.mode:8s} {phase:7s} B={B:2d}: timeline {r['total_s'] * 1e3:8.2f} ms (measured end-to-end {r['nosync_total_s'] * 1e3:8.2f} ms)"
                  f"  gpu ops {gpu * 1e3:7.2f} ms  io wait {io_t * 1e3:7.2f} ms  SSD reads {len(r['io'])}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"))


if __name__ == "__main__":
    main()
