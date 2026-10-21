"""Expert-queue plan Q1 calibration on one H200: the time model of an offloaded GLM-4.5-Air decode layer.

  experts   routed-expert compute for (distinct experts m) x (tokens per expert t), bf16, as a looped per-expert
            matmul (what HF does) and as torch._grouped_mm when available
  attn      non-expert part of one decode layer for batch B and context length S: RMSNorm, QKV projection,
            GQA attention over the KV cache, O projection, router
  h2d       host (pinned) -> GPU copies of j experts (34.6 MB each in bf16): one copy per expert vs one
            coalesced copy; effective PCIe bandwidth
  overhead  empty kernel launch, stream synchronize, a device->host scalar read (routing result to the CPU)

  CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/moe/queue_microbench.py --out outputs/moe/queue_microbench.json
"""
import argparse
import json
import time

import torch
import torch.nn.functional as F

H, I, NH, NKV, HD, E = 4096, 1408, 96, 8, 128, 128
dev = "cuda"


def timeit(fn, reps=50, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(reps):
        st.record()
        fn()
        en.record()
        torch.cuda.synchronize()
        ts.append(st.elapsed_time(en))
    ts.sort()
    return ts[len(ts) // 2] * 1e3                              # us


def bench_experts():
    out = []
    W1 = torch.randn(E, H, 2 * I, device=dev, dtype=torch.bfloat16) * 0.02
    W2 = torch.randn(E, I, H, device=dev, dtype=torch.bfloat16) * 0.02
    has_gmm = hasattr(torch, "_grouped_mm")
    for m in (1, 2, 4, 8, 16, 32, 64, 128):
        for t in (1, 2, 4, 8, 16, 32, 64):
            x = torch.randn(m * t, H, device=dev, dtype=torch.bfloat16)

            def loop():
                for e in range(m):
                    h = x[e * t:(e + 1) * t] @ W1[e]
                    g, u = h.chunk(2, -1)
                    _ = (F.silu(g) * u) @ W2[e]
            row = dict(m=m, t=t, loop_us=timeit(loop, reps=20))
            if has_gmm:
                offs = torch.arange(1, m + 1, device=dev, dtype=torch.int32) * t

                def gmm():
                    h = torch._grouped_mm(x, W1[:m], offs=offs)
                    g, u = h.chunk(2, -1)
                    torch._grouped_mm((F.silu(g) * u).contiguous(), W2[:m], offs=offs)
                try:
                    row["gmm_us"] = timeit(gmm, reps=20)
                except Exception as e:                       # noqa: BLE001
                    row["gmm_err"] = str(e)[:120]
                    has_gmm = False
            out.append(row)
            print("experts", row, flush=True)
    return out


def bench_attn():
    out = []
    Wqkv = torch.randn(H, (NH + 2 * NKV) * HD, device=dev, dtype=torch.bfloat16) * 0.02
    Wo = torch.randn(NH * HD, H, device=dev, dtype=torch.bfloat16) * 0.02
    Wr = torch.randn(H, E, device=dev, dtype=torch.bfloat16) * 0.02
    for S in (512, 2048):
        for B in (1, 4, 8, 16, 32, 64, 128, 256):
            x = torch.randn(B, H, device=dev, dtype=torch.bfloat16)
            K = torch.randn(B, NKV, S, HD, device=dev, dtype=torch.bfloat16)
            V = torch.randn(B, NKV, S, HD, device=dev, dtype=torch.bfloat16)

            def layer():
                h = F.rms_norm(x, (H,))
                qkv = h @ Wqkv
                q = qkv[:, :NH * HD].view(B, NH, 1, HD)
                o = F.scaled_dot_product_attention(q, K, V, enable_gqa=True)
                y = o.reshape(B, NH * HD) @ Wo + x
                torch.topk((F.rms_norm(y, (H,)) @ Wr).float().sigmoid(), 8, -1)
            row = dict(S=S, B=B, us=timeit(layer))
            out.append(row)
            print("attn", row, flush=True)
    return out


def bench_h2d():
    out = []
    eb = 3 * H * I                                             # elements of one expert
    host = torch.empty(16 * eb, dtype=torch.bfloat16).pin_memory()
    gpu = torch.empty(16 * eb, dtype=torch.bfloat16, device=dev)
    s = torch.cuda.Stream()
    for j in (1, 2, 4, 8, 16):
        def per_expert():
            with torch.cuda.stream(s):
                for e in range(j):
                    gpu[e * eb:(e + 1) * eb].copy_(host[e * eb:(e + 1) * eb], non_blocking=True)
            s.synchronize()

        def coalesced():
            with torch.cuda.stream(s):
                gpu[:j * eb].copy_(host[:j * eb], non_blocking=True)
            s.synchronize()
        for name, fn in (("per_expert", per_expert), ("coalesced", coalesced)):
            for _ in range(3):
                fn()
            ts = []
            for _ in range(20):
                t0 = time.perf_counter()
                fn()
                ts.append(time.perf_counter() - t0)
            us = sorted(ts)[len(ts) // 2] * 1e6
            row = dict(j=j, mode=name, us=us, GBps=j * eb * 2 / us / 1e3)
            out.append(row)
            print("h2d", row, flush=True)
    return out


def bench_overhead():
    x = torch.zeros(1, device=dev)
    r = {}
    r["launch_us"] = timeit(lambda: x.add_(1), reps=200)

    def sync():
        x.add_(1)
        torch.cuda.synchronize()
    ts = []
    for _ in range(200):
        t0 = time.perf_counter()
        sync()
        ts.append(time.perf_counter() - t0)
    r["launch_plus_sync_us"] = sorted(ts)[100] * 1e6
    idx = torch.randint(0, E, (64, 8), device=dev)
    ts = []
    for _ in range(200):
        t0 = time.perf_counter()
        idx.add_(0)
        idx.cpu()
        ts.append(time.perf_counter() - t0)
    r["routing_to_cpu_us"] = sorted(ts)[100] * 1e6
    print("overhead", r, flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    res = dict(device=torch.cuda.get_device_name(), torch=torch.__version__, overhead=bench_overhead(),
               h2d=bench_h2d(), attn=bench_attn(), experts=bench_experts())
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
