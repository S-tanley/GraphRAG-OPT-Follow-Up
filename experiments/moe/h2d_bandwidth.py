"""MoE plan Phase 0.5 (Q0): host -> GPU copy bandwidth and per-expert transfer time.

Expert sizes come from the downloaded configs when available (3 matrices hidden x intermediate),
otherwise from the values below. Measures pinned and pageable copies, single stream and
concurrent streams, for chunk sizes equal to one expert of each model.

  python experiments/moe/h2d_bandwidth.py --out outputs/moe/h2d.json
"""

import argparse
import json
import statistics as st
import time

import torch

# bytes of ONE routed expert (gate+up+down) per model, bf16 unless noted; hidden x moe_intermediate
EXPERTS = {
    "Mixtral-8x22B (bf16)": 3 * 6144 * 16384 * 2,
    "Qwen3-235B-A22B (bf16)": 3 * 4096 * 1536 * 2,
    "Qwen3-235B-A22B (fp8)": 3 * 4096 * 1536 * 1,
    "DeepSeek-V2.5 (bf16)": 3 * 5120 * 1536 * 2,
    "GLM-4.5-Air (bf16)": 3 * 4096 * 1408 * 2,
}


def bench(nbytes, pinned, streams=1, reps=10):
    src = [torch.empty(nbytes, dtype=torch.uint8, pin_memory=pinned) for _ in range(streams)]
    dst = [torch.empty(nbytes, dtype=torch.uint8, device="cuda") for _ in range(streams)]
    ss = [torch.cuda.Stream() for _ in range(streams)]
    for i in range(streams):                       # warm-up
        with torch.cuda.stream(ss[i]):
            dst[i].copy_(src[i], non_blocking=pinned)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for i in range(streams):
            with torch.cuda.stream(ss[i]):
                dst[i].copy_(src[i], non_blocking=pinned)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    t = st.median(ts)
    return dict(bytes=nbytes, pinned=pinned, streams=streams, ms=1000 * t, GBps=streams * nbytes / t / 1e9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    print(torch.cuda.get_device_name(0))
    res = []
    for name, nb in EXPERTS.items():
        for pinned in (True, False):
            for streams in (1, 4):
                r = bench(nb, pinned, streams)
                r["model"] = name
                res.append(r)
                print(f"{name:26s} expert {nb / 1e6:8.1f}MB pinned={pinned!s:5s} streams={streams}: "
                      f"{r['ms']:8.2f} ms  {r['GBps']:6.1f} GB/s")
    big = bench(4 << 30, True, 1, reps=5)
    print(f"4GB pinned single copy: {big['GBps']:.1f} GB/s")
    res.append(dict(big, model="4GB block"))
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
