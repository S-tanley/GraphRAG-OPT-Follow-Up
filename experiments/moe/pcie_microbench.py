"""Plan D3.2 (next-steps): host -> GPU copy microbenchmark for expert offloading (pinned host memory, one GPU).

For each expert size (bytes of one expert's weights) and each number of concurrent streams Q:
  - throughput: Q streams each copy `reps` experts back to back (distinct source / destination slots), wall time
    from the first issue to the last completion -> GB/s;
  - latency: every copy's own issue -> completion time (CUDA events on its stream), median / p90, while Q copies are
    in flight; Q = 1 gives the single-copy latency L;
  - one contiguous copy of n experts vs n separate copies on one stream (does merging reads help?).
The system load (load average, other GPU processes) is recorded with the results: the machine is shared.

  CUDA_VISIBLE_DEVICES=3 taskset -c 0-15,64-79 .venv/bin/python experiments/moe/pcie_microbench.py \\
      --out outputs/moe/microbench/a100_pcie.json
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
import torch

SIZES = {"mixtral8x7b_bf16": 3 * 4096 * 14336 * 2, "glm45air_bf16": 3 * 4096 * 1408 * 2,
         "qwen3_30b_bf16": 3 * 2048 * 768 * 2}


def sysinfo():
    load = os.getloadavg()
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip().splitlines()
    gpus = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip().splitlines()
    return dict(loadavg=load, gpu_apps=apps, gpus=gpus, time=time.strftime("%Y-%m-%d %H:%M:%S"))


def run(size, Q, reps, dev):
    n = Q * reps
    elems = size // 2
    host = torch.empty((min(n, 64), elems), dtype=torch.bfloat16, pin_memory=True)   # sources cycle over 64 slots
    dst = torch.empty((min(n, 64), elems), dtype=torch.bfloat16, device=dev)
    streams = [torch.cuda.Stream(dev) for _ in range(Q)]
    torch.cuda.synchronize(dev)
    ev = []
    t0 = time.perf_counter()
    for r in range(reps):
        for q, s in enumerate(streams):
            i = (r * Q + q) % host.shape[0]
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(s):
                a.record(s)
                dst[i].copy_(host[i], non_blocking=True)
                b.record(s)
            ev.append((a, b))
    torch.cuda.synchronize(dev)
    wall = time.perf_counter() - t0
    lat = np.array([a.elapsed_time(b) for a, b in ev])          # ms, on-stream (issue -> done)
    return dict(gbps=n * size / wall / 1e9, wall_s=wall, lat_ms_p50=float(np.median(lat)),
                lat_ms_p90=float(np.percentile(lat, 90)), lat_ms_min=float(lat.min()))


def merged(size, n, trials, dev):
    """One contiguous copy of n experts vs n separate copies, one stream (ms)."""
    elems = size // 2
    host = torch.empty((n, elems), dtype=torch.bfloat16, pin_memory=True)
    dst = torch.empty((n, elems), dtype=torch.bfloat16, device=dev)
    one, sep = [], []
    for _ in range(trials):
        torch.cuda.synchronize(dev)
        t = time.perf_counter()
        dst.copy_(host, non_blocking=True)
        torch.cuda.synchronize(dev)
        one.append(time.perf_counter() - t)
        t = time.perf_counter()
        for i in range(n):
            dst[i].copy_(host[i], non_blocking=True)
        torch.cuda.synchronize(dev)
        sep.append(time.perf_counter() - t)
    return dict(contiguous_ms=1e3 * float(np.median(one)), separate_ms=1e3 * float(np.median(sep)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--note", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    out = dict(device=torch.cuda.get_device_name(dev), note=a.note, before=sysinfo(), results=[], merged=[])
    print(out["device"], out["before"], flush=True)
    for name, size in SIZES.items():
        reps = max(4, min(64, int(2e9 // size)))                   # about 2 GB per trial and stream
        for Q in a.streams:
            trials = [run(size, Q, reps, dev) for _ in range(a.trials)]
            row = dict(expert=name, MB=size / 1e6, Q=Q, reps=reps,
                       gbps=float(np.median([t["gbps"] for t in trials])),
                       lat_ms_p50=float(np.median([t["lat_ms_p50"] for t in trials])),
                       lat_ms_p90=float(np.median([t["lat_ms_p90"] for t in trials])),
                       lat_ms_min=float(np.min([t["lat_ms_min"] for t in trials])),
                       gbps_trials=[round(t["gbps"], 2) for t in trials])
            out["results"].append(row)
            print(f"{name:18} {size / 1e6:7.1f} MB  Q={Q}  {row['gbps']:6.2f} GB/s  latency p50 {row['lat_ms_p50']:7.3f} "
                  f"p90 {row['lat_ms_p90']:7.3f} ms  (trials {row['gbps_trials']})", flush=True)
        for n in (2, 4, 8):
            m = merged(size, n, a.trials, dev)
            out["merged"].append(dict(expert=name, n=n, **m))
            print(f"{name:18} n={n}  one copy {m['contiguous_ms']:7.3f} ms  vs {n} copies {m['separate_ms']:7.3f} ms",
                  flush=True)
    out["after"] = sysinfo()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("after", out["after"], flush=True)


if __name__ == "__main__":
    main()
