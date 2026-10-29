"""Plan D3.2 (next-steps): Mac internal SSD read microbenchmark for expert offloading.

Reads with F_NOCACHE from a file larger than RAM (page-cache hits would otherwise fake 90 GB/s; see memory
mac-edge-setup), at random record-aligned offsets:
  - single reads: one thread, one record at a time -> latency L (and CPU time per read);
  - concurrent reads: Q threads (queue depth Q = 1..32), each reading random records -> throughput and per-read
    latency under load;
  - contiguous reads: one preadv of n consecutive records vs n separate preads (does merging help?).
Record sizes: the GLM-4.5-Air 4bit expert (as in ~/mlx_offload/GLM-4.5-Air-4bit) and a few others.
What else runs on the Mac is recorded with the results (timing runs must not overlap downloads or exports).

  .venv-mac/bin/python experiments/moe_edge/ssd_microbench.py --file ~/mlx_offload/_bench/big.bin --make_gb 80 \\
      --out outputs/moe_edge/microbench/mac_ssd.json
"""
import argparse
import fcntl
import json
import os
import resource
import subprocess
import threading
import time

import numpy as np

F_NOCACHE = 48
SIZES = {"glm45air_4bit": 9732096, "1MB": 1 << 20, "4MB": 4 << 20, "32MB": 32 << 20}


def make_file(path, gb):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) >= gb * 2 ** 30:
        return
    chunk = np.random.default_rng(0).integers(0, 255, 64 << 20, dtype=np.uint8).tobytes()
    with open(path, "wb") as f:
        for i in range(gb * 16):
            f.write(chunk[i % 7:] + chunk[:i % 7])               # not identical blocks
    os.sync()


def others():
    ps = subprocess.run(["ps", "-Ao", "pcpu,comm", "-r"], capture_output=True, text=True).stdout.splitlines()[1:8]
    return dict(loadavg=os.getloadavg(), top=[x.strip() for x in ps], time=time.strftime("%Y-%m-%d %H:%M:%S"))


def single(fd, size, nrec, n, rng):
    buf = bytearray(size)
    mv = memoryview(buf)
    lat = []
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    for _ in range(n):
        off = int(rng.integers(0, nrec)) * size
        t = time.perf_counter()
        os.preadv(fd, [mv], off)
        lat.append(time.perf_counter() - t)
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    cpu = (r1.ru_utime - r0.ru_utime + r1.ru_stime - r0.ru_stime) / n
    lat = np.array(lat) * 1e3
    return dict(lat_ms_p50=float(np.median(lat)), lat_ms_p90=float(np.percentile(lat, 90)),
                gbps=size * n / (lat.sum() / 1e3) / 1e9, cpu_ms_per_read=1e3 * cpu)


def concurrent(fd, size, nrec, Q, per_thread, seed):
    lats = [[] for _ in range(Q)]

    def worker(q):
        rng = np.random.default_rng(seed + q)
        mv = memoryview(bytearray(size))
        for _ in range(per_thread):
            off = int(rng.integers(0, nrec)) * size
            t = time.perf_counter()
            os.preadv(fd, [mv], off)
            lats[q].append(time.perf_counter() - t)
    th = [threading.Thread(target=worker, args=(q,)) for q in range(Q)]
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    t = time.perf_counter()
    for x in th:
        x.start()
    for x in th:
        x.join()
    wall = time.perf_counter() - t
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    lat = np.concatenate([np.array(x) for x in lats]) * 1e3
    n = Q * per_thread
    return dict(gbps=size * n / wall / 1e9, lat_ms_p50=float(np.median(lat)), lat_ms_p90=float(np.percentile(lat, 90)),
                cpu_ms_per_read=1e3 * (r1.ru_utime - r0.ru_utime + r1.ru_stime - r0.ru_stime) / n)


def contiguous(fd, size, nrec, n, trials, rng):
    big = memoryview(bytearray(size * n))
    one, sep = [], []
    for _ in range(trials):
        off = int(rng.integers(0, nrec - n)) * size
        t = time.perf_counter()
        os.preadv(fd, [big], off)
        one.append(time.perf_counter() - t)
        off = int(rng.integers(0, nrec - n)) * size
        t = time.perf_counter()
        for i in range(n):
            os.preadv(fd, [big[i * size:(i + 1) * size]], off + i * size)
        sep.append(time.perf_counter() - t)
    return dict(one_ms=1e3 * float(np.median(one)), separate_ms=1e3 * float(np.median(sep)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--make_gb", type=int, default=80)
    ap.add_argument("--single_n", type=int, default=200)
    ap.add_argument("--per_thread", type=int, default=40)
    ap.add_argument("--note", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.file = os.path.expanduser(a.file)
    t = time.time()
    make_file(a.file, a.make_gb)
    fsize = os.path.getsize(a.file)
    print(f"file {fsize / 2 ** 30:.0f} GiB ready in {time.time() - t:.0f}s; RAM 64 GB", flush=True)
    fd = os.open(a.file, os.O_RDONLY)
    fcntl.fcntl(fd, F_NOCACHE, 1)
    rng = np.random.default_rng(1)
    out = dict(file=a.file, file_gib=fsize / 2 ** 30, note=a.note, before=others(), single=[], concurrent=[],
               contiguous=[])
    for name, size in SIZES.items():
        nrec = fsize // size
        s = single(fd, size, nrec, a.single_n, rng)
        out["single"].append(dict(record=name, MB=size / 1e6, **s))
        print(f"{name:14} single: p50 {s['lat_ms_p50']:.3f} ms  p90 {s['lat_ms_p90']:.3f}  {s['gbps']:.2f} GB/s  "
              f"CPU {s['cpu_ms_per_read']:.3f} ms/read", flush=True)
        for Q in (1, 2, 4, 8, 16, 32):
            c = concurrent(fd, size, nrec, Q, a.per_thread if size <= 10 << 20 else max(8, a.per_thread // 4), Q)
            out["concurrent"].append(dict(record=name, MB=size / 1e6, Q=Q, **c))
            print(f"{name:14} Q={Q:2}: {c['gbps']:6.2f} GB/s  latency p50 {c['lat_ms_p50']:7.3f} p90 "
                  f"{c['lat_ms_p90']:7.3f} ms  CPU {c['cpu_ms_per_read']:.3f} ms/read", flush=True)
        for n in (2, 4, 8):
            m = contiguous(fd, size, nrec, n, 10, rng)
            out["contiguous"].append(dict(record=name, n=n, **m))
            print(f"{name:14} n={n}: one preadv {m['one_ms']:.2f} ms vs {n} preads {m['separate_ms']:.2f} ms", flush=True)
    out["after"] = others()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("written", a.out, flush=True)


if __name__ == "__main__":
    main()
