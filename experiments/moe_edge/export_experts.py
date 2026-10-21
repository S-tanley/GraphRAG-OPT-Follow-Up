"""Re-lay an MLX (quantized) MoE checkpoint's routed experts as one contiguous record per (layer, expert)
in a single file on the internal SSD (the "host memory" tier of plan §1.1), so the offload engine can fetch
an expert with one preadv straight into its cache slot.

Record = the expert's slices of switch_mlp.{gate,up,down}_proj.{weight,scales,biases} (whatever exists),
concatenated, each tensor 64-byte aligned, the record padded to 16 KiB. meta.json gives layer ids,
tensor names / per-expert shapes / dtypes / offsets and the record size.

  .venv-mac/bin/python experiments/moe_edge/export_experts.py --model mlx-community/GLM-4.5-Air-4bit \
      --out ~/mlx_offload/GLM-4.5-Air-4bit
"""

import argparse
import glob
import json
import os
import time

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download

PROJ = ("gate_proj", "up_proj", "down_proj")
PARTS = ("weight", "scales", "biases")
NP = {mx.uint32: np.uint32, mx.bfloat16: np.uint16, mx.float16: np.float16, mx.float32: np.float32,
      mx.uint8: np.uint8}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    src = snapshot_download(args.model, allow_patterns=["*.json", "*.safetensors"])
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    keys = [k for k in idx if ".switch_mlp." in k]
    prefix = sorted({k.split(".switch_mlp.")[0] for k in keys}, key=lambda s: int(s.split(".layers.")[1].split(".")[0]))
    layer_ids = [int(p.split(".layers.")[1].split(".")[0]) for p in prefix]
    # per-expert tensor layout from the first layer
    files = {}
    tensors = []
    off = 0
    first = {}
    for pr in PROJ:
        for pa in PARTS:
            k = f"{prefix[0]}.switch_mlp.{pr}.{pa}"
            if k not in idx:
                continue
            arr = mx.load(os.path.join(src, idx[k]))[k] if idx[k] not in files else files[idx[k]][k]
            E = arr.shape[0]
            shape, dt = list(arr.shape[1:]), arr.dtype
            nbytes = int(np.prod(shape)) * np.dtype(NP[dt]).itemsize
            off = (off + 63) // 64 * 64
            tensors.append(dict(name=f"{pr}.{pa}", shape=shape, dtype=str(dt).split(".")[-1], offset=off, nbytes=nbytes))
            off += nbytes
    rec = (off + 16383) // 16384 * 16384
    meta = dict(model=args.model, src=src, layer_ids=layer_ids, prefix=prefix, E=E, record_bytes=rec, tensors=tensors)
    print(f"{args.model}: {len(layer_ids)} MoE layers x {E} experts, record {rec / 1e6:.2f} MB "
          f"(payload {off / 1e6:.2f} MB), file {len(layer_ids) * E * rec / 1e9:.1f} GB", flush=True)
    path = os.path.join(out, "experts.bin")
    t0 = time.time()
    with open(path, "wb") as f:
        buf = np.zeros(rec, np.uint8)
        cache = {}
        for li, pfx in enumerate(prefix):
            arrs = {}
            for t in tensors:
                k = f"{pfx}.switch_mlp.{t['name']}"
                fn = idx[k]
                if fn not in cache:                       # keep only the current shard's lazy handles
                    cache = {fn: mx.load(os.path.join(src, fn))}
                a = cache[fn][k]
                arrs[t["name"]] = np.array(a.view(mx.uint16) if a.dtype == mx.bfloat16 else a)
            for e in range(E):
                buf[:] = 0
                for t in tensors:
                    b = arrs[t["name"]][e].tobytes()
                    buf[t["offset"]:t["offset"] + len(b)] = np.frombuffer(b, np.uint8)
                f.write(buf.tobytes())
            if li % 5 == 0 or li == len(prefix) - 1:
                print(f"  layer {layer_ids[li]} done ({time.time() - t0:.0f}s)", flush=True)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print(f"wrote {path} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
