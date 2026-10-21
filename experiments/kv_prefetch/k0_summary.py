"""K0 summary: retrieval vs prefill vs decode per (dataset, hop, model), path arrival curves, a
streaming-prefill bound, and (K-H2 preview) cross-request segment reuse.

Streaming bound (single GPU, segments prefilled in isolation as they arrive):
  prefill(n tokens) ~= f0 + c * n, fitted per model on the measured full / question-only prefills.
  The GPU takes all segments that have arrived whenever it is idle (one call: f0 + c * tokens).
  TTFT_base   = R + prefill(full prompt)
  TTFT_stream = max(R, GPU finishes the last segment) + prefill(question-only part)   (no stitching cost)
  TTFT_pool   = R + prefill(question-only part)                                       (every segment's KV ready)

  python experiments/kv_prefetch/k0_summary.py --timeline outputs/kv_prefetch/retrieval_timeline.jsonl \
      --gen outputs/kv_prefetch/gen_*.jsonl --full outputs/kv_prefetch/retrieval_full.jsonl --out outputs/kv_prefetch/k0_summary.json
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np


def fit_prefill(rows):
    x = np.array([r["prompt_len"] for r in rows] + [r["qonly_len"] for r in rows], float)
    y = np.array([r["prefill_ms"] for r in rows] + [r["qonly_prefill_ms"] for r in rows], float)
    c, f0 = np.polyfit(x, y, 1)
    return float(f0), float(c)


def stream(arr, seg, R, f0, c):
    """Finish time of the last segment when the GPU batches whatever has arrived each time it is free."""
    t, i, n = 0.0, 0, len(arr)
    while i < n:
        t = max(t, arr[i])
        j = i
        while j < n and arr[j] <= t:
            j += 1
        t += f0 + c * sum(seg[i:j])
        i = j
    return t


def reuse(rows, key):
    """Fraction of segments already seen in an earlier request (infinite pool, dataset order)."""
    seen, hit, tot = set(), 0, 0
    for r in rows:
        units = set()
        for p in r["paths"]:
            if key == "path":
                units.add(json.dumps(p))
            elif key == "triple":
                units.update(json.dumps(t) for t in p)
            else:                                    # "rel": entity-relation line head, the GraphRAG-Opt M2 unit
                units.add(json.dumps(p[0][:2]))
        hit += sum(u in seen for u in units)
        tot += len(units)
        seen |= units
    return hit / max(tot, 1), tot / max(len(rows), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeline", default="outputs/kv_prefetch/retrieval_timeline.jsonl")
    ap.add_argument("--gen", nargs="*", default=[])
    ap.add_argument("--full", default="outputs/kv_prefetch/retrieval_full.jsonl")
    ap.add_argument("--out", default="outputs/kv_prefetch/k0_summary.json")
    a = ap.parse_args()
    tl = {(r["dataset"], r["idx"], r["hop"]): r for r in map(json.loads, open(a.timeline))}
    out = {"retrieval": [], "timing": [], "reuse": []}
    by = defaultdict(list)
    for r in tl.values():
        by[(r["dataset"], r["hop"])].append(r)
    print("== retrieval (cap 50) and path arrival")
    for (ds, hop), rs in sorted(by.items()):
        ms = np.array([r["retrieval_ms"] for r in rs])
        first = [r["arrivals_ms"][0] / r["retrieval_ms"] for r in rs if r["arrivals_ms"] and r["retrieval_ms"] > 1]
        half = [np.mean(np.array(r["arrivals_ms"]) <= 0.5 * r["retrieval_ms"]) for r in rs
                if r["arrivals_ms"] and r["retrieval_ms"] > 1]
        capped = np.mean([r["n_paths"] >= r["cap"] for r in rs])
        row = dict(dataset=ds, hop=hop, mean_ms=ms.mean(), p50_ms=np.median(ms), p90_ms=np.percentile(ms, 90),
                   paths=np.mean([r["n_paths"] for r in rs]), frac_capped=capped,
                   first_path_at=float(np.median(first)) if first else None,
                   arrived_by_half=float(np.mean(half)) if half else None)
        out["retrieval"].append(row)
        print(f"{ds.split('/')[-1]:11} hop {hop}: mean {row['mean_ms']:7.1f} p50 {row['p50_ms']:7.1f} p90 {row['p90_ms']:7.1f} ms"
              f" | paths {row['paths']:4.1f} capped {capped:.0%} | first path at {row['first_path_at'] or 0:.0%} of R,"
              f" {row['arrived_by_half'] or 0:.0%} of paths by R/2")
    for g in a.gen:
        model = os.path.basename(g)[4:-6]
        rows = [json.loads(l) for l in open(g)]
        f0, c = fit_prefill(rows)
        print(f"\n== {model}: prefill ~= {f0:.1f} ms + {c * 1000:.1f} us/token")
        gb = defaultdict(list)
        for r in rows:
            gb[(r["dataset"], r["hop"])].append(r)
        for (ds, hop), rs in sorted(gb.items()):
            R, PF, DEC, Q, ST, PL, HIT = [], [], [], [], [], [], []
            for r in rs:
                t = tl[(r["dataset"], r["idx"], r["hop"])]
                R.append(t["retrieval_ms"])
                PF.append(r["prefill_ms"])
                DEC.append(max(0.0, r["total_ms"] - r["prefill_ms"]))
                Q.append(r["qonly_prefill_ms"])
                ST.append(stream(t["arrivals_ms"], r["seg_tokens"], t["retrieval_ms"], f0, c))
                PL.append(r["prompt_len"])
                HIT.append(r["hit"])
            R, PF, DEC, Q, ST = map(np.array, (R, PF, DEC, Q, ST))
            base = R + PF
            strm = np.maximum(R, ST) + Q
            pool = R + Q
            e2e = base + DEC
            row = dict(model=model, dataset=ds, hop=hop, prompt_len=float(np.mean(PL)), retrieval_ms=R.mean(),
                       prefill_ms=PF.mean(), decode_ms=DEC.mean(), qonly_prefill_ms=Q.mean(),
                       retr_share=float(R.sum() / e2e.sum()), retr_share_med=float(np.median(R / e2e)),
                       idle_over_prefill=float(np.median(R / PF)), ttft_base=base.mean(), ttft_stream=strm.mean(),
                       ttft_pool=pool.mean(), stream_gain=float(1 - strm.sum() / base.sum()),
                       pool_gain=float(1 - pool.sum() / base.sum()), hit=float(np.mean(HIT)))
            out["timing"].append(row)
            print(f"{ds.split('/')[-1]:11} hop {hop}: len {row['prompt_len']:6.0f} | R {row['retrieval_ms']:7.1f}  prefill "
                  f"{row['prefill_ms']:6.1f}  decode {row['decode_ms']:6.1f} ms | retrieval share {row['retr_share']:.0%}"
                  f" (median {row['retr_share_med']:.0%}) | R/prefill median {row['idle_over_prefill']:.1f}"
                  f" | TTFT base {row['ttft_base']:7.1f} stream {row['ttft_stream']:7.1f} ({row['stream_gain']:+.0%})"
                  f" pool {row['ttft_pool']:7.1f} ({row['pool_gain']:+.0%}) | hit {row['hit']:.2f}")
    if os.path.exists(a.full):
        full = defaultdict(list)
        for r in map(json.loads, open(a.full)):
            full[(r["dataset"], r["hop"])].append(r)
        print("\n== cross-request reuse (all test questions, dataset order, unbounded pool)")
        for (ds, hop), rs in sorted(full.items()):
            rs.sort(key=lambda r: r["idx"])
            row = dict(dataset=ds, hop=hop, n=len(rs))
            for key in ("path", "triple", "rel"):
                row[f"{key}_reuse"], row[f"{key}_per_req"] = reuse(rs, key)
            out["reuse"].append(row)
            print(f"{ds.split('/')[-1]:11} hop {hop} (n={len(rs)}): path {row['path_reuse']:.0%} ({row['path_per_req']:.1f}/req)"
                  f"  triple {row['triple_reuse']:.0%} ({row['triple_per_req']:.1f}/req)"
                  f"  entity-relation {row['rel_reuse']:.0%}")
    json.dump(out, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
