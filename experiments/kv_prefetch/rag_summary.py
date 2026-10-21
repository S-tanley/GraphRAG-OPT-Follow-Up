"""K0-v2 summary: retrieval vs generation on text multi-hop RAG, and the generation-side bound of KV prefetch.

Retrieval pipelines (per-question medians from rag_bench.py retrieve / wiki_index.py retrieve):
  dense-small      bge-large query encoding + exact search on the per-dataset corpus (6-12k passages, GPU)
  dense+rerank     the above + bge-reranker-v2-m3 over the top-50
  dense-1M-cpu     exact search over 1.04M passages on 16 CPU threads (faiss flat)
  dense-1M-ivf     IVF4096 nprobe 64 over 1.04M passages on 16 CPU threads
  wiki21M-ivf<np>  DPR encoding + IVF4096 over 21M Wikipedia passages, 32 CPU threads, nprobe 32 / 128 / 256
Generation (rag_bench.py gen, batch 1, vLLM): prefill of the full prompt, decode of the short answer, and
  rest = prefill of what remains after retrieval if every passage's KV is ready (q-last: question + tail; q-first:
  tail only, prefix + question prefilled during retrieval); early = prefill of prefix (+ question), which must fit
  in the retrieval window.

  python experiments/kv_prefetch/rag_summary.py --root outputs/kv_prefetch/rag --out outputs/kv_prefetch/rag_summary.json
"""
import argparse
import glob
import json
import os

import numpy as np


def jl(p):
    return [json.loads(l) for l in open(p)]


def med(x):
    return float(np.median(x)) if len(x) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/kv_prefetch/rag")
    ap.add_argument("--out", default="outputs/kv_prefetch/rag_summary.json")
    a = ap.parse_args()
    out = {}
    for ds in ("hotpotqa", "musique", "2wiki"):
        R = jl(f"{a.root}/{ds}/retrieval.jsonl")
        ms = {k: med([r["ms"][k] for r in R]) for k in R[0]["ms"]}
        retr = {"dense-small": ms["encode"] + ms["small_gpu"], "dense+rerank": ms["encode"] + ms["small_gpu"] + ms["rerank50"],
                "dense-1M-cpu": ms["encode"] + ms["large_cpu_flat"], "dense-1M-ivf": ms["encode"] + ms["large_cpu_ivf"]}
        wp = f"{a.root}/{ds}/retrieval_wiki.jsonl"
        if os.path.exists(wp):
            W = jl(wp)
            wm = {k: med([r["ms"][k] for r in W]) for k in W[0]["ms"]}
            for npb in (32, 128, 256):
                retr[f"wiki21M-ivf{npb}"] = wm["encode"] + wm[f"ivf_np{npb}_t32"]
        gens = {}
        for f in sorted(glob.glob(f"{a.root}/{ds}/gen/*_k*_*.jsonl")):
            G = jl(f)
            name = os.path.basename(f)[:-6]
            gens[name] = dict(n=len(G), len=float(np.mean([g["len"] for g in G])), prefill=med([g["prefill_ms"] for g in G]),
                              decode=med([g["total_ms"] - g["prefill_ms"] for g in G]), rest=med([g["rest_ms"] for g in G]),
                              early=med([g["early_ms"] for g in G]), em=float(np.mean([g["em"] for g in G])),
                              f1=float(np.mean([g["f1"] for g in G])))
        out[ds] = dict(retrieval_ms=retr, gen=gens)
        print(f"\n== {ds}: retrieval (ms) " + "  ".join(f"{k} {v:.1f}" for k, v in retr.items()))
        for name, g in gens.items():
            gen_ms = g["prefill"] + g["decode"]
            saved = g["prefill"] - g["rest"]
            print(f"  {name:42} len {g['len']:5.0f} | prefill {g['prefill']:6.1f} decode {g['decode']:5.1f} | rest {g['rest']:5.1f} "
                  f"early {g['early']:5.1f} | gen-side TTFT -{saved / g['prefill']:.0%}, gen total -{saved / gen_ms:.0%} "
                  f"| EM {g['em']:.3f} F1 {g['f1']:.3f}")
        # retrieval share of end-to-end for the k=5 q-last prompt of each model
        for name, g in gens.items():
            if "_k5_qlast" not in name:
                continue
            gen_ms = g["prefill"] + g["decode"]
            ret_key = name.split("_")[-3]
            pipes = ["dense-small", "dense+rerank", "dense-1M-ivf", "dense-1M-cpu"] if ret_key != "wiki" else \
                [k for k in retr if k.startswith("wiki")]
            print(f"  retrieval share with {name}: " + "  ".join(f"{p} {retr[p] / (retr[p] + gen_ms):.0%}" for p in pipes))
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
