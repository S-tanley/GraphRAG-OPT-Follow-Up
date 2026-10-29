"""R0 summary: retrieval time per method (traversal_profile.py) against the prefill of the same request.

Prefill of a request = f0 + c * prompt tokens, (f0, c) fitted per model on the K0 measurements
(k0_summary.fit_prefill). Prompt tokens = question-only prompt (K0, per question and model) + one line per
retrieved path; K0's prompt_len is exactly this sum. Path lines are tokenized with the Qwen3 tokenizer for
every model (Llama-3.1-70B's question-only length is its own; its path lines differ by about 1%).

Settings: retrieval measured on the matching CPU (server = cluster Xeon, profile_server.jsonl; Mac = M5 Max,
profile_mac.jsonl).
  server-8B   Qwen3-8B, vLLM, H200              gen_Qwen3-8B.jsonl
  server-70B  Llama-3.1-70B, vLLM, 2x H200 TP=2  gen_Llama-3.1-70B-Instruct.jsonl
  mac-8B      Qwen3-8B-4bit, MLX                 gen_mac_kg_Qwen3-8B-4bit.jsonl
  mac-30B     Qwen3-30B-A3B-4bit, MLX            gen_mac_kg_Qwen3-30B-A3B-4bit.jsonl

  python experiments/kv_prefetch/r0_summary.py
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from k0_summary import fit_prefill  # noqa: E402

K0 = os.path.join(ROOT, "outputs", "kv_prefetch")
R0 = os.path.join(K0, "v2", "R0")
SETTINGS = {
    "server-8B": ("server", "gen_Qwen3-8B.jsonl"),
    "server-70B": ("server", "gen_Llama-3.1-70B-Instruct.jsonl"),
    "mac-8B": ("mac", "gen_mac_kg_Qwen3-8B-4bit.jsonl"),
    "mac-30B": ("mac", "gen_mac_kg_Qwen3-30B-A3B-4bit.jsonl"),
}
# method -> mode whose paths make the prompt (jbidfs paths are not stored; its path set is within 1-2 paths
# of the BFS one, so the BFS prompt is used)
METHODS = {
    "gold": ["gold_py", "jbidfs_py", "gold_cpp", "gold_cpp_prune"],
    "open": ["open_cap_py", "open_cap", "open_top", "open_beam"],
    "plan": ["rog_plan_cap30", "rog_plan_cap50"],
}
PROMPT_FROM = {"gold_py": "gold_py", "jbidfs_py": "gold_py", "gold_cpp": "gold_py", "gold_cpp_prune": "gold_py",
               "open_cap_py": "open_cap", "open_cap": "open_cap", "open_top": "open_top", "open_beam": "open_beam",
               "rog_plan_cap30": "rog_plan_cap30", "rog_plan_cap50": "rog_plan_cap50"}


def line_text(p):
    return p[0][0] + "".join(f" -> {r} -> {t}" for _h, r, t in p) + "\n"


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else None


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("mlx-community/Qwen3-8B-4bit")
    prof = {h: [json.loads(l) for l in open(os.path.join(R0, f"profile_{h}.jsonl"))] for h in ("server", "mac")}

    # path-line tokens per (host, dataset, idx, hop, mode); identical across hosts, computed once
    cache = {}

    def ntok(paths):
        s = 0
        for p in paths:
            k = line_text(p)
            if k not in cache:
                cache[k] = len(tok.encode(k, add_special_tokens=False))
            s += cache[k]
        return s

    path_tok = {}
    for h, rows in prof.items():
        for r in rows:
            if "paths" in r:
                path_tok[(h, r["dataset"], r["idx"], r["hop"], r["mode"])] = ntok(r["paths"])

    out = {"fits": {}, "rows": [], "work": []}
    for setting, (host, genf) in SETTINGS.items():
        gen = [json.loads(l) for l in open(os.path.join(K0, genf))]
        f0, c = fit_prefill(gen)
        qonly = {(g["dataset"], g["idx"]): g["qonly_len"] for g in gen}
        # sanity: fitted vs measured prefill on the gold prompts K0 measured
        meas = {(g["dataset"], g["idx"], g["hop"]): g["prefill_ms"] for g in gen}
        err = []
        for r in prof[host]:
            key = (r["dataset"], r["idx"], r["hop"])
            if r["mode"] == "gold_py" and key in meas and (r["dataset"], r["idx"]) in qonly:
                L = qonly[(r["dataset"], r["idx"])] + path_tok[(host, *key, "gold_py")]
                err.append((f0 + c * L) / meas[key] - 1)
        out["fits"][setting] = dict(f0_ms=f0, c_us=c * 1000, n_check=len(err),
                                    fit_err_median=pct(np.abs(err), 50), fit_err_p90=pct(np.abs(err), 90))
        by = defaultdict(list)
        for r in prof[host]:
            by[(r["dataset"], r["hop"], r["mode"])].append(r)
        for (ds, hop, mode), rs in sorted(by.items()):
            fam = next((f for f, ms in METHODS.items() if mode in ms), None)
            if fam is None:
                continue
            R, P = [], []
            for r in rs:
                q = qonly.get((ds, r["idx"]))
                pk = (host, ds, r["idx"], hop, PROMPT_FROM[mode])
                if q is None or pk not in path_tok:
                    continue
                R.append(r["ms"])
                P.append(f0 + c * (q + path_tok[pk]))
            R, P = np.array(R), np.array(P)
            out["rows"].append(dict(
                setting=setting, dataset=ds.split("-")[-1], hop=hop, family=fam, mode=mode, n=len(R),
                R_mean=float(R.mean()), R_p50=pct(R, 50), R_p90=pct(R, 90), R_p99=pct(R, 99), R_max=float(R.max()),
                P_mean=float(P.mean()), P_p50=pct(P, 50),
                frac_R_ge_P=float((R >= P).mean()), frac_R_ge_halfP=float((R >= 0.5 * P).mean()),
                ratio_p50=pct(R / P, 50), ratio_p90=pct(R / P, 90),
                # upper bound of the TTFT cut from overlapping prefill with retrieval: sum min(R, P) / sum (R + P)
                overlap_bound=float(np.minimum(R, P).sum() / (R + P).sum()),
                paths=float(np.mean([r["n_paths"] for r in rs])), cover=float(np.mean([r["cover"] for r in rs]))))

    # R0.1 work / streaming table (per host)
    for host, rows in prof.items():
        by = defaultdict(list)
        for r in rows:
            by[(r["dataset"], r["hop"], r["mode"])].append(r)
        for (ds, hop, mode), rs in sorted(by.items()):
            d = dict(host=host, dataset=ds.split("-")[-1], hop=hop, mode=mode, n=len(rs),
                     ms_mean=float(np.mean([r["ms"] for r in rs])),
                     paths_per_s=float(np.sum([r["n_paths"] for r in rs]) / (np.sum([r["ms"] for r in rs]) / 1000)),
                     match=[r.get("match") for r in rs].count(1), checked=sum(r.get("match") is not None for r in rs))
            for k in ("t_first", "t10", "t50"):
                v = [r[k] for r in rs if r.get(k) is not None]
                if v:
                    d[k + "_p50"], d[k + "_mean"], d[k + "_n"] = pct(v, 50), float(np.mean(v)), len(v)
            for k in ("states", "edges", "nodes", "calls", "all_paths", "ms_core", "ms_first",
                      "prune_ms", "join_ms", "bidi_ms", "combine_ms", "indep_ms", "nonsimple", "truncated",
                      "saved_recall", "adj_ms", "build_ms", "csr_ms", "escore_ms"):
                v = [r[k] for r in rs if r.get(k) is not None]
                if v:
                    d[k + "_mean"], d[k + "_p50"], d[k + "_max"] = float(np.mean(v)), pct(v, 50), float(np.max(v))
            out["work"].append(d)
    json.dump(out, open(os.path.join(R0, "r0_summary.json"), "w"), indent=1)

    # printed tables
    print("prefill fits:", json.dumps(out["fits"], indent=1))
    hdr = f"{'setting':10} {'ds':6} {'h'} {'mode':15} {'R mean':>8} {'p50':>7} {'p90':>7} {'p99':>8} {'P mean':>7} " \
          f"{'R>=P':>5} {'R>=P/2':>6} {'R/P p50':>7} {'bound':>5} {'paths':>5} {'cover':>5}"
    print(hdr)
    for r in out["rows"]:
        print(f"{r['setting']:10} {r['dataset']:6} {r['hop']} {r['mode']:15} {r['R_mean']:8.2f} {r['R_p50']:7.2f} "
              f"{r['R_p90']:7.2f} {r['R_p99']:8.1f} {r['P_mean']:7.1f} {r['frac_R_ge_P']:5.2f} "
              f"{r['frac_R_ge_halfP']:6.2f} {r['ratio_p50']:7.3f} {r['overlap_bound']:5.2f} {r['paths']:5.1f} "
              f"{r['cover']:5.2f}")


if __name__ == "__main__":
    main()
