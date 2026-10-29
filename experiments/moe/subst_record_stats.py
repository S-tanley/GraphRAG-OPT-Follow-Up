"""Multi-user plan 2.1.2 input: what the simulator's use-time substitution changes, from its records (1.3.3).

For every record (request, decode token, layer, expert -> substitute) and the trace's router scores at that point:
  the rank of the replaced expert inside the token's top-k (0 = highest score) and the rank of the substitute
  among all experts. Saved as empirical distributions, plus the per-layer change rate of the run, for the random
  mode of subst_replay.py (changes drawn to match what the simulator does, on tasks without records).

  python experiments/moe/subst_record_stats.py --records outputs/moe/subst/records_glm_cot/<run>.npz \
      --trace outputs/moe/traces_v4/GLM-4.5-Air_cwq_M0_cot_chat_cap8 --runs outputs/moe/subst/records_glm_cot/runs.json \
      --out outputs/moe/subst/stats_<run>.npz
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--runs", default=None, help="tsim_mu output with subst_layer_rate per run")
    ap.add_argument("--out_dir", default="outputs/moe/subst")
    a = ap.parse_args()
    runs = {}
    if a.runs:
        for r in json.load(open(a.runs)):
            if r.get("subst_log"):
                runs[os.path.basename(r["subst_log"])] = r
    cache = {}
    for f in a.records:
        z = np.load(f)
        rec, stems = z["records"], z["stems"]
        E = None
        rep_rank, sub_rank = [], []
        for rid in np.unique(rec[:, 0]):
            stem = str(stems[rid])
            if stem not in cache:
                t = np.load(os.path.join(a.trace, stem))
                cache[stem] = t["dec_score"].astype(np.float32)        # [L, D, E]
            S = cache[stem]
            E = S.shape[-1]
            m = rec[rec[:, 0] == rid]
            sc = S[m[:, 2], m[:, 1]]                                     # [n, E] scores at (layer, token)
            order = np.argsort(-sc, -1, kind="stable")
            rank = np.argsort(order, -1)                                  # rank of every expert
            rep_rank.append(rank[np.arange(len(m)), m[:, 3]])
            sub_rank.append(rank[np.arange(len(m)), m[:, 4]])
        rep_rank = np.concatenate(rep_rank) if rep_rank else np.zeros(0, int)
        sub_rank = np.concatenate(sub_rank) if sub_rank else np.zeros(0, int)
        k = int(rep_rank.max()) + 1 if len(rep_rank) else 8
        rep_hist = np.bincount(rep_rank, minlength=k)
        sub_hist = np.bincount(sub_rank, minlength=E or 128)
        name = os.path.basename(f)
        run = runs.get(name, {})
        layer_rate = np.array(run.get("subst_layer_rate", []), np.float64)
        out = os.path.join(a.out_dir, "stats_" + name)
        np.savez(out, replaced_rank_hist=rep_hist, substitute_rank_hist=sub_hist, layer_rate=layer_rate,
                 change_rate=run.get("subst_change_rate", np.nan), n=len(rep_rank))
        q = np.cumsum(sub_hist) / max(sub_hist.sum(), 1)
        print(f"{name}: {len(rep_rank)} changes, change rate {run.get('subst_change_rate', float('nan')):.3%}; "
              f"replaced top-k rank (0 = best) {np.round(rep_hist / max(rep_hist.sum(), 1), 3).tolist()}; "
              f"substitute rank median {int(np.searchsorted(q, 0.5))}, p90 {int(np.searchsorted(q, 0.9))}, "
              f"share at rank k (= next best) {sub_hist[k] / max(sub_hist.sum(), 1):.2f}; "
              f"layer rate min/median/max {layer_rate.min() if len(layer_rate) else 0:.3f}/"
              f"{np.median(layer_rate) if len(layer_rate) else 0:.3f}/{layer_rate.max() if len(layer_rate) else 0:.3f}",
              flush=True)


if __name__ == "__main__":
    main()
