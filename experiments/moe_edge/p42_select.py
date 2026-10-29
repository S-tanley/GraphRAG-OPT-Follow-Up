"""Multi-user plan 4.2.3: choose the prefetch threshold (tau0, kappa_log) for the Mac prototype on the calibration
requests only (CoT 0..31), with the simulator on their own routes and GATE-h predictions: seq mode, per-layer pools
from the calibration counts, warm start, LFU, both calibrated profiles (mac_ssd_cal: host work on demand reads only;
mac_ssd_cal_all: on every read). Criterion: the lowest geometric-mean TPOT over cache {50%, 75%} x both profiles with
substitution <= 10% (the configuration the prototype runs); the no-substitution numbers are reported too.

  python experiments/moe_edge/p42_select.py --trace outputs/moe_edge/subst/gatecal_f0.75_none \\
      --preds outputs/moe_edge/subst/gatecal_preds --calib outputs/moe_edge/traces/GLM-4.5-Air-4bit_cwq_M0_cot_chat_cap8 \\
      --out outputs/moe_edge/subst/p42_select.json
"""
import argparse
import itertools
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tsim_mu as mu  # noqa: E402

_D = {}


def job(j):
    pol, hw, f, sub = j
    reqs, cnt = _D["reqs"], _D["counts"]
    L, E = reqs[0]["dec"].shape[0], 128
    C = int(f * L * E)
    caps = mu.layer_caps(cnt, C, L, E, 16)
    r = mu.run_seq(reqs, f"lfu/{pol}", mu.HW[hw], C, E, subst=sub, warm_f=cnt, caps=caps)
    return dict(policy=pol, hw=hw, cache=f, subst=sub, tpot_ms=r["tpot_ms"], loads_per_tok=r["loads_per_tok"],
                stall_ms_per_tok=r["stall_ms_per_tok"], change=r.get("subst_change_rate", 0.0),
                pf_waste=r.get("pf_waste_frac", 0.0), la=r["loads"]["la"], demand=r["loads"]["demand"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pols = ["none"] + [f"ah8:GATEs:m8:p{t}" + (f":kl{k}" if k else "") for t in (0.3, 0.5, 0.7)
                       for k in (0, 0.05, 0.1, 0.2)]
    names = set().union(*[mu.pred_names(mu.parse_pref(p)) for p in pols])
    _D["reqs"] = mu.load_reqs(a.trace, preds=a.preds, names=names, need_score=True)
    L = _D["reqs"][0]["dec"].shape[0]
    _D["counts"] = mu.use_counts(mu.load_reqs(a.calib), L, 128)
    jobs = list(itertools.product(pols, ["mac_ssd_cal", "mac_ssd_cal_all"], [0.5, 0.75], ["rate:0.1", "none"]))
    with ProcessPoolExecutor(min(a.workers, len(jobs)), mp_context=mp.get_context("fork")) as ex:
        rows = list(ex.map(job, jobs))
    json.dump(rows, open(a.out, "w"), indent=1)
    print(f"{len(_D['reqs'])} calibration requests; geo-mean TPOT over cache x profile (ms):")
    print("| policy | subst 10% | no subst | 50% / 75% with subst (cal, cal_all) |")
    print("|---|---|---|---|")
    best = None
    for p in pols:
        g = {s: float(np.exp(np.mean([np.log(r["tpot_ms"]) for r in rows if r["policy"] == p and r["subst"] == s])))
             for s in ("rate:0.1", "none")}
        det = " ".join(f"{r['tpot_ms']:.0f}" for r in rows if r["policy"] == p and r["subst"] == "rate:0.1")
        print(f"| {p} | {g['rate:0.1']:.1f} | {g['none']:.1f} | {det} |")
        if p != "none" and (best is None or g["rate:0.1"] < best[1]):
            best = (p, g["rate:0.1"])
    print(f"selected: {best[0]} (geo-mean TPOT with substitution {best[1]:.1f} ms)")


if __name__ == "__main__":
    main()
