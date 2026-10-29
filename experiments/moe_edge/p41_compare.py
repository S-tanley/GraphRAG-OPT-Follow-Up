"""Multi-user plan 4.1.3: the Mac prototype (subst_offload.py) against the simulator on the same routes.

For each configuration of a prototype run (<prefix>.json, traces in <prefix>_<subst>/), tsim_mu.run_seq replays the
requests back to back with the same cache (per-layer pools from the calibration counts, warm start, LFU) and the same
substitution budget, and the decode numbers are compared: TPOT, loads / token, stall / token, change rate.
--fit derives one measured hardware profile from the no-substitution runs of all prefixes (measured_hw) and predicts
every run with it as well; the substitution runs are then out-of-sample.

  python experiments/moe_edge/p41_compare.py outputs/moe_edge/subst/p41_glm_f0.5 outputs/moe_edge/subst/p41_glm_f0.75
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tsim_mu as mu  # noqa: E402


def measured_hw(runs, L, base="mac_ssd"):
    """A profile from no-substitution prototype runs only (the substitution runs stay out-of-sample):
      load      wait per expert read = decode stall / decode loads (pooled)
      load_cpu  with >= 2 runs at different load rates: the slope of TPOT on loads / token minus `load`, i.e. the host
                work per read that is not waiting (thread hand-off, binding); 0 with one run
      tick      set so that a token with no loads takes the regression intercept (one run: TPOT - stall)
    The attention table and per-expert cost of `base` are kept."""
    hw = dict(mu.HW[base])
    ld = np.array([r["summary"]["loads_per_tok"] for r in runs])
    st = np.array([r["summary"]["stall_ms_per_tok"] for r in runs])
    tp = np.array([r["summary"]["tpot_ms_mean"] for r in runs])
    if ld.sum() > 0:
        hw["load"] = 1e3 * st.sum() / ld.sum()
    if len(runs) >= 2 and np.ptp(ld) > 1:
        slope, icpt = np.polyfit(ld, tp, 1)
        hw["load_cpu"] = max(0.0, 1e3 * slope - hw["load"])
        comp = icpt
    else:
        comp = float(np.mean(tp - st))
    a1 = dict(hw["attn"])[1]
    model = (a1 + hw["tick"] + mu.gmm_cost(hw, 8)) * L / 1e3                                    # ms / token
    hw["tick"] = hw["tick"] + 1e3 * (comp - model) / L
    return hw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefixes", nargs="+")
    ap.add_argument("--hw", default="mac_ssd")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--fit_all", action="store_true", help="also the fitted profile with load_cpu on every read")
    ap.add_argument("--only_fit", action="store_true", help="skip the base profile")
    ap.add_argument("--preds", nargs="*", default=[], help="trace_dir=preds_dir pairs (GATEs from gate_calib.py preds)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    a.preds = dict(x.split("=", 1) for x in a.preds)
    rows = []
    metas = {pre: json.load(open(pre + ".json")) for pre in a.prefixes}
    base_runs = [r for m in metas.values() for r in m["runs"]
                 if r["summary"]["subst"] == "none" and r["summary"].get("prefetch", "none") == "none"]
    for pre in a.prefixes:
        meta = metas[pre]
        cal = mu.load_reqs(meta["calib"])
        L, E = cal[0]["dec"].shape[0], 128
        f = mu.use_counts(cal, L, E)
        C = meta["C"]
        caps = mu.layer_caps(f, C, L, E, meta["floor"])
        assert caps.tolist() == meta["caps"], "pool sizes differ from the prototype's"
        hws = [(a.hw, mu.HW[a.hw])] + ([("fit", measured_hw(base_runs, L, a.hw))] if a.fit and base_runs else [])
        if a.fit_all and a.fit and base_runs:            # the same, prefetch reads paying the host work too
            hws.append(("fit_all", dict(measured_hw(base_runs, L, a.hw), load_cpu_all=True)))
        if a.only_fit:
            hws = [h for h in hws if h[0] != a.hw]
        for run in meta["runs"]:
            m = run["summary"]
            tdir = m.get("trace_dir") or f"{pre}_{m['subst'].replace(':', '')}"
            pf = m.get("prefetch", "none")
            names = mu.pred_names(mu.parse_pref(pf))
            reqs = mu.load_reqs(tdir, preds=a.preds.get(tdir) if names else None, names=names,
                                need_score=m["subst"] != "none")
            for name, hw in hws:
                s = mu.run_seq(reqs, f"{meta['policy']}/{pf}", hw, C, E, subst=m["subst"], warm_f=f, caps=caps)
                row = dict(prefix=os.path.basename(pre), cache=meta["cache"], subst=m["subst"], prefetch=pf, hw=name,
                           load_us=hw["load"], load_cpu_us=hw.get("load_cpu", 0.0), tick_us=hw["tick"],
                           real_tpot=m["tpot_ms_mean"], sim_tpot=s["tpot_ms"],
                           real_loads=m["loads_per_tok"], sim_loads=s["loads_per_tok"],
                           real_stall=m["stall_ms_per_tok"], sim_stall=s["stall_ms_per_tok"],
                           real_change=m["change_rate"], sim_change=s.get("subst_change_rate", 0.0),
                           real_late=m["late_rate"], sim_late=s.get("subst_late_rate"))
                rows.append(row)
    print("| run | subst | profile | TPOT real / sim (ms) | err | loads/tok real / sim | err | stall real / sim | "
          "changed real / sim |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        e1 = r["sim_tpot"] / r["real_tpot"] - 1
        e2 = r["sim_loads"] / r["real_loads"] - 1 if r["real_loads"] else float("nan")
        print(f"| {r['prefix']} | {r['subst']}{'' if r['prefetch'] == 'none' else ' + ' + r['prefetch']} | {r['hw']} (load {r['load_us']:.0f} + {r['load_cpu_us']:.0f} us) | "
              f"{r['real_tpot']:.0f} / "
              f"{r['sim_tpot']:.0f} | {100 * e1:+.0f}% | {r['real_loads']:.1f} / {r['sim_loads']:.1f} | "
              f"{100 * e2:+.0f}% | {r['real_stall']:.0f} / {r['sim_stall']:.0f} | "
              f"{100 * r['real_change']:.1f}% / {100 * r['sim_change']:.1f}% |")
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
