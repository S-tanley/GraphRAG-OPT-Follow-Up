"""Checks for the v4 traces (multi-user plan 0.3): the stored GATE-h scores agree with the GATE-h ids, and a
re-collection reproduces an earlier one.

  (a) within a v4 trace: the top-k of dec_gate_idx equals dec_gate_h, per (h, l, token), as a set and in order, and
      the scores are non-increasing over all M (so the route's k experts score at least as high as the rest: ties
      allowed) and finite;
  (b) with --ref: same generated tokens, routes and GATE-h ids as the reference trace (e.g. traces_v3), per request.

  python experiments/moe/check_gate_scores.py --dir outputs/moe/traces_v4/GLM-4.5-Air_cwq_M0_cot_chat_cap8 \
      --ref outputs/moe/traces_v3/GLM-4.5-Air_cwq_M0_cot_chat_cap8 --n 50
"""
import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    files = sorted(glob.glob(a.dir + "/q*.npz"))[:a.n]
    tot = dict(cells=0, set_ok=0, order_ok=0, unsorted=0, nonfinite=0)
    ref = dict(reqs=0, gen_same=0, dec_same=0, gh_same=0, gh_cells=0, gh_cells_same=0, len_v4=0, len_ref=0)
    bad = []
    for f in files:
        z = np.load(f)
        gh, gi, gs = z["dec_gate_h"], z["dec_gate_idx"], z["dec_gate_score"]
        H, L, D, k = gh.shape
        valid = gh[..., 0] >= 0                                    # [H, L, D]: l + h inside the model
        top = gi[..., :k]
        same_set = (np.sort(top, -1) == np.sort(gh, -1)).all(-1)
        same_ord = (top == gh).all(-1)
        tot["cells"] += int(valid.sum())
        tot["set_ok"] += int((same_set & valid).sum())
        tot["order_ok"] += int((same_ord & valid).sum())
        s = gs.astype(np.float32)
        tot["unsorted"] += int(((np.diff(s, axis=-1) > 0).any(-1) & valid).sum())   # ties allowed, nothing above
        tot["nonfinite"] += int((~np.isfinite(s).all(-1) & valid).sum())
        if (~same_set & valid).any():
            bad.append(os.path.basename(f))
        if a.ref:
            rf = os.path.join(a.ref, os.path.basename(f))
            if not os.path.exists(rf):
                continue
            r = np.load(rf)
            ref["reqs"] += 1
            g4, gr = z["gen_ids"], r["gen_ids"]
            ref["len_v4"] += len(g4)
            ref["len_ref"] += len(gr)
            ref["gen_same"] += int(len(g4) == len(gr) and (g4 == gr).all())
            ref["dec_same"] += int(z["dec_idx"].shape == r["dec_idx"].shape and (z["dec_idx"] == r["dec_idx"]).all())
            rgh = r["dec_gate_h"]
            Dm = min(D, rgh.shape[2])
            Hm = min(H, rgh.shape[0])
            eq = (gh[:Hm, :, :Dm] == rgh[:Hm, :, :Dm]).all(-1)
            ref["gh_cells"] += int(eq.size)
            ref["gh_cells_same"] += int(eq.sum())
            ref["gh_same"] += int(gh.shape == rgh.shape and eq.all())
    res = dict(dir=a.dir, files=len(files), **tot,
               set_agree=tot["set_ok"] / max(tot["cells"], 1), order_agree=tot["order_ok"] / max(tot["cells"], 1),
               bad_files=bad[:20])
    if a.ref:
        res["ref"] = dict(ref, gh_cell_agree=ref["gh_cells_same"] / max(ref["gh_cells"], 1))
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
