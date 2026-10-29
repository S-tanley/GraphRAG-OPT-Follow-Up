"""Multi-user plan 4.2.0 / 4.2.2: the calibrated GATE predictor (GATEs) for the Mac prototype and the simulator.

  build   P(expert in layer l+h's true top-k | h, normalised margin) from calibration traces that carry dec_gate_idx /
          dec_gate_score [H, L, D, M] (subst_offload.py --gate_h), with anchor_dump.gate_calibration (24 quantile bins
          per h, non-decreasing in the margin)
  probs   the probability of each of the top-M GATE-h experts (gate_probs), used online by the prototype
  preds   for the simulator: per request GATEs [L, D, H, M] (the top-M ranked by probability; ties keep the GATE
          score order) and GATEs_s (the probabilities), the anchor_dump layout tsim_mu reads with --preds

  python experiments/moe_edge/gate_calib.py build --traces outputs/moe_edge/subst/gatecal_f0.75_none \\
      --out outputs/moe_edge/subst/gatecal_glm4bit.npz
  python experiments/moe_edge/gate_calib.py preds --calib outputs/moe_edge/subst/gatecal_glm4bit.npz \\
      --traces <dir with dec_gate_*> --out <preds dir>
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "moe"))
from anchor_dump import gate_calibration, gate_prob  # noqa: E402


def load_gate_reqs(trace_dir):
    reqs = []
    for f in sorted(glob.glob(trace_dir + "/q*.npz")):
        z = np.load(f)
        if "dec_gate_idx" not in z.files or z["dec_idx"].shape[1] == 0:
            continue
        reqs.append(dict(GI=z["dec_gate_idx"].transpose(2, 0, 1, 3), GS=z["dec_gate_score"].transpose(2, 0, 1, 3),
                         R=z["dec_idx"].transpose(1, 0, 2).astype(np.int64), tok=z["gen_ids"][:z["dec_idx"].shape[1]],
                         stem=os.path.basename(f)))
    return reqs


def gate_probs(gs, h, k, edges, prob):
    """gs [..., M] the top-M selection scores of GATE-h (descending) -> calibrated probabilities [..., M]."""
    return gate_prob(gs, h, k, edges, prob)


def ranked(gi, gp):
    """Rank the top-M by probability; ties keep the GATE order (gi is sorted by score) -> ids, probabilities."""
    o = np.argsort(-gp, -1, kind="stable")
    return np.take_along_axis(gi, o, -1), np.take_along_axis(gp, o, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "preds"])
    ap.add_argument("--traces", required=True)
    ap.add_argument("--calib", default=None)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    reqs = load_gate_reqs(a.traces)
    if a.cmd == "build":
        edges, prob = gate_calibration(reqs, a.k)
        H = edges.shape[0]
        np.savez(a.out, edges=edges, prob=prob, k=a.k, n_requests=len(reqs), traces=a.traces)
        print(f"{len(reqs)} requests; P(hit) per h, lowest / highest bin:")
        for h in range(1, H + 1):
            print(f"  h={h}  {prob[h - 1, 0]:.3f} .. {prob[h - 1, -1]:.3f}")
        return
    c = np.load(a.calib)
    edges, prob, k = c["edges"], c["prob"], int(c["k"])
    os.makedirs(a.out, exist_ok=True)
    for r in reqs:
        D, H, L, M = r["GI"].shape
        ids = np.full((L, D, H, M), -1, np.int16)
        ps = np.zeros((L, D, H, M), np.float16)
        for h in range(1, H + 1):
            gi, gs = r["GI"][:, h - 1].astype(np.int64), r["GS"][:, h - 1]          # [D, L, M]
            ok = gi[..., 0] >= 0
            gp = np.where(ok[..., None], gate_probs(gs, h, k, edges, prob), 0.0)
            ri, rp = ranked(gi, gp)
            ids[:, :, h - 1] = np.where(ok[..., None], ri, -1).transpose(1, 0, 2)
            ps[:, :, h - 1] = rp.transpose(1, 0, 2)
        np.savez_compressed(os.path.join(a.out, r["stem"]), GATEs=ids, GATEs_s=ps)
    print(f"wrote GATEs for {len(reqs)} requests to {a.out}")


if __name__ == "__main__":
    main()
