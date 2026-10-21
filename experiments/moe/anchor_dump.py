"""Anchor plan P3 input: per-token, per-layer multi-layer-ahead expert predictions for the simulator.

For every decode token of the selected test requests and every current layer l, ranks the experts of
layer l+h (h = 1..H) under several predictors and keeps the top M ids, plus the anchor confidence
(highest router-score similarity found in the pool). Pool and similarity as in anchor_offline_v2.py
(global kNN on the centred router scores; for --self_pool workloads the pool also holds the workload's
other requests, never the request itself).

Output: one npz per request, same stem as the trace file:
  GATE, TOKGATE, AV, AVG, ORACLE   int16 [L, D, H, M]   expert ids ranked best first (-1 where l+h >= L)
  TOKGATE_s, AV_s, AVG_s           float16 [L, D, H, M]  the scores of those ranked ids (AV: vote share in [0, 1];
                                                         AVG: vote share + beta * [in GATE set])
  conf                             float16 [L, D]        top-1 similarity at layer l
GATE ranks the recorded top-k of the layer-(l+h) gate first, then fills up with TOKID; TOKGATE = TOKID +
beta*GATE; AV = similarity-weighted vote of the m nearest; AVG = AV + beta*GATE; ORACLE = the true top-k first.

  python experiments/moe/anchor_dump.py --dir outputs/moe/traces_v3 --pool_dir outputs/moe/traces_v2 \
      --test cwq_M0_cap8 --out outputs/moe/anchor_preds
"""
import argparse
import glob
import multiprocessing as mp
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from anchor_offline_v2 import BETA, M_VOTE, onehot, unit

_CTX = {}


def load_reqs(dirname, n=None):
    files = sorted(glob.glob(dirname + "/q*.npz"))
    reqs = []
    for f in files:
        z = np.load(f)
        D = z["dec_idx"].shape[1]
        reqs.append(dict(stem=os.path.basename(f), tok=z["gen_ids"][:D], R=z["dec_idx"].transpose(1, 0, 2),
                         S=z["dec_score"].transpose(1, 0, 2).astype(np.float32),
                         G=z["dec_gate_h"].transpose(2, 0, 1, 3)))
    return reqs[:n] if n else reqs


def flat(reqs, offset=0):
    keep = [r for r in reqs if len(r["tok"])]
    return dict(tok=np.concatenate([r["tok"] for r in keep]), R=np.concatenate([r["R"] for r in keep]),
                S=np.concatenate([r["S"] for r in keep]),
                req=np.concatenate([np.full(len(r["tok"]), offset + i) for i, r in enumerate(keep)]))


def layer_job(l):
    pool, test, E, k, L, H, M, excl = (_CTX[x] for x in ("pool", "test", "E", "k", "L", "H", "M", "excl"))
    nT = len(test["tok"])
    fp, ft = unit(pool["S"][:, l], True), unit(test["S"][:, l], True)
    idx = np.zeros((nT, M_VOTE), np.int64)
    sim = np.full((nT, M_VOTE), -np.inf, np.float32)
    for c0 in range(0, nT, 2048):
        s = ft[c0:c0 + 2048] @ fp.T
        if excl:
            s[test["req"][c0:c0 + 2048, None] == pool["req"][None, :]] = -np.inf
        top = np.argpartition(-s, M_VOTE, -1)[:, :M_VOTE]
        ts = np.take_along_axis(s, top, -1)
        o = np.argsort(-ts, -1)
        idx[c0:c0 + 2048], sim[c0:c0 + 2048] = np.take_along_axis(top, o, -1), np.take_along_axis(ts, o, -1)
    ok = np.isfinite(sim)
    w = np.where(ok, np.maximum(sim, 1e-3), 0.0)
    byid = defaultdict(list)
    for i, t in enumerate(pool["tok"]):
        byid[int(t)].append(i)
    out = {m: np.full((nT, H, M), -1, np.int16) for m in ("GATE", "TOKGATE", "AV", "AVG", "ORACLE")}
    out.update({m + "_s": np.zeros((nT, H, M), np.float16) for m in ("TOKGATE", "AV", "AVG")})
    rank = lambda sc: np.argsort(-sc, -1, kind="stable")[:, :M].astype(np.int16)

    def put(name, h, sc):
        r = rank(sc)
        out[name][:, h - 1] = r
        if name + "_s" in out:                      # the ranked scores, for confidence-thresholded prefetch
            out[name + "_s"][:, h - 1] = np.take_along_axis(sc, r.astype(np.int64), -1)
    for h in range(1, H + 1):
        if l + h >= L:
            break
        poh = onehot(pool["R"][:, l + h], E)
        gfreq = poh.mean(0)
        tok = np.stack([poh[byid[int(t)]].mean(0) if int(t) in byid else gfreq for t in test["tok"]])
        vote = (poh[idx] * w[..., None]).sum(1) / (w.sum(1, keepdims=True) + 1e-8)
        gate = onehot(test["G"][:, h - 1, l], E) if h <= test["G"].shape[1] else np.zeros_like(tok)
        put("GATE", h, gate + 1e-3 * tok)
        put("TOKGATE", h, tok + BETA * gate)
        put("AV", h, vote + 1e-3 * tok)
        put("AVG", h, vote + BETA * gate + 1e-3 * tok)
        put("ORACLE", h, onehot(test["R"][:, l + h], E) + 1e-3 * tok)
    return l, out, sim[:, 0].astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/moe/traces_v3")
    ap.add_argument("--pool_dir", default="outputs/moe/traces_v2")
    ap.add_argument("--model", default="GLM-4.5-Air")
    ap.add_argument("--pool", default="train_M0_cap8")
    ap.add_argument("--test", required=True)
    ap.add_argument("--self_pool", action="store_true", help="add the workload's other requests to the pool")
    ap.add_argument("--n", type=int, default=None, help="first n requests of the workload")
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--workers", type=int, default=44)
    ap.add_argument("--out", default="outputs/moe/anchor_preds")
    a = ap.parse_args()
    pool_reqs = load_reqs(f"{a.pool_dir}/{a.model}_{a.pool}")
    all_test = load_reqs(f"{a.dir}/{a.model}_{a.test}")
    test_reqs = all_test[:a.n] if a.n else all_test
    pool = flat(pool_reqs, offset=-10 ** 6)
    if a.self_pool:
        sp = flat(all_test)
        pool = {key: np.concatenate([pool[key], sp[key]]) for key in pool}
    test = flat(test_reqs)
    test["G"] = np.concatenate([r["G"] for r in test_reqs if len(r["tok"])])
    N, L, k = pool["R"].shape
    E = pool["S"].shape[-1]
    print(f"{a.test}: {len(test_reqs)} requests, {len(test['tok'])} decode tokens; pool {N}", flush=True)
    _CTX.update(pool=pool, test=test, E=E, k=k, L=L, H=a.H, M=a.M, excl=a.self_pool)
    res = {}
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("fork")) as ex:
        for l, out, conf in ex.map(layer_job, range(L)):
            res[l] = (out, conf)
    odir = f"{a.out}/{a.model}_{a.test}"
    os.makedirs(odir, exist_ok=True)
    i0 = 0
    for r in test_reqs:
        D = len(r["tok"])
        sl = slice(i0, i0 + D)
        arrs = {m: np.stack([res[l][0][m][sl] for l in range(L)]) for m in res[0][0]}
        arrs["conf"] = np.stack([res[l][1][sl] for l in range(L)])
        np.savez_compressed(f"{odir}/{r['stem']}", **arrs)
        i0 += D
    # sanity: recall@k of each predictor (should match anchor_offline_v2)
    for m in ("GATE", "TOKGATE", "AV", "AVG"):
        rr = []
        for h in (1, 3, 6):
            hit = []
            for l in range(L - h):
                p = res[l][0][m][:, h - 1, :k]
                act = test["R"][:, l + h]
                hit.append((p[:, :, None] == act[:, None, :]).any(-1).mean())
            rr.append(np.mean(hit))
        print(f"  {m:8} recall@k h=1 {rr[0]:.3f}  h=3 {rr[1]:.3f}  h=6 {rr[2]:.3f}", flush=True)
    print("wrote", odir)


if __name__ == "__main__":
    main()
