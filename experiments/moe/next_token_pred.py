"""Multi-user plan 3.1: predict the NEXT token's experts of every decode stream, for similarity batching.

At decode position t of a request (its token t is about to run), known: everything up to token t-1. Per MoE layer l:
  P-prev     the route of token t-1 (A0's predictor)
  P-anchor   kNN over a pool of decode records on the centred router scores of token t-1 at layer l (as in
             anchor_dump: global search, the request itself excluded); vote, weighted by similarity, over what the
             neighbours' NEXT token used at layer l
  P-mix      anchor vote + beta x [in the route of token t-1]
Position 0 (no previous decode token) uses the last prompt token's route for P-prev and falls back to it for all.
Output per request (same stem as the trace): next_prev / next_anchor / next_mix int16 [D, L, k] (top-k ids, ranked)
and the recall@k of each predictor per layer (printed and saved to --summary).

  python experiments/moe/next_token_pred.py --trace outputs/moe/traces_v3/GLM-4.5-Air_cwq_M0_cot_chat_cap8 \
      --pool outputs/moe/traces_v2/GLM-4.5-Air_train_M0_cap8 --out outputs/moe/next_pred/GLM-4.5-Air_cot
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

M_VOTE = 8
_C = {}


def unit(x):
    x = x - x.mean(-1, keepdims=True)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def load(dirname, need_pre=False):
    out = []
    for f in sorted(glob.glob(dirname + "/q*.npz")):
        z = np.load(f)
        D = z["dec_idx"].shape[1]
        if D < 2:
            continue
        r = dict(stem=os.path.basename(f), R=z["dec_idx"].transpose(1, 0, 2).astype(np.int64),     # [D, L, k]
                 S=z["dec_score"].transpose(1, 0, 2))                                            # [D, L, E] fp16
        if need_pre:
            r["pre_last"] = z["pre_idx"][:, -1].astype(np.int64)                                 # [L, k]
        out.append(r)
    return out


def layer_job(l):
    pool, test, E, k, beta = (_C[x] for x in ("pool", "test", "E", "k", "beta"))
    fp = unit(pool["S"][:, l].astype(np.float32))                 # records with a next token in the same request
    lab = pool["Rnext"][:, l]                                      # [Np, k]
    nq = len(test["q_req"])
    vote_ids = np.zeros((nq, k), np.int16)
    mix_ids = np.zeros((nq, k), np.int16)
    for c0 in range(0, nq, 1024):                                   # 1024 x pool similarities per chunk
        sl = slice(c0, c0 + 1024)
        fq = unit(test["S"][sl, l].astype(np.float32))
        s = fq @ fp.T
        s[test["q_req"][sl, None] == pool["req"][None, :]] = -np.inf     # never the request itself
        top = np.argpartition(-s, M_VOTE, -1)[:, :M_VOTE]
        w = np.maximum(np.take_along_axis(s, top, -1), 1e-3)
        w[~np.isfinite(w)] = 0.0
        vote = np.zeros((len(fq), E), np.float32)
        for j in range(M_VOTE):
            np.add.at(vote, (np.arange(len(fq))[:, None], lab[top[:, j]]), w[:, j:j + 1])
        vote /= w.sum(1, keepdims=True) + 1e-8
        prev = np.zeros_like(vote)
        np.put_along_axis(prev, test["prev"][sl, l], 1.0, -1)
        vote_ids[sl] = np.argsort(-(vote + 1e-3 * prev), -1, kind="stable")[:, :k]
        mix_ids[sl] = np.argsort(-(vote + beta * prev), -1, kind="stable")[:, :k]
    return l, vote_ids, mix_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, help="the workload (all its requests are both queries and pool)")
    ap.add_argument("--pool", default=None, help="an extra pool workload (e.g. Wtrain)")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    test = load(a.trace, need_pre=True)
    extra = load(a.pool) if a.pool else []
    E, k = test[0]["S"].shape[-1], test[0]["R"].shape[-1]
    L = test[0]["R"].shape[1]

    def flat_pool(reqs, off):
        S = np.concatenate([r["S"][:-1] for r in reqs])                       # token i ...
        Rn = np.concatenate([r["R"][1:] for r in reqs])                       # ... and what token i+1 used
        req = np.concatenate([np.full(len(r["R"]) - 1, off + i) for i, r in enumerate(reqs)])
        return S, Rn, req
    S1, R1, q1 = flat_pool(test, 0)
    parts = [(S1, R1, q1)] + ([flat_pool(extra, -10 ** 6)] if extra else [])
    pool = dict(S=np.concatenate([p[0] for p in parts]), Rnext=np.concatenate([p[1] for p in parts]),
                req=np.concatenate([p[2] for p in parts]))
    # queries: every decode position t >= 1 of every request, described by token t-1; position 0 handled below
    qS = np.concatenate([r["S"][:-1] for r in test])
    qprev = np.concatenate([r["R"][:-1] for r in test])
    qreq = np.concatenate([np.full(len(r["R"]) - 1, i) for i, r in enumerate(test)])
    _C.update(pool=pool, test=dict(S=qS, prev=qprev, q_req=qreq), E=E, k=k, beta=a.beta)
    print(f"{len(test)} requests, {len(qreq)} queries, pool {len(pool['req'])} records, L={L}", flush=True)
    vote = np.zeros((len(qreq), L, k), np.int16)
    mix = np.zeros_like(vote)
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("fork")) as ex:
        for l, v, m in ex.map(layer_job, range(L)):
            vote[:, l], mix[:, l] = v, m
    os.makedirs(a.out, exist_ok=True)
    rec = {m: np.zeros(L) for m in ("prev", "anchor", "mix")}
    n = 0
    i0 = 0
    for i, r in enumerate(test):
        D = len(r["R"])
        first = r["pre_last"][None]                                             # position 0: last prompt token
        arr = dict(next_prev=np.concatenate([first, r["R"][:-1]]).astype(np.int16),
                   next_anchor=np.concatenate([first, vote[i0:i0 + D - 1]]).astype(np.int16),
                   next_mix=np.concatenate([first, mix[i0:i0 + D - 1]]).astype(np.int16))
        np.savez_compressed(os.path.join(a.out, r["stem"]), **arr)
        true = r["R"]
        for m in rec:
            p = arr["next_" + m].astype(np.int64)
            rec[m] += ((p[1:, :, :, None] == true[1:, :, None, :]).any(-1).mean(-1)).sum(0)    # [L]
        n += D - 1
        i0 += D - 1
    summ = {m: (rec[m] / n).round(4).tolist() for m in rec}
    for m in rec:
        print(f"  {m:7} recall@{k} mean {np.mean(summ[m]):.3f}  first/mid/last layer {summ[m][0]:.3f} / "
              f"{summ[m][L // 2]:.3f} / {summ[m][-1]:.3f}", flush=True)
    json.dump(dict(trace=a.trace, pool=a.pool, beta=a.beta, recall=summ), open(os.path.join(a.out, "summary.json"), "w"))


if __name__ == "__main__":
    main()
