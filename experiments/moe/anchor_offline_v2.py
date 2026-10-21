"""Anchor-pool plan P0/P2 on the v2 traces (collect_traces_v2.py): how well can the experts of layer l+h be
predicted at layer l, per decode token, from a pool of earlier routing trajectories?

Pool: every decode token of Wtrain (and, for a long-decode workload, the other requests of that workload).
Test: every decode token t of a workload, every current layer l with l+h inside the model.

Predictors of the top-k set at layer l+h:
  TOKID          experts most often used at l+h by this token id in the pool (global frequency if unseen)
  GATE           HOBBIT-style: layer-l MoE input through the layer-(l+h) gate (recorded online, h <= --gate_ahead)
  A1-<sim>       the single most similar pool record's experts at l+h
  AV-<sim>       similarity-weighted vote of the m most similar pool records
  AV+GATE-<sim>  vote fraction + beta * [e in GATE set]   (h <= 4)
  TOK+GATE       TOKID frequency + beta * [e in GATE set]  (baseline for AV+GATE: same inputs minus the pool search)
Similarity of the state at layer l (and before):
  route<w>       expert-set overlap summed over layers l-w+1..l (pool restricted to the same token id)
  logit          centred cosine of the layer-l router selection scores
  proj           cosine of the 64-d random projection of the layer-l MoE input
  *-tok          pool restricted to the same token id; *-glob  nearest neighbours over the whole pool
  *-hier         same token id when the token is in the pool, else global
  <method>@seen  the same metric on the test records whose token id is in the pool (no fallback involved)
Records with no candidate (unseen token) fall back to GATE (h <= 4) else TOKID.

  python experiments/moe/anchor_offline_v2.py --dir outputs/moe/traces_v2 --model GLM-4.5-Air \
      --test cwq_M0_cap8 webqsp_M0_cap8 cwq_M0_synr_cap8 --out outputs/moe/anchor_v2_glm.json
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HMAX = 8
M_VOTE = 8
BETA = 0.5
TAUS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def load(dirname):
    """Decode records of every request: token ids, routes [N,L,k], scores [N,L,E], proj [N,L,P], gate [N,4,L,k]."""
    tok, R, S, P, G, req = [], [], [], [], [], []
    for i, f in enumerate(sorted(glob.glob(dirname + "/q*.npz"))):
        z = np.load(f)
        D = z["dec_idx"].shape[1]
        if D == 0:
            continue
        tok.append(z["gen_ids"][:D])
        R.append(z["dec_idx"].transpose(1, 0, 2))
        S.append(z["dec_score"].transpose(1, 0, 2))
        P.append(z["dec_proj"].transpose(1, 0, 2))
        G.append(z["dec_gate_h"].transpose(2, 0, 1, 3))
        req.append(np.full(D, i))
    return dict(tok=np.concatenate(tok), R=np.concatenate(R).astype(np.int16), S=np.concatenate(S).astype(np.float32),
                P=np.concatenate(P).astype(np.float32), G=np.concatenate(G), req=np.concatenate(req))


def unit(x, centre=False):
    if centre:
        x = x - x.mean(-1, keepdims=True)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def onehot(R, E):
    """[..., k] expert ids -> [..., E] 0/1."""
    out = np.zeros(R.shape[:-1] + (E,), np.float32)
    np.put_along_axis(out, R.astype(np.int64), 1.0, -1)
    return out


def recall(score, actual, k):
    """score [n,E], actual [n,k] -> (recall@k, recall@2k) summed over rows."""
    top = np.argsort(-score, -1)
    a1 = onehot(actual, score.shape[-1])
    r1 = np.take_along_axis(a1, top[:, :k], -1).sum() / k
    r2 = np.take_along_axis(a1, top[:, :2 * k], -1).sum() / k
    return r1, r2


_CTX = {}                                           # set before forking the workers (inherited, not pickled)


def layer_job(l):
    """All predictors for current layer l; returns sums keyed by (method, h)."""
    pool, test, E, k, L, routes_w, same_req_excl = (_CTX[x] for x in ("pool", "test", "E", "k", "L", "routes_w", "excl"))
    acc = defaultdict(lambda: np.zeros(3))          # recall@k sum, recall@2k sum, count
    cov = defaultdict(lambda: np.zeros(4))          # (sim, tau): covered, vote recall sum, total, GATE recall sum
    hs = [h for h in range(1, HMAX + 1) if l + h < L]
    if not hs:
        return dict(acc), dict(cov)
    nT = len(test["tok"])
    # ---- TOKID tables for the needed target layers
    pool_oh = {h: onehot(pool["R"][:, l + h], E) for h in hs}
    glob_freq = {h: pool_oh[h].mean(0) for h in hs}
    byid = defaultdict(list)
    for i, t in enumerate(pool["tok"]):
        byid[int(t)].append(i)
    byid = {t: np.array(v) for t, v in byid.items()}
    # ---- similarities: per test record, candidate indices and similarity (top M_VOTE)
    feats = {"logit": (unit(pool["S"][:, l], True), unit(test["S"][:, l], True)),
             "proj": (unit(pool["P"][:, l]), unit(test["P"][:, l]))}
    nbrs = {}                                       # name -> (idx [nT,M] or -1, sim [nT,M])
    for name, (fp, ft) in feats.items():
        # global kNN
        idx = np.full((nT, M_VOTE), -1, np.int64)
        sim = np.full((nT, M_VOTE), -np.inf, np.float32)
        for c0 in range(0, nT, 2048):
            s = ft[c0:c0 + 2048] @ fp.T
            if same_req_excl:
                s[test["req"][c0:c0 + 2048, None] == pool["req"][None, :]] = -np.inf
            top = np.argpartition(-s, M_VOTE, -1)[:, :M_VOTE]
            ts = np.take_along_axis(s, top, -1)
            o = np.argsort(-ts, -1)
            idx[c0:c0 + 2048], sim[c0:c0 + 2048] = np.take_along_axis(top, o, -1), np.take_along_axis(ts, o, -1)
        nbrs[f"{name}-glob"] = (idx, sim)
    # token-keyed (logit, proj, route-w)
    for name in ["logit", "proj"] + [f"route{w}" for w in routes_w]:
        idx = np.full((nT, M_VOTE), -1, np.int64)
        sim = np.full((nT, M_VOTE), -np.inf, np.float32)
        nbrs[f"{name}-tok"] = (idx, sim)
    test_byid = defaultdict(list)
    for i, t in enumerate(test["tok"]):
        test_byid[int(t)].append(i)
    for t, ti in test_byid.items():
        if t not in byid:
            continue
        ti, ci = np.array(ti), byid[t]
        for name in ["logit", "proj"] + [f"route{w}" for w in routes_w]:
            if name in feats:
                s = feats[name][1][ti] @ feats[name][0][ci].T
            else:
                w = int(name[5:])
                lo = max(0, l - w + 1)
                a = onehot(test["R"][ti, lo:l + 1], E).reshape(len(ti), -1)
                b = onehot(pool["R"][ci, lo:l + 1], E).reshape(len(ci), -1)
                s = (a @ b.T) / (k * (l + 1 - lo))
            if same_req_excl:
                s[test["req"][ti, None] == pool["req"][None, ci]] = -np.inf
            m = min(M_VOTE, len(ci))
            top = np.argsort(-s, -1)[:, :m]
            idx_, sim_ = nbrs[f"{name}-tok"]
            idx_[ti, :m] = ci[top]
            sim_[ti, :m] = np.take_along_axis(s, top, -1)
    for name in ("logit", "proj"):
        ti, ts = nbrs[f"{name}-tok"]
        gi, gs = nbrs[f"{name}-glob"]
        seen_ = np.isfinite(ts[:, 0])
        nbrs[f"{name}-hier"] = (np.where(seen_[:, None], ti, gi), np.where(seen_[:, None], ts, gs))
    seen = np.isfinite(nbrs["logit-tok"][1][:, 0])     # token id present in the pool

    def add(meth, h, sc, actual):
        r1, r2 = recall(sc, actual, k)
        acc[(meth, h)] += (r1, r2, nT)
        if seen.any():
            r1, r2 = recall(sc[seen], actual[seen], k)
            acc[(meth + "@seen", h)] += (r1, r2, seen.sum())

    # breakdown for h = 3: same-token vs global search, by how often the token id occurs in the pool and by depth
    cnt = np.array([len(byid.get(int(t), ())) for t in test["tok"]])
    depth = "shallow" if l < L // 3 else ("middle" if l < 2 * L // 3 else "deep")
    if l + 3 < L:
        a3 = test["R"][:, l + 3]
        oh3 = pool_oh[3]
        for name in ("logit-tok", "logit-glob"):
            idx, sim = nbrs[name]
            ok = np.isfinite(sim) & (idx >= 0)
            w = np.where(ok, np.maximum(sim, 1e-3), 0.0)
            vote = (oh3[np.where(ok, idx, 0)] * w[..., None]).sum(1) / (w.sum(1, keepdims=True) + 1e-8)
            for bname, lo, hi in (("n1-7", 1, 8), ("n8-63", 8, 64), ("n64+", 64, 10 ** 9)):
                m = (cnt >= lo) & (cnt < hi)
                if m.any():
                    r1, _ = recall(vote[m], a3[m], k)
                    acc[(f"BRK-{name}|{bname}", 3)] += (r1, 0, m.sum())
            m = cnt > 0
            r1, _ = recall(vote[m], a3[m], k)
            acc[(f"BRK-{name}|{depth}", 3)] += (r1, 0, m.sum())
            # mean best similarity, to see whether same-token neighbours are less similar
            acc[(f"SIM-{name}|{depth}", 3)] += (np.where(m, sim[:, 0], 0).sum(), np.where(m, sim[:, M_VOTE - 1], 0).sum(), m.sum())

    tok_tab = {}
    for h in hs:
        tab = np.stack([pool_oh[h][byid[int(t)]].mean(0) if int(t) in byid else glob_freq[h] for t in test["tok"]])
        tok_tab[h] = tab
    for h in hs:
        actual = test["R"][:, l + h]
        gate = onehot(test["G"][:, h - 1, l], E) if h <= test["G"].shape[1] else None
        fallback = gate if gate is not None else tok_tab[h]
        base = [("TOKID", tok_tab[h])]
        if gate is not None:
            base += [("GATE", gate), ("TOK+GATE", tok_tab[h] + BETA * gate)]
        for meth, sc in base:
            add(meth, h, sc, actual)
        for name, (idx, sim) in nbrs.items():
            ok = np.isfinite(sim) & (idx >= 0)
            w = np.where(ok, np.maximum(sim, 1e-3), 0.0)
            oh = pool_oh[h][np.where(ok, idx, 0)]                      # [nT,M,E]
            vote = (oh * w[..., None]).sum(1) / (w.sum(1, keepdims=True) + 1e-8)
            one = oh[:, 0]
            has = ok[:, 0]
            for meth, sc in [(f"A1-{name}", one), (f"AV-{name}", vote)]:
                add(meth, h, np.where(has[:, None], sc, fallback), actual)
            if gate is not None:
                add(f"AV+GATE-{name}", h, np.where(has[:, None], vote + BETA * gate, gate), actual)
            if not name.startswith("route"):
                best = np.where(has, sim[:, 0], -np.inf)
                a1 = onehot(actual, E)
                topv = np.argsort(-vote, -1)[:, :k]
                hit = np.take_along_axis(a1, topv, -1).sum(-1) / k
                ghit = (np.take_along_axis(a1, np.argsort(-gate, -1)[:, :k], -1).sum(-1) / k
                        if gate is not None else np.full(nT, np.nan))
                for tau in TAUS:
                    c = best >= tau
                    # covered count, anchor-vote recall on covered, total, GATE recall on the same covered tokens
                    cov[(name, tau, h)] += (c.sum(), hit[c].sum(), nT, ghit[c].sum())
    return dict(acc), dict(cov)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/moe/traces_v2")
    ap.add_argument("--model", default="GLM-4.5-Air")
    ap.add_argument("--pool", default="train_M0_cap8")
    ap.add_argument("--pool_dir", default=None, help="directory of the pool workload (default: --dir)")
    ap.add_argument("--test", nargs="+", default=["cwq_M0_cap8", "webqsp_M0_cap8", "cwq_M0_synr_cap8"])
    ap.add_argument("--self_pool", nargs="*", default=["cwq_M0_cot_chat_cap8"],
                    help="workloads whose own other requests are added to the pool (leave-one-request-out)")
    ap.add_argument("--max_test", type=int, default=20000, help="subsample test records (CoT has ~150k)")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default="outputs/moe/anchor_v2_glm.json")
    a = ap.parse_args()
    pool = load(f"{a.pool_dir or a.dir}/{a.model}_{a.pool}")
    pool["req"] = pool["req"] - 10 ** 6                  # never equal to a test request id
    N, L, k = pool["R"].shape
    E = pool["S"].shape[-1]
    print(f"pool {a.pool}: {N} decode tokens, L={L} k={k} E={E}", flush=True)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for wl in a.test:
        d = f"{a.dir}/{a.model}_{wl}"
        if not os.path.isdir(d) or not os.path.exists(d + ".json"):
            print("skip (not finished)", wl)
            continue
        test = load(d)
        excl = wl in a.self_pool
        pl = pool
        if excl:
            pl = {key: np.concatenate([pool[key], test[key]]) for key in pool if key != "G"}   # pool gates unused
        n = len(test["tok"])
        if n > a.max_test:
            sel = np.sort(np.random.default_rng(0).choice(n, a.max_test, replace=False))
            test = {key: v[sel] for key, v in test.items()}
        print(f"{wl}: {len(test['tok'])} test tokens (of {n}), pool {len(pl['tok'])}", flush=True)
        acc, cov = defaultdict(lambda: np.zeros(3)), defaultdict(lambda: np.zeros(4))
        _CTX.update(pool=pl, test=test, E=E, k=k, L=L, routes_w=(1, 2, 3), excl=excl)
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("fork")) as ex:
            for ac, cv in ex.map(layer_job, range(L - 1)):
                for key, v in ac.items():
                    acc[key] += v
                for key, v in cv.items():
                    cov[key] += v
        out = {"n_test": int(len(test["tok"])), "pool": int(len(pl["tok"])),
               "recall": {f"{m}|{h}": [v[0] / v[2], v[1] / v[2]] for (m, h), v in acc.items()},
               "coverage": {f"{s}|{t}|{h}": [v[0] / v[2], v[1] / max(v[0], 1), v[3] / max(v[0], 1)] for (s, t, h), v in cov.items()}}
        res[wl] = out
        json.dump(res, open(a.out, "w"), indent=1)
        meths = sorted({m for m, _ in acc})
        print(f"{'method':24}" + "".join(f"  h={h}" for h in range(1, HMAX + 1)))
        for m in meths:
            print(f"{m:24}" + "".join(f" {out['recall'].get(f'{m}|{h}', [np.nan])[0]:.3f}" for h in range(1, HMAX + 1)),
                  flush=True)


if __name__ == "__main__":
    main()
