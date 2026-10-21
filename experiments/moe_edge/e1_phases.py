"""Phase A / E1 by "when is it known": per-decode-token recall of request-level predictors available
  T0  at admission  (from the E1 export: GLOBAL, RPP, BASE_ADMIT, OURS_ADMIT, RPP_R)
  T1  after prefill, i.e. once the first generated token g0 is known
        FIRSTTOK   decode-position table of g0 only                               (no retrieval: control)
        CANDDEC    decode-position table averaged over all candidate entity tokens  (retrieval, no g0)
        CANDFIRST  g0 selects the candidate entity it starts; table over g0 + that entity's tokens
                   (retrieval + g0); falls back to CANDDEC when g0 starts no candidate
        PREFILL    this request's prefill expert frequency
  ORACLE      the request's actual decode distribution (upper bound of any request-level predictor)
The decode-position table maps a token id to the experts it is routed to when it is the INPUT of a decode
step (built on the Wtrain build split, same split as e1_analysis.py). Decode step s consumes gen_ids[s];
step 0 consumes g0 itself, so recall is reported on steps >= 1 (step 0 is known exactly at T1).

  python experiments/moe_edge/e1_phases.py --model GLM-4.5-Air --out outputs/moe_edge/e1_phases_glm.json
"""

import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "moe"))
from e1_analysis import ROOT, attach_paths, cand_positions, counts, load_run, norm, recall_at  # noqa: E402


def cand_seqs(r):
    """Token-id sequences of the candidate entities (each run of 'ent' tokens right before a newline)."""
    out, cur = [], []
    for c, t in zip(r["cat"], r["pids"]):
        if c == "ent":
            cur.append(int(t))
        else:
            if c == "nl" and cur:
                out.append(cur)
            cur = []
    return out


class DecTable:
    def __init__(self, train, E, min_n=3):
        L = train[0]["dec"].shape[0]
        tab, n = {}, {}
        glob = np.zeros((L, E))
        for r in train:
            for s in range(r["dec"].shape[1]):
                t = int(r["gids"][s])
                tab.setdefault(t, np.zeros((L, E)))[np.arange(L)[:, None], r["dec"][:, s]] += 1
                n[t] = n.get(t, 0) + 1
            glob += counts(r["dec"], E)
        self.tab = {t: norm(v) for t, v in tab.items() if n[t] >= min_n}
        self.glob = norm(glob)

    def prof(self, ids):
        vs = [self.tab[t] for t in ids if t in self.tab]
        return (norm(np.mean(vs, 0)), len(vs) / max(1, len(ids))) if vs else (None, 0.0)


def t1_predictors(r, T, E):
    g0 = int(r["gids"][0])
    out = {}
    out["FIRSTTOK"] = T.prof([g0])[0]
    cands = cand_seqs(r)
    cd = T.prof([t for c in cands for t in c])[0]
    out["CANDDEC"] = cd if cd is not None else T.glob
    m = [c for c in cands if c and c[0] == g0]
    if m:
        p = T.prof(m[0])[0]                       # the matched entity's tokens (g0 is its first token)
        out["CANDFIRST"] = p if p is not None else out["CANDDEC"]
    else:
        out["CANDFIRST"] = out["CANDDEC"]
    out["FIRSTTOK"] = out["FIRSTTOK"] if out["FIRSTTOK"] is not None else T.glob
    body = [i for i, c in enumerate(r["cat"]) if c != "pre"]
    out["PREFILL"] = norm(counts(r["pre"][:, body], E))
    out["ORACLE"] = norm(counts(r["dec"], E))
    out["GLOBAL"] = T.glob
    return out, bool(m)


def evaluate(reqs, T, E, k, t0=None, seed=0):
    Bs = [k, 2 * k, 4 * k]
    names = None
    per_req = {}
    matched = []
    for r in reqs:
        D = r["dec"].shape[1]
        if D < 2:
            continue
        preds, hit = t1_predictors(r, T, E)
        if t0 is not None and r["q"] in t0["qi"]:
            i = t0["qi"][r["q"]]
            for n in t0["names"]:
                preds[n] = t0["arr"][n][i].astype(np.float64)
        matched.append(hit)
        names = names or list(preds)
        for n in names:
            per_req.setdefault(n, {B: [] for B in Bs})
            for B in Bs:
                per_req[n][B].append(np.mean([recall_at(preds[n], r["dec"][:, s], B) for s in range(1, D)]))
    matched = np.array(matched)
    res = {n: {B: float(np.mean(v)) for B, v in d.items()} for n, d in per_req.items()}
    split = {n: {"matched": float(np.mean(np.array(d[2 * k])[matched])) if matched.any() else None,
                 "unmatched": float(np.mean(np.array(d[2 * k])[~matched])) if (~matched).any() else None}
             for n, d in per_req.items()}
    rng = np.random.default_rng(seed)
    boot = {}
    for a, b in [("CANDFIRST", "FIRSTTOK"), ("CANDFIRST", "OURS_ADMIT"), ("CANDFIRST", "CANDDEC"),
                 ("FIRSTTOK", "GLOBAL"), ("CANDFIRST", "GLOBAL")]:
        if a in per_req and b in per_req:
            d = np.array(per_req[a][2 * k]) - np.array(per_req[b][2 * k])
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            boot[f"{a}-{b}@{2 * k}"] = [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return res, split, boot, float(matched.mean()), int(len(matched))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="GLM-4.5-Air")
    ap.add_argument("--traces", default=os.path.join(ROOT, "outputs/moe/traces"))
    ap.add_argument("--e1dir", default=os.path.join(ROOT, "outputs/moe/e1"))
    ap.add_argument("--E", type=int, default=128)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pre = os.path.join(args.traces, args.model)
    train = load_run(f"{pre}_train_M0_cap8")
    random.Random(0).shuffle(train)                       # same build / tune split as e1_analysis.py
    n_tune = max(100, len(train) // 5)
    T = DecTable(train[n_tune:], args.E)
    k = train[0]["dec"].shape[-1]
    print(f"{args.model}: decode table over {len(T.tab)} token ids (build split {len(train) - n_tune})", flush=True)
    report = {}
    for name, src, sub in [("W1-cwq", "cwq", "cwq_M0_cap8"), ("W1-webqsp", "webqsp", "webqsp_M0_cap8"),
                           ("Wsyn-cwq", "cwq", "cwq_M0_syn_cap8"), ("Wsynr-cwq", "cwq", "cwq_M0_synr_cap8"),
                           ("W2-cwq", "cwq", "cwq_M0_cot_cap8"), ("W2c-cwq", "cwq", "cwq_M0_cot_chat_cap8")]:
        d = f"{pre}_{sub}"
        if not os.path.isdir(d) or not os.path.exists(d + ".json"):   # missing or still being collected
            continue
        reqs = load_run(d)
        attach_paths(reqs, src)
        t0 = None
        f = os.path.join(args.e1dir, f"{args.model}_{name}_pred.npz")
        if os.path.exists(f):
            z = np.load(f)
            names = [n for n in ("RPP", "BASE_ADMIT", "OURS_ADMIT", "RPP_R", "KNN_R", "KNN_T") if n in z.files]
            t0 = dict(qi={q: i for i, q in enumerate(z["questions"])}, names=names, arr={n: z[n] for n in names})
        res, split, boot, mrate, n = evaluate(reqs, T, args.E, k, t0)
        report[name] = dict(n=n, match_rate=mrate, recall=res, recall_2k_by_match=split, bootstrap=boot)
        print(f"\n== {name}: n={n} (requests with >=2 decode steps), g0 starts a candidate in {mrate:.0%}")
        print(f"   {'predictor':12s}  R@{k:<3d}  R@{2 * k:<3d}  R@{4 * k:<3d}   R@{2 * k} matched / unmatched")
        for nm in sorted(res, key=lambda x: -res[x][2 * k]):
            sp = split[nm]
            print(f"   {nm:12s}  {res[nm][k]:.3f}  {res[nm][2 * k]:.3f}  {res[nm][4 * k]:.3f}   "
                  f"{sp['matched'] if sp['matched'] is None else round(sp['matched'], 3)} / "
                  f"{sp['unmatched'] if sp['unmatched'] is None else round(sp['unmatched'], 3)}")
        for kk, (m, lo, hi) in boot.items():
            print(f"   diff {kk:26s} {m * 100:+.1f} pt  95% CI [{lo * 100:+.1f}, {hi * 100:+.1f}]")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
