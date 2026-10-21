# Usage: python experiments/moe/regroup_pred.py --prefix outputs/moe/traces/GLM-4.5-Air \
#            --preds outputs/moe/e1/GLM-4.5-Air_W1-cwq_pred.npz --E 128 [--out result.json]
"""Decode regrouping with PREDICTED profiles (what is known when the decode batches are formed, i.e. after
prefill). Compares to random and to the oracle (actual decode profile).
New predictor CANDDEC: candidate answer entities come from the retrieved paths, and the answer is usually
one of them; decode routing is token-ID driven at DECODE positions, so average a decode-position token table
(built on Wtrain) over the candidate entity tokens."""
import os, sys, random, numpy as np
sys.path.insert(0, os.path.expanduser("~/mbzuai-projects/graphrag-followup/experiments/moe"))
from e1_analysis import load_run, attach_paths, cand_positions, counts, norm
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--prefix", default="outputs/moe/traces/GLM-4.5-Air", help="trace dir prefix <dir>/<model>")
_ap.add_argument("--preds", default="outputs/moe/e1/GLM-4.5-Air_W1-cwq_pred.npz")
_ap.add_argument("--E", type=int, default=128)
_ap.add_argument("--out", default=None)
_ap.add_argument("--skip0", action="store_true", help="count decode steps >= 1 only (step 0 consumes g0 itself)")
_args = _ap.parse_args()
E = _args.E
T = _args.prefix
train = load_run(f"{T}_train_M0_cap8"); attach_paths(train, "train")
test = load_run(f"{T}_cwq_M0_cap8"); attach_paths(test, "cwq")
L = train[0]["pre"].shape[0]
tab, n = {}, {}
for r in train:
    D = r["dec"].shape[1]
    for s in range(D):
        t = int(r["gids"][s]); tab.setdefault(t, np.zeros((L, E)))
        tab[t][np.arange(L)[:, None], r["dec"][:, s]] += 1; n[t] = n.get(t, 0) + 1
glob_dec = norm(sum(counts(r["dec"], E) for r in train))
tab = {t: norm(v) for t, v in tab.items() if n[t] >= 3}
cover = []
def canddec(r):
    ids = r["pids"][cand_positions(r["cat"])]
    vs = [tab[int(t)] for t in ids if int(t) in tab]
    cover.append(len(vs) / max(1, len(ids)))
    return norm(np.mean(vs, 0)) if vs else glob_dec
def cand_seqs(r):
    """token sequences of the candidate entities (each run of 'ent' tokens right before a newline)."""
    cat, ids, out, cur = r["cat"], r["pids"], [], []
    for i, c in enumerate(cat):
        if c == "ent": cur.append(int(ids[i]))
        else:
            if c == "nl" and cur: out.append(cur)
            cur = []
    return out
def tokprof(ids):
    vs = [tab[t] for t in ids if t in tab]
    return norm(np.mean(vs, 0)) if vs else None
hit = []
def candfirst(r):
    g0 = int(r["gids"][0])
    m = [c for c in cand_seqs(r) if c and c[0] == g0]
    hit.append(bool(m))
    p = tokprof([g0] + (m[0] if m else []))
    return p if p is not None else canddec(r)
def answertok(r):
    p = tokprof([int(t) for t in r["gids"][:r["dec"].shape[1]]])
    return p if p is not None else glob_dec
pred = np.load(_args.preds)
qi = {q: i for i, q in enumerate(pred["questions"])}
R = [r for r in test if r["dec"].shape[1] > 0]
def unit(x): v = np.asarray(x, np.float64).ravel(); return v / (np.linalg.norm(v) + 1e-9)
prof = {"oracle (actual decode)": [unit(norm(counts(r["dec"], E))) for r in R],
        "actual prefill profile": [unit(norm(counts(r["pre"][:, [i for i, c in enumerate(r["cat"]) if c != "pre"]], E))) for r in R],
        "CANDDEC (new)": [unit(canddec(r)) for r in R],
        "CANDFIRST (new)": [unit(candfirst(r)) for r in R],
        "FIRSTTOK (no retrieval)": [unit(tokprof([int(r["gids"][0])]) if tokprof([int(r["gids"][0])]) is not None
                                         else glob_dec) for r in R],
        "ANSWERTOK (not deployable)": [unit(answertok(r)) for r in R]}
for p in ["OURS_ADMIT", "BASE_ADMIT", "RPP_R", "RELBAG", "GLOBAL"]:
    if p in pred.files:
        prof[p] = [unit(pred[p][qi[r["q"]]]) for r in R]
for _a, _b, _nm in [("OURS_ADMIT", "FIRSTTOK (no retrieval)", "OURS_ADMIT+FIRSTTOK"),
                    ("BASE_ADMIT", "FIRSTTOK (no retrieval)", "BASE_ADMIT+FIRSTTOK (no retrieval)")]:
    if _a in prof:
        prof[_nm] = [unit(x + y) for x, y in zip(prof[_a], prof[_b])]
print(f"CANDDEC: candidate tokens covered by the decode table {np.mean(cover):.0%}; CANDFIRST: first generated token starts a candidate entity in {np.mean(hit):.0%} of requests")
rng = np.random.default_rng(0)
def group(pool, B, key):
    rem = list(pool); out = []
    if key is None:
        rng.shuffle(rem); return [rem[i:i + B] for i in range(0, len(rem), B)]
    V = prof[key]
    while rem:
        seed = rem.pop(int(rng.integers(len(rem))))
        if not rem: out.append([seed]); break
        sims = np.array([V[j] @ V[seed] for j in rem])
        pick = [rem[j] for j in np.argsort(-sims)[:B - 1]]
        rem = [j for j in rem if j not in pick]; out.append([seed] + pick)
    return out
def loads(b):
    Dm = max(R[i]["dec"].shape[1] for i in b); ld = tk = 0
    for s in range(1 if _args.skip0 else 0, Dm):
        act = [i for i in b if R[i]["dec"].shape[1] > s]; tk += len(act)
        for l in range(L):
            ld += len(set(np.concatenate([R[i]["dec"][l, s] for i in act]).tolist()))
    return ld, tk
P = 128
pools = [list(range(i, i + P)) for i in range(0, len(R) - P + 1, P)]
rows = []
for B in (4, 8, 16, 32):
    res = {}
    for name in [None, "oracle (actual decode)", "RPP_R", "OURS_ADMIT", "BASE_ADMIT", "CANDDEC (new)",
                 "FIRSTTOK (no retrieval)", "CANDFIRST (new)", "BASE_ADMIT+FIRSTTOK (no retrieval)",
                 "OURS_ADMIT+FIRSTTOK", "ANSWERTOK (not deployable)"]:
        if name is not None and name not in prof:
            continue
        tot = np.array([loads(b) for _ in range(3) for pool in pools for b in group(pool, B, name)]).sum(0)
        res[name or "random"] = tot[0] / tot[1] / L
    base, orc = res["random"], res["oracle (actual decode)"]
    print(f"B={B}: random {base:.2f} expert loads/layer/token; " + "; ".join(
        f"{k} {v:.2f} ({v / base - 1:+.0%}, {(base - v) / (base - orc):.0%} of oracle gain)" for k, v in res.items() if k != "random"), flush=True)
    rows.append(dict(B=B, **res))
if _args.out:
    import json
    json.dump(dict(rows=rows, match_rate=float(np.mean(hit)) if hit else None), open(_args.out, "w"), indent=1)
