# Usage: python experiments/moe/regroup_bound.py outputs/moe/traces/<model>_<workload> <pool size>
"""Upper bound of phase-specific regrouping: partition a pool of requests into batches of size B by
(a) random, (b) similarity of actual PREFILL expert profiles, (c) similarity of actual DECODE profiles,
and measure, per batch, the experts each phase must have resident:
  prefill: per layer |union of experts used by the batch's prompt tokens| (experts to load for one shared sweep)
  decode : per step and layer |union of experts used by the batch's active tokens| (loads per step if
           nothing is cached), and decode token-uses per distinct expert (higher = better amortised)."""
import glob, json, sys, numpy as np
E = 128
d, P = sys.argv[1], int(sys.argv[2])
R = []
for f in sorted(glob.glob(d + "/q*.npz")):
    z = np.load(f); cat = list(z["categories"])
    body = [i for i, c in enumerate(cat) if c != "pre"]
    pre, dec = z["pre_idx"][:, body], z["dec_idx"]
    if dec.shape[1] == 0:
        continue
    L = pre.shape[0]
    def prof(idx):
        c = np.zeros((L, E))
        for l in range(L): np.add.at(c[l], idx[l].ravel(), 1)
        c = c / np.maximum(c.sum(-1, keepdims=True), 1)
        v = c.ravel(); return v / (np.linalg.norm(v) + 1e-9)
    R.append(dict(pre=pre, dec=dec, pp=prof(pre), dp=prof(dec)))
rng = np.random.default_rng(0)
def group(pool, B, key):
    rem = list(pool); out = []
    if key is None:
        rng.shuffle(rem); return [rem[i:i + B] for i in range(0, len(rem), B)]
    while rem:
        seed = rem.pop(int(rng.integers(len(rem))))
        if not rem: out.append([seed]); break
        sims = np.array([R[j][key] @ R[seed][key] for j in rem])
        pick = [rem[j] for j in np.argsort(-sims)[:B - 1]]
        rem = [j for j in rem if j not in pick]; out.append([seed] + pick)
    return out
def prefill_union(b):
    L = R[b[0]]["pre"].shape[0]
    return np.mean([len(set(np.concatenate([R[i]["pre"][l].ravel() for i in b]).tolist())) for l in range(L)])
def decode_stats(b):
    L = R[b[0]]["dec"].shape[0]; Dmax = max(R[i]["dec"].shape[1] for i in b)
    loads = toks = 0
    for s in range(Dmax):
        act = [i for i in b if R[i]["dec"].shape[1] > s]
        toks += len(act)
        for l in range(L):
            loads += len(set(np.concatenate([R[i]["dec"][l, s] for i in act]).tolist()))
    whole = np.mean([len(set(np.concatenate([R[i]["dec"][l].ravel() for i in b]).tolist())) for l in range(L)])
    return loads / toks / L, whole
idx = list(range(len(R)))
pools = [idx[i:i + P] for i in range(0, len(idx) - P + 1, P)]
print(f"{d.split('/')[-1]}: {len(R)} requests, pool {P}, {len(pools)} pools; mean decode len {np.mean([r['dec'].shape[1] for r in R]):.1f}")
for B in (4, 8, 16, 32):
    if B > P: continue
    row = []
    for name, key in [("random", None), ("by prefill profile", "pp"), ("by decode profile", "dp")]:
        bs = [b for pool in pools for b in group(pool, B, key)]
        pu = np.mean([prefill_union(b) for b in bs])
        ds = [decode_stats(b) for b in bs]
        row.append(f"{name}: prefill union {pu:5.1f}/128 | decode experts/layer/step per token {np.mean([x[0] for x in ds]):.2f}, request-wide decode union {np.mean([x[1] for x in ds]):5.1f}")
    print(f" B={B}:\n   " + "\n   ".join(row), flush=True)
