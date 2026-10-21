"""E1 (H1): can request-level retrieval metadata predict a request's DECODE-phase expert usage
beyond token-based and online predictors?

Inputs: per-question npz traces from collect_traces.py for one model:
  <model>_train_M0_cap8          (Wtrain, used to build tables / fit predictors; split build/tune)
  <model>_{cwq,webqsp}_M0_cap8   (W1 evaluation)
  <model>_cwq_M0_syn_cap8        (Wsyn evaluation: relation names paraphrased)
  <model>_cwq_M0_cot_cap8        (W2 evaluation: long outputs)

Predictors (each gives, for every decode token and MoE layer, a score over experts):
  admission-time (known before prefill):
    GLOBAL      decode-expert popularity on Wtrain
    TOKAGG      mean token-ID routing histogram over the prompt tokens          (token-based, OpenMoE)
    RPP         per-layer ridge map TOKAGG -> decode distribution, fit on Wtrain  (ExpertFlow-style, token-based)
    SCHEMA      decode distribution of Wtrain requests with the same first-path schema (ours, backs off to GLOBAL)
    RELBAG      mean decode distribution of Wtrain requests containing each relation of the request (ours)
    CANDTOK     token-ID histogram of the candidate entity tokens (path-final entities)            (ours)
  after prefill:
    PREFILL     this request's prefill expert frequency (all prompt tokens)        (proposal Fig.4 baseline)
    CANDPRE     this request's prefill routing of the candidate entity tokens      (ours)
  online, during decode:
    TOKID       token-ID histogram of the token being decoded (known one step ahead)
    TEMPORAL    routing of the previous token at the same layer
    LOOKAHEAD   next-layer gate applied to the current layer's hidden state (one layer ahead)
  combinations: weighted sums of normalised scores, weights grid-searched on the Wtrain tune split.

Metric: recall@B = |top-B(score) ∩ actual top-k| / k per decode token and layer, averaged.
Controls: CROSS (a random other request's PREFILL), Wsyn, decode-length bins.

  python experiments/moe/e1_analysis.py --model GLM-4.5-Air --traces outputs/moe/traces --out outputs/moe/e1
"""

import argparse
import glob
import itertools
import json
import os
import random
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATHS = {"cwq": "paths_rmanluo-RoG-cwq.json", "webqsp": "paths_rmanluo-RoG-webqsp.json"}


# ----------------------------------------------------------------------------- data
def load_run(tdir):
    reqs = []
    for f in sorted(glob.glob(os.path.join(tdir, "q*.npz"))):
        z = np.load(f, allow_pickle=False)
        reqs.append(dict(pre=z["pre_idx"].astype(np.int32), dec=z["dec_idx"].astype(np.int32),
                         la_dec=z["la_dec_idx"].astype(np.int32) if z["la_dec_idx"].size else None,
                         pids=z["prompt_ids"], gids=z["gen_ids"], cat=z["categories"].astype(str),
                         q=str(z["question"])))
    return reqs


def attach_paths(reqs, source):
    """Relation schemas of each request (cap 8), from the same path source the prompt used."""
    if source == "train":
        rows = {json.loads(l)["question"]: json.loads(l)["paths"]
                for l in open(os.path.join(ROOT, "outputs/exp3/train_paths.jsonl"))}
    else:
        rows = json.load(open(os.path.join(ROOT, "graphrag-main/precomputed_paths_v2", PATHS[source])))
    for r in reqs:
        P = [p for p in rows.get(r["q"], [])[:8] if p]
        r["schemas"] = [tuple(x[1] for x in p) for p in P]
        r["topic"] = P[0][0][0] if P else None
        r["rels"] = sorted({x[1] for p in P for x in p})


def cand_positions(cat):
    """Indices of the path-final entity tokens (the run of 'ent' tokens right before each '\\n')."""
    out = []
    for i, c in enumerate(cat):
        if c == "nl":
            j = i - 1
            while j >= 0 and cat[j] == "ent":
                out.append(j)
                j -= 1
    return sorted(out)


def counts(idx, E):
    """idx [L, N, k] -> per-layer expert counts [L, E]."""
    L = idx.shape[0]
    c = np.zeros((L, E), np.float32)
    if idx.shape[1] == 0:
        return c
    for l in range(L):
        np.add.at(c[l], idx[l].ravel(), 1)
    return c


def norm(x):
    s = x.sum(-1, keepdims=True)
    return np.where(s > 0, x / np.maximum(s, 1e-9), 1.0 / x.shape[-1])


# ----------------------------------------------------------------------------- tables from Wtrain
class Tables:
    def __init__(self, train, E, min_tok=3):
        L = train[0]["pre"].shape[0]
        self.E, self.L = E, L
        self.glob_dec = np.zeros((L, E), np.float32)
        tok = defaultdict(lambda: np.zeros((L, E), np.float32))
        tokn = defaultdict(int)
        self.schema, self.rel = defaultdict(lambda: np.zeros((L, E), np.float32)), defaultdict(lambda: np.zeros((L, E), np.float32))
        for r in train:
            dc = counts(r["dec"], E)
            self.glob_dec += dc
            dn = norm(dc)
            if r["schemas"]:
                self.schema[r["schemas"][0]] += dn
            for rel in r["rels"]:
                self.rel[rel] += dn
            for ids, idx in ((r["pids"], r["pre"]), (r["gids"], r["dec"])):
                ids = np.asarray(ids[:idx.shape[1]])
                for t in np.unique(ids):
                    pos = np.nonzero(ids == t)[0]
                    sub = idx[:, pos, :]                                   # [L, n, k]
                    li = np.broadcast_to(np.arange(L)[:, None, None], sub.shape)
                    np.add.at(tok[int(t)], (li.ravel(), sub.ravel()), 1)
                    tokn[int(t)] += len(pos)
        self.tok = {t: norm(v) for t, v in tok.items() if tokn[t] >= min_tok}
        self.glob_all = norm(sum(self.tok.values()) if self.tok else np.ones((L, E)))
        self.glob_dec = norm(self.glob_dec)
        self.schema = {k: norm(v) for k, v in self.schema.items()}
        self.rel_raw = dict(self.rel)                       # unnormalised sums, for leave-one-out features
        self.rel = {k: norm(v) for k, v in self.rel.items()}
        self.rpp = None

    def tokvec(self, t):
        return self.tok.get(int(t), self.glob_all)

    def tokagg(self, ids):
        return norm(np.mean([self.tokvec(t) for t in ids], 0)) if len(ids) else self.glob_all

    def fit_knn(self, reqs, K=20):
        """Instance-based predictors: decode distributions of the K most similar build requests.
        KNN_R: cosine over idf-weighted relation bags + candidate-entity token bags (retrieval features);
        KNN_T: cosine over idf-weighted question token bags (token features, the fair counterpart)."""
        self.knn_K = K
        self.knn_Y = np.stack([norm(counts(r["dec"], self.E)).ravel() for r in reqs])
        self.knn_feats = {}
        for kind in ("R", "T"):
            bags = [self._bag(r, kind) for r in reqs]
            df = defaultdict(int)
            for b in bags:
                for t in b:
                    df[t] += 1
            idf = {t: np.log((1 + len(bags)) / (1 + c)) + 1 for t, c in df.items()}
            vocab = {t: i for i, t in enumerate(idf)}
            M = np.zeros((len(bags), len(vocab)), np.float32)
            for i, b in enumerate(bags):
                for t, c in b.items():
                    M[i, vocab[t]] = c * idf[t]
            M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
            self.knn_feats[kind] = (vocab, idf, M)

    @staticmethod
    def _bag(r, kind):
        b = defaultdict(float)
        if kind == "R":
            for rel in r["rels"]:
                b[("rel", rel)] += 1
            for i in cand_positions(r["cat"]):
                b[("cand", int(r["pids"][i]))] += 1
        else:
            for i, c in enumerate(r["cat"]):
                if c == "question":
                    b[("tok", int(r["pids"][i]))] += 1
        return b

    def knn_pred(self, r, kind):
        vocab, idf, M = self.knn_feats[kind]
        v = np.zeros(M.shape[1], np.float32)
        for t, c in self._bag(r, kind).items():
            if t in vocab:
                v[vocab[t]] = c * idf[t]
        n = np.linalg.norm(v)
        if n == 0:
            return self.glob_dec
        sim = M @ (v / n)
        top = np.argpartition(-sim, self.knn_K - 1)[:self.knn_K]
        w = np.maximum(sim[top], 0) + 1e-6
        return norm((w[:, None] * self.knn_Y[top]).sum(0).reshape(self.L, self.E))

    def fit_rpp(self, reqs, lam=1.0):
        """Per-layer ridge regression TOKAGG(prompt) -> decode distribution (ExpertFlow-style, token-based)."""
        X = np.stack([self.tokagg(r["pids"][[i for i, c in enumerate(r["cat"]) if c != "pre"]]) for r in reqs])
        Y = np.stack([norm(counts(r["dec"], self.E)) for r in reqs])   # [N, L, E]
        self.rpp = []
        for l in range(self.L):
            A = X[:, l]
            W = np.linalg.solve(A.T @ A + lam * np.eye(self.E), A.T @ Y[:, l])
            self.rpp.append(W)

    def rpp_pred(self, ids):
        a = self.tokagg(ids)
        return norm(np.clip(np.stack([a[l] @ self.rpp[l] for l in range(self.L)]), 0, None))

    def rppr_feats(self, r, loo=False):
        """Features of RPP_R: the RPP input plus the retrieval signals (relation bag, candidate tokens).
        loo=True (requests of the build split): drop the request's own decode from the relation table."""
        body = [i for i, c in enumerate(r["cat"]) if c != "pre"]
        cpos = cand_positions(r["cat"])
        if loo:
            own = norm(counts(r["dec"], self.E))
            rv = [norm(self.rel_raw[x] - own) for x in r["rels"]
                  if x in self.rel_raw and self.rel_raw[x].sum() - own.sum() > 1e-6]
        else:
            rv = [self.rel[x] for x in r["rels"] if x in self.rel]
        return np.concatenate([self.tokagg(r["pids"][body]),
                               norm(np.mean(rv, 0)) if rv else self.glob_dec,
                               self.tokagg(r["pids"][cpos]) if cpos else self.glob_all], -1)   # [L, 3E]

    def fit_rppr(self, reqs, lam=1.0):
        """Same ridge regression as RPP, with retrieval features added: the learned counterpart of RPP.
        The relation table is built from the same build split, so training features use leave-one-out
        relation vectors; lam is picked on held-out requests (the tune split) by the caller."""
        X = np.stack([self.rppr_feats(r, loo=True) for r in reqs])
        Y = np.stack([norm(counts(r["dec"], self.E)) for r in reqs])
        self.rppr = [np.linalg.solve(X[:, l].T @ X[:, l] + lam * np.eye(X.shape[-1]), X[:, l].T @ Y[:, l])
                     for l in range(self.L)]

    def rppr_pred(self, r):
        f = self.rppr_feats(r)
        return norm(np.clip(np.stack([f[l] @ self.rppr[l] for l in range(self.L)]), 0, None))


# ----------------------------------------------------------------------------- predictors
def request_scores(r, T, other=None):
    """Request-level predictors: dict name -> [L, E]."""
    E = T.E
    cpos = cand_positions(r["cat"])
    cat = r["cat"]
    body = [i for i, c in enumerate(cat) if c != "pre"]                       # drop shared few-shot preamble
    qpos = [i for i, c in enumerate(cat) if c == "question"]
    ppos = [i for i, c in enumerate(cat) if c in ("ent", "rel", "arrow", "nl", "path")]
    s = {"GLOBAL": T.glob_dec, "TOKAGG": T.tokagg(r["pids"][body]), "TOKAGG_Q": T.tokagg(r["pids"][qpos]),
         "RPP": T.rpp_pred(r["pids"][body])}
    if getattr(T, "rppr", None) is not None:
        s["RPP_R"] = T.rppr_pred(r)
    if getattr(T, "knn_feats", None):
        s["KNN_R"] = T.knn_pred(r, "R")
        s["KNN_T"] = T.knn_pred(r, "T")
    s["SCHEMA"] = T.schema.get(r["schemas"][0], T.glob_dec) if r["schemas"] else T.glob_dec
    rv = [T.rel[x] for x in r["rels"] if x in T.rel]
    s["RELBAG"] = norm(np.mean(rv, 0)) if rv else T.glob_dec
    s["CANDTOK"] = T.tokagg(r["pids"][cpos]) if cpos else T.glob_all
    s["PREFILL"] = norm(counts(r["pre"][:, body], E))
    s["PREFILL_Q"] = norm(counts(r["pre"][:, qpos], E)) if qpos else s["PREFILL"]
    s["PREFILL_PATH"] = norm(counts(r["pre"][:, ppos], E)) if ppos else s["PREFILL"]
    s["CANDPRE"] = norm(counts(r["pre"][:, cpos], E)) if cpos else s["PREFILL"]
    if other is not None:
        ob = [i for i, c in enumerate(other["cat"]) if c != "pre"]
        s["CROSS"] = norm(counts(other["pre"][:, ob], E))
    s["_cov_schema"] = float(bool(r["schemas"]) and r["schemas"][0] in T.schema)
    return s


def token_scores(r, T, s_idx):
    """Online predictors for decode token s: dict name -> [L, E]."""
    E, L = T.E, T.L
    out = {"TOKID": T.tokvec(r["gids"][s_idx])}
    prev = r["dec"][:, s_idx - 1] if s_idx > 0 else r["pre"][:, -1]
    t = np.zeros((L, E), np.float32)
    t[np.arange(L)[:, None], prev] = 1.0
    out["TEMPORAL"] = t
    if r["la_dec"] is not None:
        la = np.full((L, E), 1.0 / E, np.float32)                     # layer 0 has no lookahead
        la[1:] = 0
        la[np.arange(1, L)[:, None], r["la_dec"][:, s_idx]] = 1.0
        out["LOOKAHEAD"] = la
    return out


def recall_at(score, actual, B):
    """score [L, E], actual [L, k] -> mean over layers of |topB ∩ actual| / k (vectorised)."""
    L, E = score.shape
    top = np.argpartition(-score, B - 1, axis=-1)[:, :B]
    m = np.zeros((L, E), bool)
    m[np.arange(L)[:, None], actual] = True
    return float(np.take_along_axis(m, top, -1).sum()) / (L * actual.shape[-1])


def evaluate(reqs, T, combos, Bs, seed=0):
    rnd = random.Random(seed)
    res = defaultdict(lambda: defaultdict(list))
    per_req = defaultdict(lambda: defaultdict(list))           # name -> B -> [request-mean recall]
    by_len = defaultdict(lambda: defaultdict(list))
    cover = []
    for r in reqs:
        D = r["dec"].shape[1]
        if D == 0:
            continue
        other = rnd.choice([x for x in reqs if x is not r])
        rs = request_scores(r, T, other)
        cover.append(rs.pop("_cov_schema"))
        acc = defaultdict(lambda: defaultdict(list))
        for s in range(D):
            ts = token_scores(r, T, s)
            allsc = {**rs, **ts}
            actual = r["dec"][:, s]
            for B in Bs:
                sc_B = dict(allsc)
                for name, (parts, ws) in combos[B].items():
                    if all(p in allsc for p in parts):
                        sc_B[name] = sum(w * norm(allsc[p]) for p, w in zip(parts, ws))
                for name, sc in sc_B.items():
                    v = recall_at(sc, actual, B)
                    res[name][B].append(v)
                    acc[name][B].append(v)
                    by_len[(name, B)]["short" if D <= 8 else "long"].append(v)
        for name, d in acc.items():
            for B, v in d.items():
                per_req[name][B].append(float(np.mean(v)))
    out = {n: {B: float(np.mean(v)) for B, v in d.items()} for n, d in res.items()}
    lens = {f"{n}@{B}": {k: float(np.mean(v)) for k, v in d.items()} for (n, B), d in by_len.items()}
    # paired bootstrap over REQUESTS (tokens within a request are correlated)
    pairs = [("OURS_ADMIT", "BASE_ADMIT"), ("OURS_ONLINE", "BASE_ONLINE"), ("SCHEMA", "GLOBAL"),
             ("RELBAG", "RPP"), ("OURS_ADMIT", "GLOBAL"), ("PREFILL", "GLOBAL"), ("RPP_R", "RPP"), ("KNN_R", "KNN_T")]
    boot, rng = {}, np.random.default_rng(seed)
    for a, b in pairs:
        if a not in per_req or b not in per_req:
            continue
        for B in per_req[a]:
            d = np.array(per_req[a][B]) - np.array(per_req[b][B])
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            boot[f"{a}-{b}@{B}"] = (float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    return out, lens, float(np.mean(cover)) if cover else 0.0, boot


def tune_combos(tune, T, B):
    """Grid-search weights of combinations on the Wtrain tune split (recall@B)."""
    grid = [0.0, 0.25, 0.5, 1.0, 2.0]
    specs = {"BASE_ADMIT": ["GLOBAL", "RPP", "TOKAGG", "TOKAGG_Q", "KNN_T"],
             "OURS_ADMIT": ["GLOBAL", "RPP", "TOKAGG", "TOKAGG_Q", "KNN_T", "SCHEMA", "RELBAG", "CANDTOK", "RPP_R",
                            "KNN_R"],
             "BASE_ONLINE": ["PREFILL", "PREFILL_Q", "TOKID", "TEMPORAL", "LOOKAHEAD"],
             "OURS_ONLINE": ["PREFILL", "PREFILL_Q", "TOKID", "TEMPORAL", "LOOKAHEAD", "SCHEMA", "RELBAG", "CANDPRE"]}
    best = {}
    sample = tune[:150]
    cache = []
    for r in sample:
        if r["dec"].shape[1] == 0:
            continue
        rs = request_scores(r, T)
        rs.pop("_cov_schema")
        for s in range(min(r["dec"].shape[1], 8)):
            cache.append(({**rs, **token_scores(r, T, s)}, r["dec"][:, s]))
    for name, parts in specs.items():
        parts = [p for p in parts if all(p in c[0] for c in cache[:1])]
        bw, bv = None, -1
        # coordinate ascent over the grid (full grid is too large for 7 parts)
        w = [1.0] * len(parts)
        for _ in range(2):
            for i in range(len(parts)):
                for g in grid:
                    trial = w[:i] + [g] + w[i + 1:]
                    if sum(trial) == 0:
                        continue
                    v = np.mean([recall_at(sum(x * norm(sc[p]) for p, x in zip(parts, trial)), a, B) for sc, a in cache])
                    if v > bv:
                        bv, bw = v, trial
                w = bw
        best[name] = (parts, bw)
        print(f"  tuned {name}: {dict(zip(parts, bw))} recall@{B}={bv:.3f}", flush=True)
    return best


def schema_similarity(reqs, E, which="path"):
    """Fig.7-style: mean cosine of per-request expert profiles within vs across first-path schemas,
    plus a 2x2 split by (same schema?) x (same topic entity = head of the first path?)."""
    prof, keys, topics = [], [], []
    for r in reqs:
        if not r["schemas"]:
            continue
        if which == "path":
            sel = [i for i, c in enumerate(r["cat"]) if c in ("ent", "rel", "arrow", "nl")]
            idx = r["pre"][:, sel]
        else:
            idx = r["dec"]
        if idx.shape[1] == 0:
            continue
        p = norm(counts(idx, E)).ravel()
        prof.append(p / (np.linalg.norm(p) + 1e-9))
        keys.append(r["schemas"][0])
        topics.append(r.get("topic"))
    prof = np.stack(prof)
    S = prof @ prof.T
    same = np.array([[a == b for b in keys] for a in keys])
    st = np.array([[a == b and a is not None for b in topics] for a in topics])
    off = ~np.eye(len(keys), dtype=bool)

    def m(mask):
        mask = mask & off
        return (round(float(S[mask].mean()), 4), int(mask.sum() // 2)) if mask.any() else (float("nan"), 0)
    return dict(within=m(same)[0], across=m(~same)[0], pairs=m(same)[1],
                same_schema_diff_topic=m(same & ~st), diff_schema_same_topic=m(~same & st),
                same_schema_same_topic=m(same & st))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="trace-name prefix, e.g. GLM-4.5-Air")
    ap.add_argument("--traces", default=os.path.join(ROOT, "outputs/moe/traces"))
    ap.add_argument("--E", type=int, default=None, help="number of routed experts (default: max id + 1)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--export", action="store_true",
                    help="save request-level predicted decode distributions per workload for the simulator")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    pre = os.path.join(args.traces, args.model)
    train = load_run(f"{pre}_train_M0_cap8")
    attach_paths(train, "train")
    E = args.E or int(max(r["pre"].max() for r in train[:50]) + 1)
    k = train[0]["pre"].shape[-1]
    Bs = sorted({k, 2 * k, 4 * k, max(k, E // 8), max(k, E // 4)})
    random.Random(0).shuffle(train)
    n_tune = max(100, len(train) // 5)
    tune, build = train[:n_tune], train[n_tune:]
    print(f"{args.model}: E={E} k={k} B={Bs} train build={len(build)} tune={len(tune)}", flush=True)
    T = Tables(build, E)
    T.fit_rpp(build)
    # RPP_R: pick the ridge strength on the tune split
    best_lam, best_v = None, -1
    for lam in [1.0, 10.0, 100.0]:
        T.fit_rppr(build, lam)
        v = np.mean([recall_at(T.rppr_pred(r), r["dec"][:, s], 2 * k) for r in tune[:150]
                     for s in range(min(r["dec"].shape[1], 8))])
        print(f"  RPP_R lam={lam}: tune recall@{2 * k}={v:.3f}", flush=True)
        if v > best_v:
            best_lam, best_v = lam, v
    T.fit_rppr(build, best_lam)
    T.fit_knn(build)
    combos = {}
    for B in Bs:
        print(f" tuning combinations for B={B}", flush=True)
        combos[B] = tune_combos(tune, T, B=B)

    report = {"model": args.model, "E": E, "k": k, "Bs": Bs,
              "combos": {B: {n: dict(zip(p, w)) for n, (p, w) in c.items()} for B, c in combos.items()}}
    for name, src, sub in [("W1-cwq", "cwq", "cwq_M0_cap8"), ("W1-webqsp", "webqsp", "webqsp_M0_cap8"),
                           ("Wsyn-cwq", "cwq", "cwq_M0_syn_cap8"), ("Wsynr-cwq", "cwq", "cwq_M0_synr_cap8"),
                           ("W2-cwq", "cwq", "cwq_M0_cot_cap8"), ("W2c-cwq", "cwq", "cwq_M0_cot_chat_cap8")]:
        d = f"{pre}_{sub}"
        if not os.path.isdir(d) or not os.path.exists(d + ".json"):   # missing or still being collected
            continue
        reqs = load_run(d)
        attach_paths(reqs, src)
        res, lens, cov, boot = evaluate(reqs, T, combos, Bs)
        sim_path, sim_dec = schema_similarity(reqs, E, "path"), schema_similarity(reqs, E, "dec")
        report[name] = {"n": len(reqs), "decode_tokens": int(sum(r["dec"].shape[1] for r in reqs)),
                        "schema_coverage": cov, "recall": res, "by_len": lens, "bootstrap_diff": boot,
                        "schema_sim_path(within,across,pairs)": sim_path, "schema_sim_dec(within,across,pairs)": sim_dec}
        print(f"\n== {name}: n={len(reqs)} decode tokens={report[name]['decode_tokens']} schema coverage={cov:.2f}")
        print(f"   PATH-token profile similarity: {sim_path}")
        print(f"   DECODE profile similarity:     {sim_dec}")
        for kk, (m_, lo, hi) in sorted(boot.items()):
            print(f"   diff {kk:28s} {m_:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
        hdr = "   " + f"{'predictor':12s}" + "".join(f"  R@{B:<4d}" for B in Bs)
        print(hdr)
        for n in sorted(res, key=lambda x: -res[x][Bs[1] if len(Bs) > 1 else Bs[0]]):
            print("   " + f"{n:12s}" + "".join(f"  {res[n][B]:.3f} " for B in Bs))
        if args.export:
            names = ["GLOBAL", "TOKAGG", "RPP", "RPP_R", "KNN_R", "KNN_T", "SCHEMA", "RELBAG", "CANDTOK", "PREFILL", "PREFILL_Q",
                     "CANDPRE"]
            pred = {n: [] for n in names + ["OURS_ADMIT", "BASE_ADMIT", "ORACLE"]}
            Bc = 2 * k if 2 * k in combos else Bs[0]
            for r in reqs:
                rs = request_scores(r, T)
                rs.pop("_cov_schema")
                for n in names:
                    pred[n].append(rs[n].astype(np.float16))
                for cn in ("OURS_ADMIT", "BASE_ADMIT"):
                    parts, ws = combos[Bc][cn]
                    pred[cn].append(norm(sum(w * norm(rs[p]) for p, w in zip(parts, ws))).astype(np.float16))
                pred["ORACLE"].append(norm(counts(r["dec"], E)).astype(np.float16))
            np.savez_compressed(os.path.join(args.out, f"{args.model}_{name}_pred.npz"),
                                questions=np.array([r["q"] for r in reqs]),
                                **{n: np.stack(v) for n, v in pred.items()})
    json.dump(report, open(os.path.join(args.out, f"{args.model}.json"), "w"), indent=1)
    print("wrote", os.path.join(args.out, f"{args.model}.json"))


if __name__ == "__main__":
    main()
