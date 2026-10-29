"""Multi-user plan 4.1.1: use-time substitution (cache-aware routing) on the Mac offload engine; measures TPOT, loads
and accuracy, and saves traces for the simulator check (4.1.3, tsim_mu --mode seq on the same routes).

Rule (as tsim_mu.Subst): in decode, an expert of a token's top-k that is late -- not in its layer's pool, or still
being read -- is replaced, least routing weight first and within the budget, by the pool expert with the highest
router selection score (GLM: sigmoid + correction bias; Qwen3: softmax) that the token does not already use. The
weights are recomputed from the router's scores of the new set (as subst_replay.py). A replaced expert is not loaded.
Budget: none | rate:<rho> (running share of changed uses, random rounding) | tok:<b> (per token and layer).
Prefill is never changed. Cache: per-layer pools sized from calibration counts (tsim_mu.layer_caps with a floor per
layer), filled with them before each configuration (eng.warm), LFU.

Each request's trace (q*.npz, tsim_mu.load_reqs format): pre_idx [L, P, k] (prefill routes), dec_idx [L, D, k] (the
router's own top-k, before substitution), dec_exec (what ran), dec_score fp16 (selection scores minus a per-layer
constant: the order within a layer is kept), dec_wt (the routing weights of dec_idx); with --gate_h H also
dec_gate_idx / dec_gate_score [H, L, D, M] (as collect_traces_v2: the top-M selection scores of layer l+h's router on
layer l's MoE input, -1 / 0 beyond the last layer), computed on the fly (GLM only; plan 4.2).
Prefetch (plan 4.2, --prefetch ah<H>:GATEs:m<M>:p<tau0>[:k<kappa>][:kl<kappa>][:hl<d>]): at each decode layer l the
calibrated GATE-h probabilities (gate_calib.py) of the top-M experts of layers l+1..l+H, summed over the batch, are
compared with tau = tau0 + kappa x load + kappa_log x log2(1 + load) (load = queued reads / experts readable per
layer, from the hardware profile); the passing experts go to prefetch.Prefetcher for target position l+h (as
tsim_mu.prefetch_adv). An expert replaced by substitution has its queued read cancelled.

  .venv-mac/bin/python experiments/moe_edge/subst_offload.py --model mlx-community/GLM-4.5-Air-4bit \\
      --store ~/mlx_offload/GLM-4.5-Air-4bit --calib outputs/moe_edge/traces/GLM-4.5-Air-4bit_cwq_M0_cot_chat_cap8 \\
      --cache 0.5 --subst none rate:0.1 --start 32 --n 20 --out outputs/moe_edge/subst/glm_f0.5
"""
import argparse
import glob
import json
import os
import sys
import time

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models import glm4_moe, qwen3_moe
from mlx_lm.models.cache import make_prompt_cache

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from collect_traces_mlx import COT, PATHS, ROOT, build_chat_prompt, evaluate_metrics, load_ds  # noqa: E402
from gate_calib import gate_probs, ranked  # noqa: E402
from offload_engine import OffloadEngine  # noqa: E402
from prefetch import Prefetcher  # noqa: E402
from tsim_mu import HW, attn_cost, gmm_cost, layer_caps, parse_pref  # noqa: E402


class Subst:
    def __init__(self, spec, L, E, seed=0):
        self.spec, self.b, self.rate = spec, None, None
        parts = [] if spec == "none" else spec.split(":")
        i = 0
        while i < len(parts):
            if parts[i] == "tok":
                self.b = int(parts[i + 1])
            elif parts[i] == "rate":
                self.rate = float(parts[i + 1])
            else:
                raise ValueError(f"subst spec {spec}")
            i += 2
        self.rng = np.random.default_rng(seed)
        self.uses = self.changed = self.late = 0
        self.layer_uses, self.layer_changed = np.zeros(L, np.int64), np.zeros(L, np.int64)

    def apply(self, eng, j, inds, w, sel):
        """inds [N, k] the router's top-k, w [N, k] their weights, sel [N, E] selection scores -> ids to run."""
        N, k = inds.shape
        ready = eng.ready_mask(j)
        out = inds.copy()
        for n in range(N):
            self.uses += k
            self.layer_uses[j] += k
            late = ~ready[inds[n]]
            if not late.any():
                continue
            idx = np.flatnonzero(late)
            self.late += len(idx)
            idx = idx[np.argsort(w[n, idx], kind="stable")]                # least important first
            m = len(idx) if self.spec != "none" else 0
            if self.b is not None:
                m = min(m, self.b)
            if self.rate is not None:
                m = min(m, max(0, int(self.rate * self.uses + self.rng.random()) - self.changed))
            if m <= 0:
                continue
            ok = ready.copy()
            ok[inds[n]] = False
            cand = [int(c) for c in np.argsort(-sel[n], kind="stable") if ok[c]]
            pf = getattr(eng, "prefetcher", None)
            for jj, s in zip(idx[:m], cand):
                if pf is not None:                     # a queued read of the replaced expert is cancelled
                    pf.cancel(j, int(out[n, jj]))
                out[n, jj] = s
                self.changed += 1
                self.layer_changed[j] += 1
        return out

    def summary(self):
        return dict(change_rate=self.changed / max(self.uses, 1), late_rate=self.late / max(self.uses, 1),
                    layer_rate=(self.layer_changed / np.maximum(self.layer_uses, 1)).round(5).tolist())


# ---------------------------------------------------------------- patched MoE blocks (decode only)
_ORIG = dict(glm=glm4_moe.MoE.__call__, qwen=qwen3_moe.Qwen3MoeSparseMoeBlock.__call__)


def gate_ahead(eng, j, x):
    """GATE-h of layer j's MoE input: the selection scores of layers j+1 .. j+H in one batched product -> (ids, scores)
    [N, H', M] mx arrays (H' = layers left), or None."""
    H = min(eng.gate_h, eng.L - 1 - j)
    if H <= 0:
        return None
    xs = x.reshape(-1, x.shape[-1])                                    # [N, D]
    logit = mx.einsum("nd,hed->nhe", xs, eng.gate_w[j + 1:j + 1 + H])   # [N, H, E]
    sel = mx.sigmoid(logit.astype(mx.float32)) + eng.gate_b[j + 1:j + 1 + H]
    ids = mx.argsort(-sel, axis=-1)[..., :eng.gate_m]
    if eng.gate_checks > 0:                    # h = 1 against the next layer's own router on the same input
        eng.gate_checks -= 1
        nat, _ = eng.blocks[j + 1].gate(x)
        a_ = np.sort(np.array(ids[:, 0, :eng.k]).reshape(-1, eng.k), -1)
        b_ = np.sort(np.array(nat).reshape(-1, eng.k), -1)
        assert (a_ == b_).all(), ("GATE-1 differs from the next router", j, a_, b_)
    return ids, mx.take_along_axis(sel, ids, axis=-1)


def _decide(sw, inds, scores, sel, ahead=None):
    """Run the substitution for one layer; record the step; -> new inds (mx) or None if nothing changed."""
    eng, j = sw._eng, sw._j
    mx.eval(inds, scores, sel, *(ahead or ()))
    if ahead is not None and eng.route_log is not None:
        eng.gate_log[j] = (np.array(ahead[0]).astype(np.int16), np.array(ahead[1]))
    k = inds.shape[-1]
    I = np.array(inds).reshape(-1, k).astype(np.int64)
    W = np.array(scores.astype(mx.float32)).reshape(-1, k)
    S = np.array(sel).reshape(I.shape[0], -1)
    new = eng.subst.apply(eng, j, I, W, S)
    if eng.route_log is not None:
        eng.route_log[j] = (I, new, S, W)
    if (new == I).all():
        return None
    return mx.array(new.reshape(inds.shape).astype(np.uint32))


def issue_prefetch(eng, j, gi, gs):
    """gi / gs [N, H', M] GATE-h ids and selection scores at layer j -> prefetcher (tsim_mu.prefetch_adv's rule)."""
    pp, (edges, prob, k), pf = eng.pf_pp, eng.pf_cal, eng.prefetcher
    load = pf.n_queued() / eng.pf_cap
    tau = (pp["thr"] or 0.0) + pp["kappa"] * load + pp["kappa_log"] * np.log2(1 + load)
    M = pp["M"] or k
    for h in range(1, min(gi.shape[1], pp["H"]) + 1):
        ri, rp = ranked(gi[:, h - 1].astype(np.int64), gate_probs(gs[:, h - 1], h, k, edges, prob))
        sv = np.zeros(eng.E)
        np.add.at(sv, ri[:, :M], rp[:, :M])
        sel = np.flatnonzero(sv >= tau + (pp["hl"] if h == 1 else 0.0))
        pf.lookahead(j + h, sel[np.argsort(-sv[sel], kind="stable")], eng.pos_base + j + h)


def glm_call(self, x):
    sw = self.switch_mlp
    eng = getattr(sw, "_eng", None)
    if eng is None or eng.subst is None or x.shape[1] != 1 or self.sharding_group is not None:
        return _ORIG["glm"](self, x)
    g = self.gate
    inds, scores = g(x)                                    # native selection (identical when nothing changes)
    orig = mx.sigmoid((x @ g.weight.T).astype(mx.float32))
    ahead = gate_ahead(eng, sw._j, x) if getattr(eng, "gate_h", 0) else None
    new = _decide(sw, inds, scores, orig + g.e_score_correction_bias, ahead)
    if ahead is not None and getattr(eng, "prefetcher", None) is not None:
        issue_prefetch(eng, sw._j, np.array(ahead[0]), np.array(ahead[1]))
    if new is not None:
        inds = new
        scores = mx.take_along_axis(orig, inds, axis=-1)
        if g.top_k > 1 and g.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        scores = scores * g.routed_scaling_factor
    y = sw(x, inds)
    y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
    if self.config.n_shared_experts is not None:
        y = y + self.shared_experts(x)
    return y


def qwen_call(self, x):
    sw = self.switch_mlp
    eng = getattr(sw, "_eng", None)
    if eng is None or eng.subst is None or x.shape[1] != 1 or self.sharding_group is not None:
        return _ORIG["qwen"](self, x)
    gates = mx.softmax(self.gate(x), axis=-1, precise=True)
    k = self.top_k
    inds = mx.stop_gradient(mx.argpartition(gates, kth=-k, axis=-1)[..., -k:])
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if self.norm_topk_prob:
        scores /= mx.sum(scores, axis=-1, keepdims=True)
    new = _decide(sw, inds, scores, gates)
    if new is not None:
        inds = new
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores /= mx.sum(scores, axis=-1, keepdims=True)
    y = sw(x, inds)
    return (y * scores[..., None]).sum(axis=-2).astype(y.dtype)


def install():
    glm4_moe.MoE.__call__ = glm_call
    qwen3_moe.Qwen3MoeSparseMoeBlock.__call__ = qwen_call


# ---------------------------------------------------------------- runs
def calib_counts(trace_dir, L, E):
    f = np.zeros((L, E))
    for fn in sorted(glob.glob(trace_dir + "/q*.npz")):
        d = np.load(fn)["dec_idx"]
        for l in range(L):
            f[l] += np.bincount(d[l].ravel().astype(np.int64), minlength=E)
    return f


def run_request(model, eng, ids, max_new, eos, record):
    """Greedy generation of one request -> dict (tokens, timings, per-step routes when record)."""
    cache = make_prompt_cache(model)
    eng.new_request()
    pre = {}
    if record:
        for b in eng.blocks:
            b.switch_mlp.on_call = (lambda x, ind, j=b.switch_mlp._j: pre.__setitem__(j, ind))
    s0 = dict(eng.stats)
    t0 = time.perf_counter()
    eng.pos_base = eng.n_forward * eng.L                # positions as tsim_mu: L per prefill and per decode step
    eng.n_forward += 1
    logits = model(mx.array([ids]), cache=cache)[:, -1]
    nxt = int(mx.argmax(logits, axis=-1).item())
    t_pre = time.perf_counter() - t0
    for b in eng.blocks:
        b.switch_mlp.on_call = None
    pre_idx = np.stack([np.array(pre[j]).reshape(-1, eng.k) for j in range(eng.L)]).astype(np.int16) if record else None
    s1 = dict(eng.stats)
    gen, steps, dec = [], [], []
    while nxt not in eos and len(gen) < max_new:
        gen.append(nxt)
        if len(gen) == max_new:
            break
        eng.route_log = {} if record else None
        eng.gate_log = {}
        t = time.perf_counter()
        eng.pos_base = eng.n_forward * eng.L
        eng.n_forward += 1
        logits = model(mx.array([[nxt]]), cache=cache)[:, -1]
        nxt = int(mx.argmax(logits, axis=-1).item())
        steps.append(time.perf_counter() - t)
        if record:
            dec.append((eng.route_log, eng.gate_log))
    eng.route_log = None
    s2 = dict(eng.stats)
    out = dict(gen=gen, prefill_s=t_pre, steps_s=steps, pre_idx=pre_idx, dec=dec,
               pre_loads=s1["loads"] - s0["loads"], dec_loads=s2["loads"] - s1["loads"],
               dec_stall_s=s2["stall_s"] - s1["stall_s"], dec_staged=s2["staged"] - s1["staged"])
    return out


def save_trace(fn, ids, r, L, shift, gate_h=0, gate_m=16):
    D = len(r["dec"])
    k = r["pre_idx"].shape[-1]
    extra = {}
    if D:
        rl = [st[0] for st in r["dec"]]
        I = np.stack([np.stack([rl[s][l][0][0] for s in range(D)]) for l in range(L)])       # [L, D, k]
        X = np.stack([np.stack([rl[s][l][1][0] for s in range(D)]) for l in range(L)])
        S = np.stack([np.stack([rl[s][l][2][0] for s in range(D)]) for l in range(L)])       # [L, D, E]
        W = np.stack([np.stack([rl[s][l][3][0] for s in range(D)]) for l in range(L)])
        if gate_h:
            gix = np.full((gate_h, L, D, gate_m), -1, np.int16)
            gsc = np.zeros((gate_h, L, D, gate_m), np.float32)
            for s_, (_rl, gl) in enumerate(r["dec"]):
                for l, (gi, gs) in gl.items():
                    gix[:gi.shape[1], l, s_], gsc[:gi.shape[1], l, s_] = gi[0], gs[0]
            extra = dict(dec_gate_idx=gix, dec_gate_score=gsc)
    else:
        I = X = np.zeros((L, 0, k), np.int16)
        S, W = np.zeros((L, 0, 1), np.float16), np.zeros((L, 0, k), np.float16)
    np.savez_compressed(fn, prompt_ids=np.array(ids, np.int32), gen_ids=np.array(r["gen"][:D], np.int32),
                        pre_idx=r["pre_idx"], dec_idx=I.astype(np.int16), dec_exec=X.astype(np.int16),
                        dec_score=(S - shift[:, None, None]).astype(np.float16), score_shift=shift,
                        dec_wt=W.astype(np.float16), la_dec_idx=np.zeros(0, np.int16), **extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/GLM-4.5-Air-4bit")
    ap.add_argument("--store", default="~/mlx_offload/GLM-4.5-Air-4bit")
    ap.add_argument("--calib", required=True, help="Mac trace dir whose decode routes size and fill the pools")
    ap.add_argument("--cache", type=float, required=True, help="cache size: fraction of all experts")
    ap.add_argument("--floor", type=int, default=16, help="minimum pool slots per layer")
    ap.add_argument("--policy", default="lfu")
    ap.add_argument("--subst", nargs="+", default=["none"])
    ap.add_argument("--dataset", default="rmanluo/RoG-cwq")
    ap.add_argument("--start", type=int, default=32, help="first CWQ test question (0-31: the calibration traces)")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_trace", action="store_true")
    ap.add_argument("--gate_h", type=int, default=0, help="record GATE-h for h = 1..H (GLM; 0: off)")
    ap.add_argument("--gate_m", type=int, default=16)
    ap.add_argument("--prefetch", nargs="+", default=["none"], help="none | ah<H>:GATEs:m<M>:p<tau0>[:kl<kappa>]...")
    ap.add_argument("--pf_calib", default=None, help="gate_calib.py build output (GATE-h calibration)")
    ap.add_argument("--hw", default="mac_ssd_cal", help="profile for the prefetch queue load (experts readable / layer)")
    ap.add_argument("--skip", nargs="*", default=[], help="prefetch/subst pairs not to run, e.g. none/none")
    ap.add_argument("--check_native", type=int, default=1,
                    help="first run this many requests on the native (unpatched) path; the none run must match them")
    ap.add_argument("--out", required=True, help="output prefix: <out>.json, traces in <out>_<subst>/")
    a = ap.parse_args()

    t0 = time.time()
    model, wrapper = load(a.model, lazy=True)
    tok = wrapper._tokenizer
    L0 = sum(1 for layer in model.layers if hasattr(getattr(layer, "mlp", None), "switch_mlp"))
    E = model.args.n_routed_experts if hasattr(model.args, "n_routed_experts") else model.args.num_experts
    counts = calib_counts(a.calib, L0, E)
    C = int(a.cache * L0 * E)
    caps = layer_caps(counts, C, L0, E, a.floor)
    eng = OffloadEngine.attach(model, a.store, caps, policy=a.policy)
    eng.route_log = None
    eng.gate_log = {}
    pps = {spec: parse_pref(spec) for spec in a.prefetch}
    hmax = max(pp["H"] for pp in pps.values())
    for spec, pp in pps.items():
        assert spec == "none" or (pp["pred"] == "GATEs" and pp["gamma"] is None and not pp["freeze"]), spec
    a.gate_h = max(a.gate_h, hmax)
    eng.gate_h, eng.gate_m = a.gate_h, a.gate_m
    eng.n_forward = 0
    if hmax:
        c = np.load(a.pf_calib)
        eng.pf_cal = (c["edges"], c["prob"], int(c["k"]))
        hw = HW[a.hw]
        eng.pf_cap = (attn_cost(hw, 1) + hw["tick"] + gmm_cost(hw, eng.k)) / hw["load"]    # tsim_mu.step, B = 1
    eng.gate_checks = 200 if a.gate_h else 0
    if a.gate_h:
        assert eng.mt == "glm4_moe", "GATE-h recording: GLM only"
        eng.gate_w = mx.stack([b.gate.weight for b in eng.blocks])                      # [L, E, D]
        eng.gate_b = mx.stack([b.gate.e_score_correction_bias for b in eng.blocks])     # [L, E]
        mx.eval(eng.gate_w, eng.gate_b)
    if eng.mt == "glm4_moe":
        g = eng.blocks[0].gate
        assert g.n_group == 1 and g.weight.dtype in (mx.bfloat16, mx.float16, mx.float32), (g.n_group, g.weight.dtype)
    install()
    shift = np.array([float(mx.mean(b.gate.e_score_correction_bias).item()) if eng.mt == "glm4_moe" else 0.0
                      for b in eng.blocks], np.float32)
    eos = {tok.eos_token_id} | set(getattr(wrapper, "eos_token_ids", []) or [])
    print(f"loaded {a.model} in {time.time() - t0:.0f}s; L={eng.L} E={E} k={eng.k}; cache {a.cache:.0%} = {C} experts, "
          f"slots/layer {caps.min()}..{caps.max()}; resident {mx.get_active_memory() / 1e9:.1f} GB; eos {sorted(eos)}",
          flush=True)

    full = load_ds(a.dataset)
    ds = full.select(range(a.start, min(len(full), a.start + a.n)))
    pdict = json.load(open(os.path.join(ROOT, "graphrag-main/precomputed_paths_v2", PATHS[a.dataset])))
    items = []
    for i in range(len(ds)):
        q = ds[i]["question"]
        paths = [[tuple(t) for t in p] for p in pdict.get(q, [])]
        ids, _cat = build_chat_prompt(tok, q + " " + COT, paths, a.cap)
        items.append((q, ds[i]["answer"], ids))

    native = []
    if a.check_native:
        eng.warm(counts)
        eng.subst = None
        nat_s = nat_n = 0
        for q, ans, ids in items[:a.check_native]:
            r = run_request(model, eng, ids, a.max_new_tokens, eos, False)
            native.append(r["gen"])
            nat_s += sum(r["steps_s"])
            nat_n += len(r["steps_s"])
        print(f"[native] {a.check_native} requests, TPOT {1e3 * nat_s / max(nat_n, 1):.0f} ms (no substitution code)",
              flush=True)
    runs = []
    first_answers = None
    for pf_spec, spec in [(p, s) for p in a.prefetch for s in a.subst if f"{p}/{s}" not in a.skip]:
        if eng.prefetcher is not None:
            eng.prefetcher.close()
            eng.prefetcher = None
        eng.pf_pp = pps[pf_spec]
        eng.warm(counts)
        if eng.pf_pp["H"]:
            eng.prefetcher = Prefetcher(eng)
        eng.subst = Subst(spec, eng.L, E, a.seed)
        tag = spec.replace(':', '') if pf_spec == "none" else f"{pf_spec.replace(':', '')}_{spec.replace(':', '')}"
        tdir = f"{a.out}_{tag}"
        label = spec if pf_spec == "none" else f"{pf_spec} {spec}"
        if not a.no_trace:
            os.makedirs(tdir, exist_ok=True)
        preds, gold, reqs = [], [], []
        tc = time.time()
        for qi, (q, ans, ids) in enumerate(items):
            r = run_request(model, eng, ids, a.max_new_tokens, eos, not a.no_trace)
            text = tok.decode(r["gen"], skip_special_tokens=True)
            pred = text.split("Answer:")[-1].strip().split("\n")[0]
            preds.append(pred)
            gold.append(ans)
            st = np.array(r["steps_s"])
            reqs.append(dict(q=a.start + qi, prompt_len=len(ids), tokens=len(r["gen"]), prefill_s=r["prefill_s"],
                             decode_s=float(st.sum()), tpot_ms=float(1e3 * st.mean()) if len(st) else None,
                             pre_loads=r["pre_loads"], dec_loads=r["dec_loads"], dec_stall_s=r["dec_stall_s"],
                             dec_staged=r["dec_staged"], pred=pred, gen_ids=r["gen"]))
            if not a.no_trace:
                save_trace(os.path.join(tdir, f"q{a.start + qi:05d}.npz"), ids, r, eng.L, shift, a.gate_h, a.gate_m)
            print(f"  [{label}] q{a.start + qi} {len(ids)}+{len(r['gen'])} tok  prefill {r['prefill_s']:.1f}s  "
                  f"TPOT {reqs[-1]['tpot_ms'] or 0:.0f} ms  loads/tok {r['dec_loads'] / max(len(st), 1):.1f}  "
                  f"changed {eng.subst.summary()['change_rate']:.1%}", flush=True)
        if spec == "none" and native:                  # prefetch only changes timing, never the result
            same = [x["gen_ids"] == g for x, g in zip(reqs, native)]
            print(f"[{label}] identical to the native path: {sum(same)}/{len(same)} requests", flush=True)
            assert all(same), "a no-substitution run differs from the native path"
        m = evaluate_metrics(preds, gold)
        steps = np.concatenate([np.array([x["decode_s"] / max(x["tokens"] - 1, 1)]) for x in reqs])
        dec_tok = sum(max(x["tokens"] - 1, 0) for x in reqs)
        pfs = dict(eng.prefetcher.stats) if eng.prefetcher is not None else {}
        summ = dict(subst=spec, prefetch=pf_spec, trace_dir=tdir, prefetch_stats=pfs, hit=m["hits_at_1"], f1=m["f1"], n=len(reqs), wall_s=time.time() - tc,
                    tpot_ms_mean=float(1e3 * sum(x["decode_s"] for x in reqs) / max(dec_tok, 1)),
                    tpot_ms_req_p50=float(1e3 * np.median(steps)),
                    loads_per_tok=sum(x["dec_loads"] for x in reqs) / max(dec_tok, 1),
                    stall_ms_per_tok=1e3 * sum(x["dec_stall_s"] for x in reqs) / max(dec_tok, 1),
                    staged_per_tok=sum(x["dec_staged"] for x in reqs) / max(dec_tok, 1),
                    prefill_s_mean=float(np.mean([x["prefill_s"] for x in reqs])),
                    gen_tokens=sum(x["tokens"] for x in reqs),
                    prefetch_loads_per_tok=pfs.get("loaded", 0) / max(dec_tok, 1), **eng.subst.summary())
        if first_answers is None:
            first_answers = preds
        summ["same_answer_as_first"] = float(np.mean([x == y for x, y in zip(preds, first_answers)]))
        print(f"[{label}] hit {summ['hit']:.3f} f1 {summ['f1']:.3f}  TPOT {summ['tpot_ms_mean']:.0f} ms  loads/tok "
              f"{summ['loads_per_tok']:.2f}  stall {summ['stall_ms_per_tok']:.0f} ms/tok  changed {summ['change_rate']:.2%}"
              f" (late {summ['late_rate']:.2%})  same answer {summ['same_answer_as_first']:.0%}  {summ['wall_s']:.0f}s"
              + (f"  prefetch {pfs}" if pfs else ""), flush=True)
        runs.append(dict(summary=summ, requests=reqs))
        json.dump(dict(model=a.model, store=a.store, cache=a.cache, C=C, caps=caps.tolist(), floor=a.floor,
                       policy=a.policy, calib=a.calib, start=a.start, n=a.n, max_new_tokens=a.max_new_tokens,
                       pf_calib=a.pf_calib, hw=a.hw, runs=runs), open(a.out + ".json", "w"))
    eng.subst = None
    if eng.prefetcher is not None:
        eng.prefetcher.close()


if __name__ == "__main__":
    main()
