"""Multi-user plan 4.1.1 (concurrent requests): subst_offload.py with B requests decoding together (mlx-lm
BatchGenerator, continuous batching: a finished request is replaced by the next prompt; prompts are prefilled one at a
time between decode steps). Same cache (per-layer pools from calibration counts, warm start, LFU) and substitution
rule; the patched MoE blocks substitute row by row in each decode step. --prefetch as subst_offload.py (plan 4.2):
the batch's GATE-h probabilities are summed per expert, and every model call (prefill or decode step) advances the
position by L.

Reported per configuration: throughput (generated tokens / wall time), the gap between consecutive tokens of each
request (p50 / p99; it includes prefills of newly admitted requests), loads per generated token (all, and without the
prefill staging reads), stall per generated token (prefill included), change rate, Hit / F1 and the share of answers
equal to the first configuration's.

  .venv-mac/bin/python experiments/moe_edge/subst_offload_batch.py --calib outputs/moe_edge/traces/GLM-4.5-Air-4bit_cwq_M0_cot_chat_cap8 \\
      --cache 0.5 --batch 4 --subst none rate:0.1 --start 32 --n 20 --out outputs/moe_edge/subst/p41b_glm_f0.5_b4
"""
import argparse
import json
import os
import sys
import time

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.generate import BatchGenerator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_traces_mlx import COT, PATHS, ROOT, build_chat_prompt, evaluate_metrics, load_ds  # noqa: E402
from offload_engine import OffloadEngine  # noqa: E402
from prefetch import Prefetcher  # noqa: E402
from subst_offload import Subst, calib_counts, install  # noqa: E402
from tsim_mu import HW, attn_cost, gmm_cost, layer_caps, parse_pref  # noqa: E402


def run_batch(model, eng, prompts, B, max_new, eos):
    gen = BatchGenerator(model, max_tokens=max_new, stop_tokens=[[t] for t in sorted(eos)],
                         completion_batch_size=B, prefill_batch_size=1)
    uids = gen.insert(prompts)
    toks = {u: [] for u in uids}
    last, gaps = {}, []
    s0 = dict(eng.stats)
    t0 = time.perf_counter()
    while resp := gen.next_generated():                 # [] once every request has finished
        now = time.perf_counter()
        for r in resp:
            if r.finish_reason != "stop":
                toks[r.uid].append(r.token)
            if r.uid in last:
                gaps.append(now - last[r.uid])
            last[r.uid] = now
    wall = time.perf_counter() - t0
    gen.close()
    s1 = dict(eng.stats)
    return [toks[u] for u in uids], wall, np.array(gaps), {k: s1[k] - s0[k] for k in ("loads", "stall_s", "staged")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/GLM-4.5-Air-4bit")
    ap.add_argument("--store", default="~/mlx_offload/GLM-4.5-Air-4bit")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--cache", type=float, required=True)
    ap.add_argument("--floor", type=int, default=16, help="minimum pool slots per layer (as subst_offload.py; a B=4 step "
                                                          "needs <= 32, more uses the staging path)")
    ap.add_argument("--policy", default="lfu")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--subst", nargs="+", default=["none"])
    ap.add_argument("--dataset", default="rmanluo/RoG-cwq")
    ap.add_argument("--start", type=int, default=32)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefetch", nargs="+", default=["none"])
    ap.add_argument("--pf_calib", default=None)
    ap.add_argument("--hw", default="mac_ssd_cal")
    ap.add_argument("--skip", nargs="*", default=[], help="prefetch/subst pairs not to run")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

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
    eng.gate_h, eng.gate_m, eng.gate_checks = hmax, 16, 200 if hmax else 0
    eng.n_forward = 0
    if hmax:
        eng.gate_w = mx.stack([b.gate.weight for b in eng.blocks])
        eng.gate_b = mx.stack([b.gate.e_score_correction_bias for b in eng.blocks])
        mx.eval(eng.gate_w, eng.gate_b)
        c = np.load(a.pf_calib)
        eng.pf_cal = (c["edges"], c["prob"], int(c["k"]))
        hw = HW[a.hw]
        eng.pf_cap = (attn_cost(hw, a.batch) + hw["tick"] + gmm_cost(hw, eng.k)) / hw["load"]   # tsim_mu.step
    cls = type(model)                                  # every model call (prefill or decode step): position += L
    orig_call = cls.__call__

    def counted(self, *args, **kw):
        eng.pos_base = eng.n_forward * eng.L
        eng.n_forward += 1
        return orig_call(self, *args, **kw)
    cls.__call__ = counted
    install()
    eos = {tok.eos_token_id} | set(getattr(wrapper, "eos_token_ids", []) or [])
    print(f"B={a.batch}; cache {a.cache:.0%} = {C} experts, slots/layer {caps.min()}..{caps.max()}; "
          f"resident {mx.get_active_memory() / 1e9:.1f} GB", flush=True)
    full = load_ds(a.dataset)
    ds = full.select(range(a.start, min(len(full), a.start + a.n)))
    pdict = json.load(open(os.path.join(ROOT, "graphrag-main/precomputed_paths_v2", PATHS[a.dataset])))
    prompts, gold = [], []
    for i in range(len(ds)):
        q = ds[i]["question"]
        paths = [[tuple(t) for t in p] for p in pdict.get(q, [])]
        prompts.append(build_chat_prompt(tok, q + " " + COT, paths, a.cap)[0])
        gold.append(ds[i]["answer"])
    runs, first = [], None
    for pf_spec, spec in [(p, s_) for p in a.prefetch for s_ in a.subst if f"{p}/{s_}" not in a.skip]:
        if eng.prefetcher is not None:
            eng.prefetcher.close()
            eng.prefetcher = None
        eng.pf_pp = pps[pf_spec]
        eng.warm(counts)
        if eng.pf_pp["H"]:
            eng.prefetcher = Prefetcher(eng)
        eng.subst = Subst(spec, eng.L, E, a.seed)
        label = spec if pf_spec == "none" else f"{pf_spec} {spec}"
        gens, wall, gaps, st = run_batch(model, eng, prompts, a.batch, a.max_new_tokens, eos)
        preds = [tok.decode(g, skip_special_tokens=True).split("Answer:")[-1].strip().split("\n")[0] for g in gens]
        m = evaluate_metrics(preds, gold)
        ntok = sum(len(g) for g in gens)
        first = first or preds
        pfs = dict(eng.prefetcher.stats) if eng.prefetcher is not None else {}
        summ = dict(subst=spec, prefetch=pf_spec, prefetch_stats=pfs, batch=a.batch, hit=m["hits_at_1"], f1=m["f1"], n=len(gens), wall_s=wall,
                    throughput_tok_s=ntok / wall, gap_ms_p50=float(1e3 * np.median(gaps)),
                    gap_ms_p99=float(1e3 * np.percentile(gaps, 99)), gap_ms_mean=float(1e3 * gaps.mean()),
                    loads_per_tok=st["loads"] / ntok, dec_loads_per_tok=(st["loads"] - st["staged"]) / ntok,
                    stall_ms_per_tok=1e3 * st["stall_s"] / ntok,
                    staged_per_tok=st["staged"] / ntok, gen_tokens=ntok,
                    same_answer_as_first=float(np.mean([x == y for x, y in zip(preds, first)])),
                    **eng.subst.summary())
        print(f"[B={a.batch} {label}] {summ['throughput_tok_s']:.1f} tok/s  gap p50 {summ['gap_ms_p50']:.0f} p99 "
              f"{summ['gap_ms_p99']:.0f} ms  loads/tok {summ['loads_per_tok']:.1f} (decode {summ['dec_loads_per_tok']:.1f})  stall {summ['stall_ms_per_tok']:.0f} "
              f"ms/tok  changed {summ['change_rate']:.2%} (late {summ['late_rate']:.2%})  hit {summ['hit']:.3f}  same "
              f"{summ['same_answer_as_first']:.0%}  {wall:.0f}s" + (f"  prefetch {pfs}" if pfs else ""), flush=True)
        runs.append(dict(summary=summ, preds=preds, gen_ids=gens))
        json.dump(dict(model=a.model, cache=a.cache, C=C, caps=caps.tolist(), floor=a.floor, batch=a.batch,
                       calib=a.calib, start=a.start, n=a.n, pf_calib=a.pf_calib, hw=a.hw, runs=runs),
                  open(a.out + ".json", "w"))
    eng.subst = None
    if eng.prefetcher is not None:
        eng.prefetcher.close()


if __name__ == "__main__":
    main()
