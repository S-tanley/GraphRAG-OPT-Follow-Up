"""Multi-user plan 4.1.2: measure use-time substitution with experts offloaded to pinned host memory (hf_offload.py),
one H200 over PCIe: B requests decode together (static batch, left padding, greedy), per decode step time, loads and
stall, change rate, accuracy. Pools are sized and filled from calibration traces' decode routes (tsim_mu.layer_caps);
each configuration starts from the same warm cache.

Timing runs need a GPU nobody else uses (user's rule: idle for 15-30 min); functional runs may share one.

  CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/moe/hf_offload_run.py --model Qwen/Qwen3-30B-A3B --task gsm8k \\
      --thinking --calib outputs/moe/traces_v4/Qwen3-30B-A3B_gsm8k_think --skip 50 --cache 0.5 --batch 16 --n 64 \\
      --subst none rate:0.05 rate:0.1 --out outputs/moe/hf_offload/qwen3_f0.5_b16.json
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "moe_edge"))
import hf_offload as ho  # noqa: E402
from subst_replay import answers, eos_ids, final_answer, load_task, score  # noqa: E402
from tsim_mu import layer_caps  # noqa: E402


def calib_counts(trace_dir, L, E, skip_stems=()):
    f = np.zeros((L, E))
    n = 0
    for fn in sorted(glob.glob(trace_dir + "/q*.npz")):
        if os.path.basename(fn) in skip_stems:
            continue
        d = np.load(fn)["dec_idx"]
        for l in range(L):
            f[l] += np.bincount(d[l].ravel().astype(np.int64), minlength=E)
        n += 1
    return f, n


def gpu_free_gb(dev):
    free, _total = torch.cuda.mem_get_info(dev)
    return free / 2 ** 30


def run(model, tok, eng, prompts, batch, max_new, dev):
    eos = eos_ids(model, tok)
    outs, steps = [], []
    zero = dict(loads=0, stall_s=0.0, staged=0)
    s0 = dict(eng.stats) if eng is not None else zero
    t_pre = 0.0
    for b0 in range(0, len(prompts), batch):
        chunk = prompts[b0:b0 + batch]
        n, Lm = len(chunk), max(len(p) for p in chunk)
        inp = torch.full((n, Lm), tok.pad_token_id, dtype=torch.long)
        am = torch.zeros((n, Lm), dtype=torch.long)
        for i, p in enumerate(chunk):
            inp[i, Lm - len(p):] = torch.tensor(p)
            am[i, Lm - len(p):] = 1
        inp, am = inp.to(dev), am.to(dev)
        pos = (am.cumsum(-1) - 1).clamp(min=0)
        torch.cuda.synchronize(dev)
        t = time.perf_counter()
        with torch.no_grad():
            out = model(input_ids=inp, attention_mask=am, position_ids=pos, use_cache=True, logits_to_keep=1)
        torch.cuda.synchronize(dev)
        t_pre += time.perf_counter() - t
        cache, logits, cur = out.past_key_values, out.logits[:, -1], pos[:, -1]
        gen, done = [[] for _ in range(n)], [False] * n
        for _ in range(max_new):
            nxt = logits.argmax(-1)
            for i in range(n):
                if not done[i]:
                    tkn = int(nxt[i])
                    if tkn in eos:
                        done[i] = True
                    else:
                        gen[i].append(tkn)
            if all(done):
                break
            active = n - sum(done)
            am = torch.cat([am, torch.ones((n, 1), dtype=am.dtype, device=dev)], dim=1)
            cur = cur + 1
            t = time.perf_counter()
            with torch.no_grad():
                out = model(input_ids=nxt[:, None], attention_mask=am, position_ids=cur[:, None],
                            past_key_values=cache, use_cache=True)
            torch.cuda.synchronize(dev)
            steps.append((time.perf_counter() - t, active, n))
            cache, logits = out.past_key_values, out.logits[:, -1]
        outs += gen
        del cache, out
        torch.cuda.empty_cache()
    s1 = dict(eng.stats) if eng is not None else zero
    return outs, steps, t_pre, {k: s1[k] - s0[k] for k in ("loads", "stall_s", "staged")}


def ab_steps(model, eng, prompts, steps, dev, pad_id):
    """One batch, a fixed number of decode steps (no stopping) -> (mean step s, reads per step)."""
    n, Lm = len(prompts), max(len(p) for p in prompts)
    inp = torch.full((n, Lm), pad_id, dtype=torch.long)
    am = torch.zeros((n, Lm), dtype=torch.long)
    for i, p in enumerate(prompts):
        inp[i, Lm - len(p):] = torch.tensor(p)
        am[i, Lm - len(p):] = 1
    inp, am = inp.to(dev), am.to(dev)
    pos = (am.cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        out = model(input_ids=inp, attention_mask=am, position_ids=pos, use_cache=True, logits_to_keep=1)
    cache, logits, cur = out.past_key_values, out.logits[:, -1], pos[:, -1]
    l0 = eng.stats["loads"] if eng is not None else 0
    torch.cuda.synchronize(dev)
    t = time.perf_counter()
    for _ in range(steps):
        nxt = logits.argmax(-1)
        am = torch.cat([am, torch.ones((n, 1), dtype=am.dtype, device=dev)], dim=1)
        cur = cur + 1
        with torch.no_grad():
            out = model(input_ids=nxt[:, None], attention_mask=am, position_ids=cur[:, None], past_key_values=cache,
                        use_cache=True)
        cache, logits = out.past_key_values, out.logits[:, -1]
        logits[0, 0].item()                                   # per-step sync, as in run()
    torch.cuda.synchronize(dev)
    dt = (time.perf_counter() - t) / steps
    loads = (eng.stats["loads"] - l0) / steps if eng is not None else 0.0
    del cache, out
    torch.cuda.empty_cache()
    return dt, loads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", choices=["cot_trace", "gsm8k"], required=True)
    ap.add_argument("--trace", default=None, help="cot_trace: trace dir whose prompts are used")
    ap.add_argument("--split", default="test")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--skip", type=int, default=0, help="gsm8k: start at this question (the first ones calibrate)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_new", type=int, default=1024)
    ap.add_argument("--calib", required=True, help="trace dir whose decode routes size and fill the pools")
    ap.add_argument("--calib_skip_test", action="store_true", help="cot_trace: leave the test requests out of calib")
    ap.add_argument("--cache", type=float, required=True)
    ap.add_argument("--floor", type=int, default=16)
    ap.add_argument("--policy", default="lfu")
    ap.add_argument("--subst", nargs="+", default=["none"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_free_gb", type=float, default=15.0)
    ap.add_argument("--experts_impl", default="grouped_mm", choices=["grouped_mm", "batched_mm", "eager"])
    ap.add_argument("--resident", action="store_true", help="baseline: the whole model on the GPU, no offload")
    ap.add_argument("--fast", action="store_true", help="fast decode path (hf_offload.route_fast, plan 4.2.6)")
    ap.add_argument("--ab_rounds", type=int, default=0, help="A/B timing: rounds over {old, fast} x subst configs")
    ap.add_argument("--ab_steps", type=int, default=256)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.eval_n = None
    dev = "cuda:0"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 device_map=dev if a.resident else "cpu").eval()
    blocks = ho.moe_blocks(model)
    L, E = len(blocks), blocks[0].experts.gate_up_proj.shape[0]
    n = a.n
    a.n = a.skip + n
    prompts, golds, extra = load_task(a, tok)
    a.n = n
    prompts, golds = prompts[a.skip:a.skip + n], golds[a.skip:a.skip + n]
    skip = set(extra.get("stems", [])[a.skip:a.skip + n]) if a.calib_skip_test else set()
    counts, ncal = calib_counts(a.calib, L, E, skip)
    C = int(a.cache * L * E)
    caps = layer_caps(counts, C, L, E, a.floor)
    eng = None if a.resident else ho.HFOffload.attach(model, caps, dev, a.policy)
    if eng is not None:
        eng.fast = a.fast
    model.config._experts_implementation = a.experts_impl
    print(f"loaded {a.model} in {time.time() - t0:.0f}s; L={L} E={E}; "
          + ("resident (no offload)" if a.resident else f"cache {a.cache:.0%} = {C} experts (slots/layer "
             f"{caps.min()}..{caps.max()}, calibration {ncal} requests)")
          + f"; experts {model.config._experts_implementation}; GPU free {gpu_free_gb(dev):.1f} GB", flush=True)
    if a.resident:
        a.subst = ["none"]
    if a.ab_rounds:                                       # interleaved timing: external load hits every config alike
        cfgs = [("resident", "none")] if a.resident else [(f, s_) for s_ in a.subst for f in ("old", "fast")]
        res = {c: [] for c in cfgs}
        for rnd in range(a.ab_rounds):
            for fast, spec in cfgs:
                if eng is not None:
                    eng.fast = fast == "fast"
                    eng.warm(counts)
                    eng.subst = ho.Subst(spec, L, E, a.seed)
                dt, ld = ab_steps(model, eng, prompts[:a.batch], a.ab_steps, dev, tok.pad_token_id)
                res[(fast, spec)].append((dt, ld))
                print(f"  round {rnd} {fast:8} {spec:9} step {1e3 * dt:6.1f} ms  reads/step {ld:6.1f}", flush=True)
        summ = {f"{f} {s_}": dict(step_ms_median=float(1e3 * np.median([x[0] for x in v])),
                                  step_ms_all=[round(1e3 * x[0], 2) for x in v],
                                  reads_per_step=float(np.mean([x[1] for x in v]))) for (f, s_), v in res.items()}
        for k_, v in summ.items():
            print(f"[{k_}] B={a.batch} median step {v['step_ms_median']:.1f} ms  ({v['step_ms_all']})  reads/step "
                  f"{v['reads_per_step']:.1f}", flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(dict(args=vars(a), ab=summ), open(a.out, "w"), indent=1)
        return
    if gpu_free_gb(dev) < a.min_free_gb:
        print("STOP: GPU headroom below the limit", flush=True)
        sys.exit(3)
    runs, first = [], None
    for spec in a.subst:
        sub = ho.Subst(spec, L, E, a.seed)
        if eng is not None:
            eng.warm(counts)
            eng.subst = sub
        tc = time.time()
        gens, steps, t_pre, st = run(model, tok, eng, prompts, a.batch, a.max_new, dev)
        preds = answers(a, tok, gens)
        hit, f1 = score(a, preds, golds)
        fin = [final_answer(a, p) for p in preds]
        first = first or fin
        dt = np.array([s[0] for s in steps])
        act = np.array([s[1] for s in steps])
        ntok = sum(len(g) for g in gens)
        full = np.array([s[1] == s[2] for s in steps])
        summ = dict(subst=spec, batch=a.batch, hit=hit, f1=f1, n=len(gens), wall_s=time.time() - tc, prefill_s=t_pre,
                    decode_s=float(dt.sum()), decode_tok_s=float(act.sum() / dt.sum()),
                    step_ms_mean=float(1e3 * dt.mean()), step_ms_full_batch=float(1e3 * dt[full].mean()) if full.any()
                    else None, loads_per_tok=st["loads"] / max(act.sum(), 1),
                    loads_per_step=st["loads"] / max(len(dt), 1), stall_ms_per_step=1e3 * st["stall_s"] / max(len(dt), 1),
                    gen_tokens=ntok, same_answer_as_first=float(np.mean([x == y for x, y in zip(fin, first)])),
                    gpu_free_gb=gpu_free_gb(dev), resident=a.resident, experts_impl=a.experts_impl, fast=a.fast,
                    **sub.summary())
        print(f"[{spec}] B={a.batch} decode {summ['decode_tok_s']:.1f} tok/s  step {summ['step_ms_mean']:.0f} ms  "
              f"loads/step {summ['loads_per_step']:.1f}  stall {summ['stall_ms_per_step']:.0f} ms/step  changed "
              f"{summ['change_rate']:.2%} (late {summ['late_rate']:.2%})  acc {hit:.3f}  same "
              f"{summ['same_answer_as_first']:.0%}  {summ['wall_s']:.0f}s  GPU free {summ['gpu_free_gb']:.1f} GB",
              flush=True)
        runs.append(dict(summary=summ, answers=fin))
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(dict(args=vars(a), L=L, E=E, C=C, caps=caps.tolist(), calib_requests=ncal, runs=runs),
                  open(a.out, "w"), indent=1)
        if summ["gpu_free_gb"] < a.min_free_gb:
            print("STOP: GPU headroom below the limit", flush=True)
            sys.exit(3)


if __name__ == "__main__":
    main()
