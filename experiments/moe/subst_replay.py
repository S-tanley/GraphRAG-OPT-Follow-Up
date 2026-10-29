"""Multi-user plan 2.1 / 2.2 (key experiment): accuracy of the simulator's use-time substitution on the real model.

The router of every MoE layer is wrapped: the native routing runs unchanged, and only when a change applies are the
top-k ids replaced and the weights recomputed from the same router scores (GLM: sigmoid, normalised, x routed
scaling; Qwen3: softmax, normalised). So the run without changes is the native model, token for token.
Changes, during decode only:
  replay:<records.npz>         the simulator's records (tsim_mu --subst_log): at decode step d, layer l of request r,
                               expert e -> substitute s (applied while e is still in the top-k and s is not); once the
                               generation leaves the recorded trajectory this is a position-wise approximation
  emp:<stats.npz>:<rate>       random, matched to the simulator (subst_record_stats.py): per (token, layer) the number
                               of changes ~ Binomial(k, p_l) with the recorded per-layer profile scaled to `rate`; which
                               of the top-k is replaced and the rank of its substitute follow the recorded distributions
  uni:<stats.npz>:<rate>       the same, but the replaced expert is uniform within the top-k (a harsher control)
Tasks: cot_trace (the prompts of a trace dir, e.g. GLM W2c CoT; answers on the "Answer:" line, Hit@1 / F1, plus
"same answer as the unchanged run" and agreement with the trace's tokens), gsm8k (thinking on / off).

  CUDA_VISIBLE_DEVICES=0,1,2 .venv/bin/python experiments/moe/subst_replay.py --model zai-org/GLM-4.5-Air \
      --task cot_trace --trace outputs/moe/traces_v4/GLM-4.5-Air_cwq_M0_cot_chat_cap8 --n 72 --eval_n 50 --batch 24 \
      --max_new 1024 --max_memory 0:90GiB,1:90GiB,2:90GiB --runs none replay:<npz> emp:<stats>:0.05 --out out.json
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
import types

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STATE = dict(decode=False, step=0, rows=None, mode=None, replay=None, rate=None, rep_p=None, sub_p=None, rng=None,
             total=0, changed=0, layer_total=None, layer_changed=None, min_free_gb=15.0)


# ---------------------------------------------------------------- changes
def plan_changes(li, idx, choice):
    """New top-k ids [N, k] for this layer, or None. idx: native top-k ids; choice: the scores the model ranks by."""
    if not STATE["decode"] or STATE["mode"] in (None, "none"):
        return None
    idx_c = idx.detach().cpu().numpy().copy()
    N, k = idx_c.shape
    STATE["total"] += N * k
    STATE["layer_total"][li] += N * k
    changed = 0
    if STATE["mode"] == "replay":
        d = STATE["step"]
        for i, rid in enumerate(STATE["rows"]):
            for e, s in STATE["replay"].get(rid, {}).get((d, li), ()):
                row = idx_c[i]
                pos = np.flatnonzero(row == e)
                if len(pos) and not (row == s).any():
                    row[pos[0]] = s
                    changed += 1
    else:
        p = float(STATE["rate"][li])
        n_ch = STATE["rng"].binomial(k, min(max(p, 0.0), 1.0), size=N)
        rows = np.flatnonzero(n_ch)
        if len(rows):
            ch = choice[torch.as_tensor(rows, device=choice.device)].detach().float().cpu().numpy()
            order = np.argsort(-ch, -1, kind="stable")
            for j, i in enumerate(rows):
                row = idx_c[i]
                in_rank = np.argsort(np.argsort(-ch[j][row], kind="stable"))      # rank of each top-k id inside it
                if STATE["mode"] == "emp":
                    w = STATE["rep_p"][in_rank]
                    w = w / w.sum() if w.sum() > 0 else None
                else:
                    w = None
                pos = STATE["rng"].choice(k, size=min(n_ch[i], k), replace=False, p=w)
                for q in pos:
                    for _ in range(8):                                             # substitute rank from the record
                        r = int(STATE["rng"].choice(len(STATE["sub_p"]), p=STATE["sub_p"]))
                        s = int(order[j][min(r, ch.shape[1] - 1)])
                        if not (row == s).any():
                            row[q] = s
                            changed += 1
                            break
    if not changed:
        return None
    STATE["changed"] += changed
    STATE["layer_changed"][li] += changed
    return torch.as_tensor(idx_c, device=idx.device, dtype=idx.dtype)


def _bind_instances(cls, name, fn, model):
    """Class-level patch, plus instance level for modules whose forward accelerate wrapped at load time (device_map
    hooks keep the original bound forward in `_old_forward`, so a class-level patch alone is never called)."""
    setattr(cls, name, fn)
    for m in model.modules():
        if isinstance(m, cls) and name == "forward" and hasattr(m, "_old_forward"):
            m._old_forward = types.MethodType(fn, m)


def install(model):
    """Wrap the routers of a loaded GLM-4.5 (transformers 5.2: Glm4MoeMoE.route_tokens_to_experts, where the gate
    module only returns logits; later versions: Glm4MoeTopkRouter.forward returns (logits, weights, ids)) or Qwen3-MoE
    model; returns the number of MoE layers."""
    mt = model.config.model_type
    if mt == "glm4_moe":
        from transformers.models.glm4_moe import modeling_glm4_moe as mg
        if not hasattr(mg.Glm4MoeMoE, "route_tokens_to_experts"):
            cls, orig = mg.Glm4MoeTopkRouter, mg.Glm4MoeTopkRouter.forward

            def fwd(self, hidden_states):
                logits, w, idx = orig(self, hidden_states)
                sc = logits.float().sigmoid()
                new = plan_changes(self._layer, idx, sc + self.e_score_correction_bias)
                if new is None:
                    return logits, w, idx
                nw = sc.gather(1, new)
                if self.norm_topk_prob:
                    nw = nw / (nw.sum(-1, keepdim=True) + 1e-20)
                return logits, (nw * self.routed_scaling_factor).to(w.dtype), new
            _bind_instances(cls, "forward", fwd, model)
        else:
            cls, orig = mg.Glm4MoeMoE, mg.Glm4MoeMoE.route_tokens_to_experts

            def route(self, router_logits):
                idx, w = orig(self, router_logits)
                sc = router_logits.float().sigmoid()
                new = plan_changes(self._layer, idx, sc + self.gate.e_score_correction_bias.float())
                if new is None:
                    return idx, w
                nw = sc.gather(1, new)
                if self.norm_topk_prob:
                    nw = nw / (nw.sum(-1, keepdim=True) + 1e-20)
                return new, (nw * self.routed_scaling_factor).to(w.dtype)
            _bind_instances(cls, "route_tokens_to_experts", route, model)
    elif mt == "qwen3_moe":
        from transformers.models.qwen3_moe import modeling_qwen3_moe as mq
        cls, orig = mq.Qwen3MoeTopKRouter, mq.Qwen3MoeTopKRouter.forward

        def qfwd(self, hidden_states):
            logits, w, idx = orig(self, hidden_states)
            probs = torch.softmax(logits, dtype=torch.float, dim=-1)
            new = plan_changes(self._layer, idx, probs)
            if new is None:
                return logits, w, idx
            nw = probs.gather(1, new)
            if self.norm_topk_prob:
                nw = nw / nw.sum(-1, keepdim=True)
            return logits, nw.to(w.dtype), new
        _bind_instances(cls, "forward", qfwd, model)
    else:
        raise ValueError(mt)
    layers = [m for m in model.modules() if isinstance(m, cls)]
    for i, m in enumerate(layers):
        m._layer = i
    return len(layers)


# ---------------------------------------------------------------- generation
def eos_ids(model, tok):
    e = model.generation_config.eos_token_id
    return set(e if isinstance(e, (list, tuple)) else [e]) | {tok.eos_token_id}


@torch.no_grad()
def generate(model, tok, prompts, rids, batch, max_new):
    """Greedy decode as in collect_traces_v2 (left padding, explicit positions, logits_to_keep=1); STATE["step"] is the
    decode-forward index (= the trace's decode token index), STATE["rows"] the request ids of the batch rows."""
    eos = eos_ids(model, tok)
    dev = next(model.parameters()).device
    outs = []
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
        STATE["decode"], STATE["rows"] = False, rids[b0:b0 + n]
        out = model(input_ids=inp, attention_mask=am, position_ids=pos, use_cache=True, logits_to_keep=1)
        cache, logits, cur = out.past_key_values, out.logits[:, -1], pos[:, -1]
        gen, done = [[] for _ in range(n)], [False] * n
        for step in range(max_new):
            nxt = logits.argmax(-1)
            for i in range(n):
                if not done[i]:
                    t = int(nxt[i])
                    if t in eos:
                        done[i] = True
                    else:
                        gen[i].append(t)
            if all(done):
                break
            am = torch.cat([am, torch.ones((n, 1), dtype=am.dtype, device=dev)], dim=1)
            cur = cur + 1
            STATE["decode"], STATE["step"] = True, step
            out = model(input_ids=nxt[:, None], attention_mask=am, position_ids=cur[:, None], past_key_values=cache,
                        use_cache=True)
            STATE["decode"] = False
            cache, logits = out.past_key_values, out.logits[:, -1]
        outs += gen
        del cache, out
        free = gpu_free()                          # shared GPUs: stop as soon as any card has too little headroom
        if free and min(free.values()) < STATE["min_free_gb"]:
            print(f"STOP: GPU headroom {free} below {STATE['min_free_gb']} GB after a batch", flush=True)
            sys.exit(3)
    return outs


def gpu_free():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        free = {int(i): (float(t) - float(u)) / 1024 for i, u, t in (l.split(",") for l in out.strip().splitlines())}
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        return {i: round(free[i], 1) for i in (map(int, vis.split(",")) if vis else free)}
    except Exception:                                                   # noqa: BLE001 (no nvidia-smi: local test)
        return {}


# ---------------------------------------------------------------- tasks
def load_task(a, tok):
    """(prompts, golds, extra) for the task."""
    if a.task == "cot_trace":
        files = sorted(glob.glob(a.trace + "/q*.npz"))[:a.n]
        zs = [np.load(f) for f in files]
        return ([z["prompt_ids"].tolist() for z in zs], [json.loads(str(z["gold"])) for z in zs],
                dict(stems=[os.path.basename(f) for f in files], trace_gen=[z["gen_ids"].tolist() for z in zs]))
    from datasets import load_dataset
    from subst_accuracy import GSM_INSTR
    ds = load_dataset("openai/gsm8k", "main")[a.split]
    prompts, golds = [], []
    for i in range(a.skip, min(a.skip + a.n, len(ds))):
        txt = tok.apply_chat_template([{"role": "user", "content": ds[i]["question"] + GSM_INSTR}], tokenize=False,
                                      add_generation_prompt=True, enable_thinking=a.thinking)
        prompts.append(tok.encode(txt, add_special_tokens=False))
        golds.append(ds[i]["answer"].split("####")[-1].strip().replace(",", ""))
    return prompts, golds, {}


def answers(a, tok, gens):
    texts = [tok.decode(g, skip_special_tokens=True) for g in gens]
    if a.task == "cot_trace":
        return [t.split("Answer:")[-1].strip().split("\n")[0] for t in texts]
    return [t.split("</think>")[-1].strip() for t in texts]


def final_answer(a, pred):
    """What 'same answer' compares: the Answer line (cot_trace) or the last number after #### (gsm8k)."""
    if a.task == "cot_trace":
        return pred
    import re
    tail = pred.split("####")[-1] if "####" in pred else pred
    nums = re.findall(r"-?\d[\d,]*\.?\d*", tail)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def score(a, preds, golds):
    if a.task == "cot_trace":
        from collect_traces import evaluate_metrics
        m = evaluate_metrics(preds, golds)
        return m["hits_at_1"], m["f1"]
    from subst_accuracy import gsm_score
    return gsm_score(preds, [(None, g) for g in golds])


def load_records(path, stems):
    z = np.load(path)
    rec, rst = z["records"], [str(s) for s in z["stems"]]
    pos = {s: i for i, s in enumerate(stems)}
    out = {}
    for rid, d, l, e, s in rec:
        r = pos.get(rst[rid])
        if r is not None:
            out.setdefault(r, {}).setdefault((int(d), int(l)), []).append((int(e), int(s)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", choices=["cot_trace", "gsm8k"], required=True)
    ap.add_argument("--trace", default=None, help="cot_trace: trace dir whose prompts are replayed")
    ap.add_argument("--split", default="test")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--n", type=int, default=50, help="requests generated (cot_trace: keep the trace's batches)")
    ap.add_argument("--eval_n", type=int, default=None, help="score only the first eval_n")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--batches", type=int, nargs="+", default=None,
                    help="run every --runs spec at each of these batch sizes, one model load (neutral-perturbation "
                         "control, plan D11.3: only floating-point differences)")
    ap.add_argument("--skip", type=int, default=0, help="gsm8k: first question index (the first ones calibrate)")
    ap.add_argument("--max_new", type=int, default=1024)
    ap.add_argument("--max_memory", default=None)
    ap.add_argument("--runs", nargs="+", default=["none"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_free_gb", type=float, default=15.0)
    ap.add_argument("--require_identity", action="store_true",
                    help="cot_trace: stop unless the first (unchanged) run reproduces the trace token for token")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    mm = ({(int(k) if k.isdigit() else k): v for k, v in (x.split(":") for x in a.max_memory.split(","))}
          if a.max_memory else None)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto" if mm else "cuda",
                                                 max_memory=mm).eval()
    devs = sorted({str(d) for d in getattr(model, "hf_device_map", {}).values()})
    print(f"loaded in {time.time() - t0:.0f}s on {devs}; GPU free {gpu_free()}", flush=True)
    L = install(model)
    STATE["min_free_gb"] = a.min_free_gb
    prompts, golds, extra = load_task(a, tok)
    E = a.eval_n or len(prompts)
    rids = list(range(len(prompts)))
    results, base_preds, base_gens = [], None, None
    for spec, batch in [(sp, b) for b in (a.batches or [a.batch]) for sp in a.runs]:
        parts = spec.split(":")
        STATE.update(mode=parts[0], replay=None, total=0, changed=0, layer_total=np.zeros(L, np.int64),
                     layer_changed=np.zeros(L, np.int64), rng=np.random.default_rng(a.seed))
        if parts[0] == "replay":
            STATE["replay"] = load_records(parts[1], extra["stems"])
        elif parts[0] in ("emp", "uni"):
            st = np.load(parts[1])
            prof = st["layer_rate"] if len(st["layer_rate"]) == L else np.ones(L)
            STATE["rate"] = prof / max(prof.mean(), 1e-12) * float(parts[2])
            rp = st["replaced_rank_hist"].astype(np.float64)
            STATE["rep_p"] = rp / rp.sum()
            sp = st["substitute_rank_hist"].astype(np.float64)
            STATE["sub_p"] = sp / sp.sum()
        ts = time.time()
        gens = generate(model, tok, prompts, rids, batch, a.max_new)
        preds = answers(a, tok, gens)
        h, f1 = score(a, preds[:E], golds[:E])
        if parts[0] != "none" and STATE["total"] == 0:
            print(f"ERROR: run {spec} never reached the router hook (patch not active)", flush=True)
            sys.exit(4)
        res = dict(run=spec, batch=batch, hit=h, f1=f1, n=E, sec=round(time.time() - ts),
                   change_rate=STATE["changed"] / max(STATE["total"], 1),
                   layer_rate=(STATE["layer_changed"] / np.maximum(STATE["layer_total"], 1)).round(5).tolist(),
                   gen_tokens=int(sum(len(g) for g in gens[:E])), gpu_free=gpu_free())
        if base_preds is None:
            base_preds, base_gens = preds, gens
            if "trace_gen" in extra:                                     # 2.1.3: unchanged run == the trace
                res["same_tokens_as_trace"] = sum(g == t for g, t in zip(gens[:E], extra["trace_gen"][:E]))
        res["same_answer_as_first_run"] = sum(final_answer(a, p) == final_answer(a, q)
                                              for p, q in zip(preds[:E], base_preds[:E])) / E
        res["answers"] = [final_answer(a, p) for p in preds[:E]]
        fd = []
        for g, b in zip(gens[:E], base_gens[:E]):                         # first token that differs
            m = min(len(g), len(b))
            j = next((i for i in range(m) if g[i] != b[i]), m if len(g) != len(b) else None)
            fd.append(j)
        res["diverged_requests"] = sum(x is not None for x in fd)
        res["median_first_divergence"] = (float(np.median([x for x in fd if x is not None]))
                                          if any(x is not None for x in fd) else None)
        results.append(res)
        print(f"{spec:52} b={batch:<4} hit {h:.3f} f1 {f1:.3f}  changed {res['change_rate']:.2%}  same answer "
              f"{res['same_answer_as_first_run']:.0%}  diverged {res['diverged_requests']}/{E}  {res['sec']}s  "
              f"GPU free {res['gpu_free']}", flush=True)
        json.dump(dict(model=a.model, task=a.task, args=vars(a), results=results), open(a.out, "w"), indent=1)
        if a.require_identity and "same_tokens_as_trace" in res and res["same_tokens_as_trace"] < E:
            print(f"STOP: the unchanged run reproduces only {res['same_tokens_as_trace']}/{E} trace requests", flush=True)
            sys.exit(2)
        if res["gpu_free"] and min(res["gpu_free"].values()) < a.min_free_gb:
            print("STOP: GPU headroom below the limit", flush=True)
            break
    print("ALL RUNS DONE", flush=True)


if __name__ == "__main__":
    main()
