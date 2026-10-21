"""Expert-queue plan Q3: accuracy when decode may only use resident experts (cache-conditional routing).

The resident set of each MoE layer = its most used experts (fraction f) in the decode of calibration requests
(Wtrain, never the test set). During decode only (prefill unchanged):
  subst  non-resident experts get -inf router logits, so the top-k is re-chosen among resident experts
  drop   the normal top-k is kept, non-resident experts get weight 0 and the rest are renormalised
Task cwq: CWQ (M0, cap 8) short answers, Hit@1 / F1. Task gsm8k: GSM8K step-by-step solutions (long decode, up to
--max_new tokens), exact-match accuracy of the final number; calibration on GSM8K train. Reports for f in {1.0, 0.9, 0.75, 0.5} x {subst, drop}, plus the share of decode
expert uses that were changed. Qwen3-MoE router (softmax, top-k, norm_topk_prob); prompts as in the H200 traces
(few-shot M0 inside the chat template, thinking off).

  CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/moe/subst_accuracy.py --model Qwen/Qwen3-30B-A3B --n 300
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_traces import (PATHS, ROOT, build_prompt, evaluate_metrics, extract_answer,  # noqa: E402
                            first_line_done)
from collect_traces_v2 import wrap_chat  # noqa: E402

STATE = dict(decode=False, mode=None, allow=None, counts=None, changed=0, total=0)


def patched_forward(self, hidden_states):
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    logits = F.linear(hidden_states, self.weight)
    li = self._layer
    if STATE["decode"] and STATE["counts"] is not None:                     # calibration: count decode uses
        idx = torch.topk(logits.float().softmax(-1), self.top_k, -1).indices
        STATE["counts"][li] += torch.bincount(idx.flatten(), minlength=logits.shape[-1]).cpu()
    probs = logits.float().softmax(-1)
    if STATE["decode"] and STATE["mode"] and STATE["allow"] is not None:
        allow = STATE["allow"][li].to(probs.device)
        top0 = torch.topk(probs, self.top_k, -1).indices
        STATE["total"] += top0.numel()
        STATE["changed"] += int((~allow[top0]).sum())
        if STATE["mode"] == "subst":
            probs = probs.masked_fill(~allow, 0.0)
            val, idx = torch.topk(probs, self.top_k, -1)
        else:
            val, idx = torch.topk(probs, self.top_k, -1)
            val = val * allow[idx]
    else:
        val, idx = torch.topk(probs, self.top_k, -1)
    if self.norm_topk_prob:
        val = val / val.sum(-1, keepdim=True).clamp_min(1e-9)
    # same contract as Qwen3MoeTopKRouter.forward: (softmaxed logits [T, E], top-k weights [T, k], top-k ids [T, k])
    return logits.float().softmax(-1), val.to(logits.dtype), idx


def patched_glm_route(self, router_logits):
    """Glm4MoeMoE.route_tokens_to_experts with the same calibration / subst / drop hooks."""
    probs = router_logits.sigmoid()
    choice = probs + self.gate.e_score_correction_bias
    E = self.n_routed_experts
    gs = choice.view(-1, self.n_group, E // self.n_group).topk(2, dim=-1)[0].sum(dim=-1)
    gidx = torch.topk(gs, k=self.topk_group, dim=-1, sorted=False)[1]
    gmask = torch.zeros_like(gs).scatter_(1, gidx, 1)
    smask = gmask.unsqueeze(-1).expand(-1, self.n_group, E // self.n_group).reshape(-1, E).bool()
    choice = choice.masked_fill(~smask, 0.0)
    li = self._layer
    idx = torch.topk(choice, k=self.top_k, dim=-1, sorted=False)[1]
    if STATE["decode"] and STATE["counts"] is not None:
        STATE["counts"][li] += torch.bincount(idx.flatten(), minlength=E).cpu()
    w = probs.gather(1, idx)
    if STATE["decode"] and STATE["mode"] and STATE["allow"] is not None:
        allow = STATE["allow"][li].to(probs.device)
        STATE["total"] += idx.numel()
        STATE["changed"] += int((~allow[idx]).sum())
        if STATE["mode"] == "subst":
            idx = torch.topk(choice.masked_fill(~allow, -1e9), k=self.top_k, dim=-1, sorted=False)[1]
            w = probs.gather(1, idx)
        else:
            w = w * allow[idx]
    if self.norm_topk_prob:
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
    return idx, w * self.routed_scaling_factor


def prompts(tok, ds_name, n, paths_jsonl=None):
    if paths_jsonl:
        rows = [json.loads(l) for l in open(paths_jsonl)][:n]
        items = [(r["question"], r["answer"], r["paths"]) for r in rows]
    else:
        from datasets import load_dataset
        full = load_dataset(ds_name, split="test")
        ds = full.select(range(min(n, len(full))))
        pdict = json.load(open(os.path.join(ROOT, "graphrag-main/precomputed_paths_v2", PATHS[ds_name])))
        items = [(ds[i]["question"], ds[i]["answer"], pdict.get(ds[i]["question"], [])) for i in range(len(ds))]
    out = []
    for q, a, paths in items:
        paths = [[tuple(t) for t in p] for p in paths]
        ids, _ = wrap_chat(tok, *build_prompt(tok, q, paths, "M0", 8))
        out.append((ids, a))
    return out


GSM_INSTR = "\nSolve it step by step, then give the final numeric answer on the last line after '####'."


def gsm_items(tok, split, n):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main")[split]
    out = []
    for i in range(min(n, len(ds))):
        msg = [{"role": "user", "content": ds[i]["question"] + GSM_INSTR}]
        txt = tok.apply_chat_template(msg, add_generation_prompt=True, enable_thinking=False, tokenize=False)
        ids = tok.encode(txt, add_special_tokens=False)
        out.append((ids, ds[i]["answer"].split("####")[-1].strip().replace(",", "")))
    return out


def gsm_score(preds, items):
    import re
    ok = 0
    for p, (_, gold) in zip(preds, items):
        tail = p.split("####")[-1] if "####" in p else p
        nums = re.findall(r"-?\d[\d,]*\.?\d*", tail)
        if nums:
            try:
                ok += abs(float(nums[-1].replace(",", "").rstrip(".")) - float(gold)) < 1e-6
            except ValueError:
                pass
    return ok / len(items), ok / len(items)


def eos_ids(model, tok):
    """All end-of-turn ids: GLM ends a turn with <|user|>, Qwen with <|im_end|> (generation_config lists them)."""
    if not hasattr(model, "_eos_ids"):
        e = model.generation_config.eos_token_id
        model._eos_ids = set(e if isinstance(e, (list, tuple)) else [e]) | {tok.eos_token_id}
    return model._eos_ids


@torch.no_grad()
def generate(model, tok, items, batch, max_new=32, first_line=True):
    preds = []
    for b0 in range(0, len(items), batch):
        chunk = items[b0:b0 + batch]
        Lm = max(len(x[0]) for x in chunk)
        inp = torch.full((len(chunk), Lm), tok.pad_token_id, dtype=torch.long)
        am = torch.zeros_like(inp)
        for i, (ids, _) in enumerate(chunk):
            inp[i, Lm - len(ids):] = torch.tensor(ids)
            am[i, Lm - len(ids):] = 1
        inp, am = inp.to(model.device), am.to(model.device)
        STATE["decode"] = False
        pos = (am.cumsum(-1) - 1).clamp(min=0)                 # left padding: explicit positions
        out = model(input_ids=inp, attention_mask=am, position_ids=pos, use_cache=True, logits_to_keep=1)
        cur = pos[:, -1]
        cache, nxt = out.past_key_values, out.logits[:, -1].argmax(-1)
        gen = [[] for _ in chunk]
        done = [False] * len(chunk)
        STATE["decode"] = True
        for step in range(max_new):
            for i in range(len(chunk)):
                if not done[i]:
                    t = int(nxt[i])
                    if t in eos_ids(model, tok):
                        done[i] = True
                    else:
                        gen[i].append(t)
                        if first_line and first_line_done(tok.decode(gen[i], skip_special_tokens=True)):
                            done[i] = True
            if all(done):
                break
            am = torch.cat([am, torch.ones((len(chunk), 1), dtype=am.dtype, device=am.device)], 1)
            cur = cur + 1
            out = model(input_ids=nxt[:, None], attention_mask=am, position_ids=cur[:, None], past_key_values=cache,
                        use_cache=True)
            cache, nxt = out.past_key_values, out.logits[:, -1].argmax(-1)
        STATE["decode"] = False
        txt = [tok.decode(g, skip_special_tokens=True).strip() for g in gen]
        preds += [x.split("\n")[0] for x in txt] if first_line else txt
    return preds


def score(preds, items):
    m = evaluate_metrics([extract_answer(p) for p in preds], [gold for _, gold in items])
    return m["hits_at_1"], m["f1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--calib", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.9, 0.75, 0.5])
    ap.add_argument("--task", default="cwq", choices=["cwq", "gsm8k"])
    ap.add_argument("--max_new", type=int, default=384)
    ap.add_argument("--max_memory", default=None, help="e.g. 0:118GiB,1:118GiB (multi-GPU models)")
    ap.add_argument("--out", default="outputs/moe/subst_accuracy.json")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    if "glm" in a.model.lower():
        from transformers.models.glm4_moe import modeling_glm4_moe as mg
        mg.Glm4MoeMoE.route_tokens_to_experts = patched_glm_route
        mm = {int(k): v for k, v in (x.split(":") for x in a.max_memory.split(","))} if a.max_memory else None
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto",
                                                     max_memory=mm).eval()
        routers = [m for m in model.modules() if isinstance(m, mg.Glm4MoeMoE)]
        E = routers[0].n_routed_experts
    else:
        from transformers.models.qwen3_moe import modeling_qwen3_moe as mq
        mq.Qwen3MoeTopKRouter.forward = patched_forward
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
        routers = [m for m in model.modules() if isinstance(m, mq.Qwen3MoeTopKRouter)]
        E = routers[0].weight.shape[0]
    for i, r in enumerate(routers):
        r._layer = i
    # calibration on Wtrain decode
    STATE["counts"] = torch.zeros(len(routers), E, dtype=torch.long)
    gsm = a.task == "gsm8k"
    gen_kw = dict(max_new=a.max_new, first_line=False) if gsm else {}
    sc = gsm_score if gsm else score
    calib = (gsm_items(tok, "train", a.calib) if gsm else
             prompts(tok, None, a.calib, os.path.join(ROOT, "outputs/exp3/train_paths.jsonl")))
    generate(model, tok, calib, a.batch, **gen_kw)
    counts = STATE["counts"]
    STATE["counts"] = None
    test = gsm_items(tok, "test", a.n) if gsm else prompts(tok, "rmanluo/RoG-cwq", a.n)
    res = []
    base = generate(model, tok, test, a.batch, **gen_kw)
    h, f = sc(base, test)
    res.append(dict(frac=1.0, mode="none", hit=h, f1=f, changed=0.0))
    print(f"full        Hit {h:.3f} F1 {f:.3f}", flush=True)
    for frac in a.fracs:
        C = int(round(frac * E))
        allow = torch.zeros(len(routers), E, dtype=torch.bool)
        allow.scatter_(1, torch.argsort(counts, 1, descending=True)[:, :C], True)
        for mode in ("subst", "drop"):
            STATE.update(mode=mode, allow=allow, changed=0, total=0)
            p = generate(model, tok, test, a.batch, **gen_kw)
            h, f = sc(p, test)
            ch = STATE["changed"] / max(STATE["total"], 1)
            same = sum(x == y for x, y in zip(p, base)) / len(p)
            res.append(dict(frac=frac, mode=mode, hit=h, f1=f, changed=ch, same_answer=same))
            print(f"f={frac:.2f} {mode:5} Hit {h:.3f} F1 {f:.3f}  decode expert uses changed {ch:.1%}  "
                  f"same answer as full {same:.0%}", flush=True)
            STATE.update(mode=None, allow=None)
    json.dump(dict(model=a.model, task=a.task, n=a.n, results=res), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
