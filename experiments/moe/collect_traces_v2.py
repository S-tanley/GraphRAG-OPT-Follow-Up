"""Routing traces v2 (H200, bf16, HF transformers) for the trajectory-anchor plan (P1) and the expert-queue
plan: one process loads the model once and runs a list of jobs (rule after the 2026-10-18 GPU incident:
no reloading a large model per workload, no expandable_segments, headroom on every GPU).

Per question npz, in addition to the v1 fields (prompt_ids, categories, gen_ids, pre_idx, dec_idx,
la_pre_idx, la_dec_idx, ...), for every DECODE token and MoE layer l:
  dec_score   [L, D, E] fp16  the router's selection scores (softmax / sigmoid+bias, as the model ranks)
  dec_proj    [L, D, P] fp16  the MoE input hidden state times a fixed random matrix (P=64, seed 0)
  dec_gate_h  [H, L, D, k]    GATE-h (HOBBIT-style): layer l+h's router applied to layer l's MoE input,
                              h = 1..H (entries for l+h beyond the last layer are -1)
With --gate_topm M (traces_v4), also the GATE-h scores, so GATE can be thresholded like the anchor predictors:
  dec_gate_idx   [H, L, D, M] int16  top-M experts of layer l+h's selection scores on layer l's MoE input
  dec_gate_score [H, L, D, M] fp32   those scores (softmax / sigmoid+bias, as the model ranks); -1 / 0 beyond L
                                     the first k ids are exactly dec_gate_h (the route's own tie-breaking)

  python experiments/moe/collect_traces_v2.py --model zai-org/GLM-4.5-Air --jobs jobs.json \
      --max_memory 0:118GiB,1:118GiB,cpu:400GiB --out_dir outputs/moe/traces_v2
jobs.json: [{"dataset": "rmanluo/RoG-cwq", "n": 500, "batch": 48, "max_new_tokens": 32, "cond": "M0"},
            {"paths_jsonl": "outputs/exp3/train_paths.jsonl", "n": 2000, ...},
            {"dataset": "rmanluo/RoG-cwq", "n": 200, "cot": true, "chat": true, "max_new_tokens": 1024},
            {"gsm8k": true, "split": "test", "n": 200, "batch": 50, "max_new_tokens": 1024, "thinking": false}, ...]
GSM8K jobs: the subst_accuracy.py prompt inside the chat template (thinking on / off), long-form decode until an
end-of-turn id from generation_config; accuracy = exact match of the last number after "####" (after </think>).
After every batch the free memory of each visible GPU is read with nvidia-smi; below --min_free_gb the process
stops after saving that batch (the job resumes from the saved files on the next run).
"""

import argparse
import gc
import json
import os
import random
import re
import subprocess
import sys
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_traces import (COT, EXPERT_CLASSES, PATHS, ROOT, GateBank, build_prompt, evaluate_metrics,  # noqa: E402
                            extract_answer, first_line_done, line_units, load_ds)


@torch.no_grad()
def gate_scores(G, L, h):
    """The selection scores MoE layer L ranks experts by (before top-k), for hidden states h [N, H]."""
    W, b = G._w(L, h.device)
    if G.mt in ("mixtral", "qwen3_moe"):
        return torch.nn.functional.linear(h, W.to(h.dtype)).float().softmax(-1)
    logits = torch.nn.functional.linear(h.float(), W.float())
    if G.mt == "deepseek_v2":
        return logits.softmax(-1)
    return logits.sigmoid() + (b.float() if b is not None else 0)       # glm4_moe


def build_chat_prompt(tok, q, paths, cap):
    """W2c: retrieved paths + question inside the model's chat template (template text labelled 'pre')."""
    mark = "@@USER@@"
    full = tok.apply_chat_template([{"role": "user", "content": mark}], tokenize=False, add_generation_prompt=True)
    head, tail = full.split(mark)
    ids = tok.encode(head, add_special_tokens=False) + tok.encode(
        "Use the knowledge graph paths below to answer the question.\n", add_special_tokens=False)
    cat = ["pre"] * len(ids)
    for p in paths[:cap]:
        if p:
            lid, lcat, _ = line_units(tok, p)
            ids += lid
            cat += lcat
    qi = tok.encode(f"Question: {q}", add_special_tokens=False)
    ti = tok.encode(tail, add_special_tokens=False)
    return ids + qi + ti, cat + ["question"] * len(qi) + ["pre"] * len(ti)


def wrap_chat(tok, ids, cat):
    """The raw few-shot prompt unchanged inside the chat template as the user message, thinking off."""
    mark = "@@USER@@"
    try:
        full = tok.apply_chat_template([{"role": "user", "content": mark}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        full = tok.apply_chat_template([{"role": "user", "content": mark}], tokenize=False, add_generation_prompt=True)
    head, tail = full.split(mark)
    hi, ti = tok.encode(head, add_special_tokens=False), tok.encode(tail, add_special_tokens=False)
    if tok.bos_token_id is not None and ids and ids[0] == tok.bos_token_id:
        ids, cat = ids[1:], cat[1:]
    return hi + list(ids) + ti, ["pre"] * len(hi) + list(cat) + ["pre"] * len(ti)


def gpu_free_gb():
    """Free memory (GB) of each visible GPU, from nvidia-smi (counts other users' processes too)."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    free = {int(i): (float(t) - float(u)) / 1024 for i, u, t in (l.split(",") for l in out.strip().splitlines())}
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    return {i: free[i] for i in (map(int, vis.split(",")) if vis else free)}


class Recorder2:
    def __init__(self, model, gates, H, P, hidden, top_m=0):
        self.top_m = top_m
        self.layers, self.layer_ids = [], []
        name2mod = dict(model.named_modules())
        for name, mod in model.named_modules():
            if type(mod).__name__.endswith(EXPERT_CLASSES):
                self.layers.append(name)
                self.layer_ids.append(int(re.search(r"layers\.(\d+)\.", name).group(1)))
        self.G, self.H = gates, H
        g = torch.Generator().manual_seed(0)
        self.R = (torch.randn(hidden, P, generator=g) / P ** 0.5)
        self.Rdev = {}
        self.decode = False
        for li, name in enumerate(self.layers):
            name2mod[name].register_forward_pre_hook(self._exp_hook(li), with_kwargs=True)
            name2mod[name.rsplit(".", 1)[0]].register_forward_pre_hook(self._block_hook(li), with_kwargs=True)
        self.buf = None

    def _exp_hook(self, li):
        def h(mod, args, kwargs):
            idx = kwargs.get("top_k_index", args[1] if len(args) > 1 else None)
            if self.buf is not None and idx is not None:
                self.buf[("r", li)] = idx.detach().to("cpu", torch.int16)
        return h

    def _block_hook(self, li):
        n = len(self.layer_ids)

        def h(mod, args, kwargs):
            x = args[0] if args else kwargs.get("hidden_states")
            if self.buf is None or x is None:
                return
            x = x.reshape(-1, x.shape[-1])
            routes = {}
            if li + 1 < n:
                routes[1] = self.G.route(self.layer_ids[li + 1], x)
                self.buf[("la", li)] = routes[1].to("cpu", torch.int16)
            if self.decode:
                self.buf[("sc", li)] = gate_scores(self.G, self.layer_ids[li], x).to("cpu", torch.float16)
                dev = str(x.device)
                if dev not in self.Rdev:
                    self.Rdev[dev] = self.R.to(x.device, torch.float32)
                self.buf[("pj", li)] = (x.float() @ self.Rdev[dev]).to("cpu", torch.float16)
                for hh in range(2, self.H + 1):
                    if li + hh < n:
                        routes[hh] = self.G.route(self.layer_ids[li + hh], x)
                        self.buf[("gh", hh, li)] = routes[hh].to("cpu", torch.int16)
                if self.top_m and routes:
                    # first k = the route's own top-k (bf16 router logits tie often, and topk(16) may break a
                    # tie differently from topk(k)), then the next M-k by score; fp32 (GLM's sigmoid + bias ~ 20
                    # would lose its differences in fp16). One [N, h', M] tensor per layer, not one per h.
                    vs, ixs = [], []
                    for hh in sorted(routes):
                        s = gate_scores(self.G, self.layer_ids[li + hh], x)
                        ik = routes[hh].long()
                        vr, ir = s.scatter(-1, ik, float("-inf")).topk(self.top_m - ik.shape[-1], dim=-1)
                        vs.append(torch.cat([s.gather(-1, ik), vr], -1))
                        ixs.append(torch.cat([ik, ir], -1))
                    self.buf[("gs", li)] = torch.stack(vs, 1).to("cpu", torch.float32)
                    self.buf[("gi", li)] = torch.stack(ixs, 1).to("cpu", torch.int16)
        return h

    def start(self, decode):
        self.buf, self.decode = {}, decode

    def take(self):
        b, self.buf = self.buf, None
        return b


def run_job(job, model, tok, rec, args, eos, dev):
    # drop the previous job's cached blocks: a long-decode job after a batch-48 job otherwise sits ~20 GB higher
    gc.collect()
    torch.cuda.empty_cache()
    free = gpu_free_gb()
    print(f"job start {job}; GPU free " + ", ".join(f"{g}: {f:.0f} GB" for g, f in free.items()), flush=True)
    if min(free.values()) < args.min_free_gb:
        print(f"STOP: GPU headroom below {args.min_free_gb} GB before the job", flush=True)
        sys.exit(3)
    cap, B = job.get("cap", 8), job.get("batch", 48)
    gsm, think = job.get("gsm8k", False), job.get("thinking", False)
    if gsm:
        from datasets import load_dataset
        from subst_accuracy import GSM_INSTR, gsm_score
        split, s0 = job.get("split", "test"), job.get("start", 0)
        ds = load_dataset("openai/gsm8k", "main")[split]
        items = [(ds[i]["question"], ds[i]["answer"].split("####")[-1].strip().replace(",", ""), None)
                 for i in range(s0, min(len(ds), s0 + job["n"]))]
    elif job.get("paths_jsonl"):
        rows = [json.loads(l) for l in open(job["paths_jsonl"])][job.get("start", 0):job.get("start", 0) + job["n"]]
        items, dname = [(r["question"], r["answer"], r["paths"]) for r in rows], "train"
    else:
        full = load_ds(job["dataset"])
        ds = full.select(range(job.get("start", 0), min(len(full), job.get("start", 0) + job["n"])))
        pdict = json.load(open(os.path.join(ROOT, "graphrag-main/precomputed_paths_v2", PATHS[job["dataset"]])))
        items = [(ds[i]["question"], ds[i]["answer"], pdict.get(ds[i]["question"], [])) for i in range(len(ds))]
        dname = job["dataset"].split("-")[-1]
    rel_map = json.load(open(job["rel_map"])) if job.get("rel_map") else None
    rand_syn = bool(rel_map) and any(isinstance(v, list) for v in rel_map.values())
    cot, chat, cf = job.get("cot", False), job.get("chat", False), job.get("chat_fewshot", False)
    if gsm:
        tag = f"{args.name}_gsm8k{'' if split == 'test' else '_' + split}{'_think' if think else ''}"
    else:
        tag = (f"{args.name}_{dname}_{job.get('cond', 'M0')}{'_cot' if cot else ''}{'_chat' if chat else ''}"
               f"{('_synr' if rand_syn else '_syn') if rel_map else ''}_cap{cap}")
    if os.path.exists(os.path.join(args.out_dir, tag + ".json")):
        print(f"[{tag}] done, skipped", flush=True)
        return
    tdir = os.path.join(args.out_dir, tag)
    os.makedirs(tdir, exist_ok=True)
    L, H = len(rec.layers), rec.H
    preds, gold, timing = [], [], []
    for b0 in range(0, len(items), B):
        chunk = items[b0:b0 + B]
        qfiles = [os.path.join(tdir, f"q{b0 + i:05d}.npz") for i in range(len(chunk))]
        if all(os.path.exists(f) for f in qfiles):
            for f in qfiles:
                z = np.load(f)
                preds.append(str(z["pred"]))
                gold.append(json.loads(str(z["gold"])))
            continue
        prompts, cats = [], []
        for q, _a, paths in chunk:
            if gsm:
                txt = tok.apply_chat_template([{"role": "user", "content": q + GSM_INSTR}], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=think)
                ids = tok.encode(txt, add_special_tokens=False)
                prompts.append(ids)
                cats.append(["question"] * len(ids))
                continue
            paths = [[tuple(t) for t in p] for p in paths]
            if rel_map:
                pick = ({r: random.Random(q).choice(v) if isinstance(v, list) else v for r, v in rel_map.items()}
                        if rand_syn else rel_map)
                paths = [[(h, pick.get(r, r), t) for h, r, t in p] for p in paths]
            qq = q + ((" " + COT) if cot else "")
            if chat:
                ids, cat = build_chat_prompt(tok, qq, paths, cap)
            else:
                ids, cat = build_prompt(tok, qq, paths, job.get("cond", "M0"), cap)
                if cf:
                    ids, cat = wrap_chat(tok, ids, cat)
            prompts.append(ids)
            cats.append(cat)
        n, Lmax = len(prompts), max(len(p) for p in prompts)
        inp = torch.full((n, Lmax), tok.pad_token_id, dtype=torch.long)
        am = torch.zeros((n, Lmax), dtype=torch.long)
        for i, p in enumerate(prompts):
            inp[i, Lmax - len(p):] = torch.tensor(p)
            am[i, Lmax - len(p):] = 1
        inp, am = inp.to(dev), am.to(dev)
        pos = (am.cumsum(-1) - 1).clamp(min=0)
        gen, done = [[] for _ in range(n)], [False] * n
        dec = [[] for _ in range(n)]
        ts = time.time()
        with torch.no_grad():
            rec.start(decode=False)
            # logits_to_keep=1: full prompt logits are [n, Lmax, vocab] (~8 GB at batch 48), only the last is used
            out = model(input_ids=inp, attention_mask=am, position_ids=pos, use_cache=True, logits_to_keep=1)
            pre = rec.take()
            cache, logits, cur = out.past_key_values, out.logits[:, -1], pos[:, -1]
            for step in range(job.get("max_new_tokens", 32)):
                nxt = logits.argmax(-1)
                appended = [False] * n
                for i in range(n):
                    if not done[i]:
                        t = int(nxt[i])
                        if t in eos:
                            done[i] = True
                        else:
                            gen[i].append(t)
                            appended[i] = True
                            if not (cot or gsm) and first_line_done(tok.decode(gen[i], skip_special_tokens=True)):
                                done[i] = True
                if all(done):
                    break
                am = torch.cat([am, torch.ones((n, 1), dtype=am.dtype, device=dev)], dim=1)
                cur = cur + 1
                rec.start(decode=True)
                out = model(input_ids=nxt[:, None], attention_mask=am, position_ids=cur[:, None],
                            past_key_values=cache, use_cache=True)
                b = rec.take()
                for i in range(n):
                    if appended[i]:
                        dec[i].append({k: v[i] for k, v in b.items()})
                cache, logits = out.past_key_values, out.logits[:, -1]
        dt = time.time() - ts
        timing.append(dict(batch=b0, size=n, prompt_len=Lmax, steps=step + 1, sec=dt))
        for i, (q, a, _p) in enumerate(chunk):
            text = tok.decode(gen[i], skip_special_tokens=True)
            if gsm:
                ans = text.split("</think>")[-1].strip()
            else:
                ans = text.split("Answer:")[-1].strip().split("\n")[0] if cot else extract_answer(text)
            preds.append(ans)
            gold.append(a)
            off, m = Lmax - len(prompts[i]), len(prompts[i])
            k = pre[("r", 0)].shape[-1]
            pidx = np.stack([pre[("r", l)].view(n, Lmax, -1)[i, off:].numpy() for l in range(L)])
            plaidx = np.stack([pre[("la", l)].view(n, Lmax, -1)[i, off:].numpy() for l in range(L - 1)])
            D = min(len(gen[i]), len(dec[i]))
            ds_ = dec[i][:D]
            E = ds_[0][("sc", 0)].shape[-1] if D else 0
            didx = np.stack([[s[("r", l)].numpy() for s in ds_] for l in range(L)]) if D else np.zeros((L, 0, k), np.int16)
            dla = np.stack([[s[("la", l)].numpy() for s in ds_] for l in range(L - 1)]) if D else np.zeros((L - 1, 0, k), np.int16)
            dsc = np.stack([[s[("sc", l)].numpy() for s in ds_] for l in range(L)]) if D else np.zeros((L, 0, 0), np.float16)
            dpj = np.stack([[s[("pj", l)].numpy() for s in ds_] for l in range(L)]) if D else np.zeros((L, 0, 0), np.float16)
            gh = np.full((H, L, D, k), -1, np.int16)
            for s_, s in enumerate(ds_):
                for l in range(L):
                    if ("la", l) in s:
                        gh[0, l, s_] = s[("la", l)].numpy()
                    for hh in range(2, H + 1):
                        if ("gh", hh, l) in s:
                            gh[hh - 1, l, s_] = s[("gh", hh, l)].numpy()
            extra = {}
            if rec.top_m:
                M = rec.top_m
                gix = np.full((H, L, D, M), -1, np.int16)
                gsc = np.zeros((H, L, D, M), np.float32)
                for l in range(L):
                    if D and ("gi", l) in ds_[0]:
                        v = np.stack([s[("gs", l)].numpy() for s in ds_])          # [D, h', M]
                        ix = np.stack([s[("gi", l)].numpy() for s in ds_])
                        gsc[:v.shape[1], l] = v.transpose(1, 0, 2)
                        gix[:ix.shape[1], l] = ix.transpose(1, 0, 2)
                extra = dict(dec_gate_idx=gix, dec_gate_score=gsc)
            np.savez_compressed(qfiles[i], prompt_ids=np.array(prompts[i], np.int32), categories=np.array(cats[i]),
                                gen_ids=np.array(gen[i][:D], np.int32), pre_idx=pidx, dec_idx=didx,
                                la_pre_idx=plaidx, la_dec_idx=dla, dec_score=dsc, dec_proj=dpj, dec_gate_h=gh,
                                layer_ids=np.array(rec.layer_ids), pred=np.array(ans),
                                gold=np.array(json.dumps(list(a) if not isinstance(a, str) else [a])), question=np.array(q),
                                **extra)
        free = gpu_free_gb()
        print(f"[{tag}] {b0 + n}/{len(items)} batch {dt:.0f}s (prompt {Lmax}, steps {step + 1}); GPU free "
              + ", ".join(f"{g}: {f:.0f} GB" for g, f in free.items()), flush=True)
        if min(free.values()) < args.min_free_gb:
            print(f"[{tag}] STOP: GPU headroom below {args.min_free_gb} GB after this batch", flush=True)
            sys.exit(3)
    if gsm:
        acc = gsm_score(preds, [(None, g[0] if isinstance(g, list) else g) for g in gold])[0]
        m = dict(hits_at_1=acc, f1=acc)
    else:
        m = evaluate_metrics(preds, gold)
    json.dump({"model": args.model, "job": job, "hit": m["hits_at_1"], "f1": m["f1"], "predictions": preds,
               "answers": gold, "timing": timing, "layer_ids": rec.layer_ids, "gate_h": H},
              open(os.path.join(args.out_dir, f"{tag}.json"), "w"))
    print(f"[{tag}] Hit {m['hits_at_1']:.3f} F1 {m['f1']:.3f} n={len(preds)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--max_memory", required=True)
    ap.add_argument("--gate_ahead", type=int, default=4)
    ap.add_argument("--proj_dim", type=int, default=64)
    ap.add_argument("--gate_topm", type=int, default=0, help="also store the top-M GATE-h ids and scores (v4: 16)")
    ap.add_argument("--min_free_gb", type=float, default=15.0, help="stop when a GPU has less free memory")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    args.name = args.name or args.model.split("/")[-1]
    mm = {(int(k) if k.isdigit() else k): v for k, v in (kv.split(":") for kv in args.max_memory.split(","))}
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(args.model)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto", device_map="auto", max_memory=mm).eval()
    print(f"loaded in {time.time() - t0:.0f}s; devices {sorted({str(d) for d in getattr(model, "hf_device_map", {}).values()})}", flush=True)
    rec = Recorder2(model, None, args.gate_ahead, args.proj_dim, cfg.hidden_size, args.gate_topm)
    rec.G = GateBank(args.model, cfg, rec.layer_ids)
    g_eos = getattr(model.generation_config, "eos_token_id", None)
    eos = {tok.eos_token_id} | (set(g_eos) if isinstance(g_eos, list) else ({g_eos} if g_eos is not None else set()))
    dev = next(model.parameters()).device
    os.makedirs(args.out_dir, exist_ok=True)
    for job in json.load(open(args.jobs)):
        run_job(job, model, tok, rec, args, eos, dev)
    print("ALL JOBS DONE", flush=True)


if __name__ == "__main__":
    main()
