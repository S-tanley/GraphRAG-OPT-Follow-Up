"""E0 on the Mac: routing traces with MLX (mlx-lm), same prompts / npz format as
experiments/moe/collect_traces.py, so e1_analysis.py, sim.py and the phase-A scripts read them unchanged.

Hook: every mlx-lm MoE block ends with `self.switch_mlp(x, inds)` (SwitchGLU). We wrap SwitchGLU.__call__
to record `inds` (the experts actually used) and `x` (the MoE input). The one-layer-ahead lookahead applies
the NEXT MoE layer's gate to this layer's `x` (Mixtral-offloading / HOBBIT signal): Linear gates (OLMoE,
Qwen3, Mixtral) -> top-k of the logits (softmax is monotone); module gates (DeepSeek-V2, GLM-4.5) are called
directly, so each model's own selection rule (group-limited greedy, sigmoid + bias) is used as is.
`--check` verifies that applying a layer's OWN gate to its own `x` reproduces the recorded experts.

Requests are run one at a time (batch 1: the single-user setting), greedy decoding, raw few-shot text
prompts (no chat template), exactly like the H200 collector.

  .venv-mac/bin/python experiments/moe_edge/collect_traces_mlx.py --model allenai/OLMoE-1B-7B-0924-Instruct \
      --dataset rmanluo/RoG-cwq --cond M0 --n 500 --max_new_tokens 32 --out_dir outputs/moe_edge/traces
"""

import argparse
import json
import os
import random
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.models import switch_layers
from mlx_lm.models.cache import make_prompt_cache

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "graphrag-main"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "common"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "exp2"))
from faithful_pfkv import extract_answer             # noqa: E402
from metrics import evaluate_metrics                  # noqa: E402
from table2_worker import load_ds                     # noqa: E402
from conditions import line_units, pre_suf            # noqa: E402
from run import PATHS, first_line_done                # noqa: E402

COT = "Let's think step by step, then give the final answer on a line starting with 'Answer:'."


def build_prompt(tok, q, paths, cond, cap):
    P, S = pre_suf(tok, q, "fewshot")
    ids, cat = list(P), ["pre"] * len(P)
    if cond == "M0":
        for p in paths[:cap]:
            if p:
                lid, lcat, _ = line_units(tok, p)
                ids += lid
                cat += lcat
    ids += S
    cat += ["question"] * len(S)
    return ids, cat


def build_chat_prompt(tok, q, paths, cap):
    """W2 for instruct models: the same retrieved paths and question inside the model's own chat template
    (raw few-shot prompts make small models answer at once, so there is no long decode). Template text
    around the user message is labelled 'pre'; Qwen3 keeps its default thinking mode."""
    mark = "@@USER@@"
    full = tok.apply_chat_template([{"role": "user", "content": mark}], tokenize=False, add_generation_prompt=True)
    head, tail = full.split(mark)
    intro = "Use the knowledge graph paths below to answer the question.\n"
    ids = tok.encode(head, add_special_tokens=False) + tok.encode(intro, add_special_tokens=False)
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
    """Put an already-built raw prompt (few-shot preamble + paths + 'Q: ... A:') unchanged inside the model's
    chat template as the user message, thinking disabled (Qwen3 emits its thinking / format debris on raw
    text prompts). Template tokens are labelled 'pre'."""
    mark = "@@USER@@"
    kw = {"enable_thinking": False}
    try:
        full = tok.apply_chat_template([{"role": "user", "content": mark}], tokenize=False,
                                       add_generation_prompt=True, **kw)
    except TypeError:
        full = tok.apply_chat_template([{"role": "user", "content": mark}], tokenize=False, add_generation_prompt=True)
    head, tail = full.split(mark)
    hi = tok.encode(head, add_special_tokens=False)
    ti = tok.encode(tail, add_special_tokens=False)
    bos = tok.bos_token_id
    body, bcat = (ids[1:], cat[1:]) if bos is not None and ids and ids[0] == bos else (ids, cat)
    return hi + list(body) + ti, ["pre"] * len(hi) + list(bcat) + ["pre"] * len(ti)


class Recorder:
    def __init__(self, model, k):
        self.k = k
        self.mt = model.args.model_type
        self.moe = []                                   # (layer id, block)
        for i, layer in enumerate(model.layers):
            blk = getattr(layer, "mlp", None) or getattr(layer, "block_sparse_moe", None)
            if blk is not None and hasattr(blk, "switch_mlp"):
                self.moe.append((i, blk))
        self.pos = {id(b.switch_mlp): j for j, (_i, b) in enumerate(self.moe)}
        for j, (_i, b) in enumerate(self.moe):        # offloaded blocks (offload_engine.OffloadSwitch)
            if hasattr(b.switch_mlp, "on_call"):
                b.switch_mlp.on_call = (lambda x, ind, j=j: self.buf.__setitem__(j, (ind, x))
                                        if self.buf is not None else None)
        self.layer_ids = [i for i, _b in self.moe]
        self.buf = None
        rec = self
        orig = switch_layers.SwitchGLU.__call__

        def patched(sw, x, indices):
            j = rec.pos.get(id(sw))
            if rec.buf is not None and j is not None:
                rec.buf[j] = (indices, x)
            return orig(sw, x, indices)
        switch_layers.SwitchGLU.__call__ = patched

    def gate_select(self, j, x):
        """Experts MoE layer j would pick for x, copying each block's own selection code exactly
        (bf16 logits tie often, so softmax-vs-logits or argpartition direction changes tie-breaking)."""
        g, k = self.moe[j][1].gate, self.k
        if isinstance(g, (nn.Linear, nn.QuantizedLinear)):
            logits = g(x)
            if self.mt == "olmoe":
                w = mx.softmax(logits, axis=1 if logits.ndim == 2 else -1, precise=True)
                return mx.argpartition(-w, kth=k - 1, axis=-1)[..., :k]
            if self.mt == "qwen3_moe":
                w = mx.softmax(logits, axis=-1, precise=True)
                return mx.argpartition(w, kth=-k, axis=-1)[..., -k:]
            return mx.argpartition(-logits, kth=k - 1, axis=-1)[..., :k]      # mixtral
        out = g(x)                                                               # deepseek_v2, glm4_moe
        return out[0] if isinstance(out, tuple) else out

    def start(self):
        self.buf = {}

    def take(self, lookahead=True, check=False):
        """-> routes [L][T, k] int16, lookahead [L-1][T, k] int16 (or None), self-consistency or None."""
        b, self.buf = self.buf, None
        n = len(self.moe)
        idx = [b[j][0].reshape(-1, self.k) for j in range(n)]
        la = [self.gate_select(j + 1, b[j][1]).reshape(-1, self.k) for j in range(n - 1)] if lookahead else None
        own = [self.gate_select(j, b[j][1]).reshape(-1, self.k) for j in range(n)] if check else None
        mx.eval(idx, la or [], own or [])
        routes = [np.sort(np.array(a), -1).astype(np.int16) for a in idx]
        las = [np.sort(np.array(a), -1).astype(np.int16) for a in la] if la is not None else None
        cons = None
        if own is not None:
            cons = float(np.mean([(np.sort(np.array(o), -1) == r).all(-1).mean() for o, r in zip(own, routes)]))
        return routes, las, cons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="rmanluo/RoG-cwq")
    ap.add_argument("--paths_jsonl", default=None, help="Wtrain: jsonl of {question, answer, paths}")
    ap.add_argument("--rel_map", default=None, help="Wsyn / Wsyn-rand relation paraphrase map")
    ap.add_argument("--cond", default="M0", choices=["direct", "M0"])
    ap.add_argument("--cot", action="store_true", help="W2: long chain-of-thought output")
    ap.add_argument("--chat", action="store_true", help="W2 for instruct models: use the chat template")
    ap.add_argument("--chat_fewshot", action="store_true",
                    help="the raw few-shot prompt inside the chat template, thinking off (Qwen3)")
    ap.add_argument("--name", default=None, help="model name used in output tags (default: repo name)")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--no_trace", action="store_true")
    ap.add_argument("--no_lookahead", action="store_true")
    ap.add_argument("--check", type=int, default=0, help="self-consistency check on the first N requests")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--offload_store", default=None, help="B tier: experts.bin dir (export_experts.py)")
    ap.add_argument("--slots", type=int, default=64, help="B tier: expert cache slots per layer")
    ap.add_argument("--policy", default="sw-moeinf")
    args = ap.parse_args()

    t0 = time.time()
    eng = None
    if args.offload_store:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from offload_engine import OffloadEngine
        model, wrapper = load(args.model, lazy=True)
        eng = OffloadEngine.attach(model, args.offload_store, args.slots, policy=args.policy)
        print(f"offload: {args.slots} slots/layer, resident {mx.get_active_memory() / 1e9:.1f} GB", flush=True)
    else:
        try:
            model, wrapper = load(args.model)
        except TypeError as e:
            # OLMoE-0924's config omits rms_norm_eps (transformers fills in its default 1e-5; mlx-lm does not)
            if "rms_norm_eps" not in str(e):
                raise
            print("config has no rms_norm_eps; using the transformers default 1e-5", flush=True)
            model, wrapper = load(args.model, model_config={"rms_norm_eps": 1e-5})
    tok = wrapper._tokenizer
    cfg_k = None
    for attr in ("num_experts_per_tok", "moe_top_k", "top_k"):
        cfg_k = cfg_k or getattr(model.args, attr, None)
    rec = Recorder(model, cfg_k)
    print(f"loaded {args.model} in {time.time() - t0:.0f}s; {len(rec.moe)} MoE layers "
          f"({rec.layer_ids[0]}..{rec.layer_ids[-1]}), k={cfg_k}", flush=True)
    eos = {tok.eos_token_id} | set(getattr(wrapper, "eos_token_ids", []) or [])

    if args.paths_jsonl:
        rows = [json.loads(l) for l in open(args.paths_jsonl)][args.start:args.start + args.n]
        items = [(r["question"], r["answer"], r["paths"]) for r in rows]
        dname = "train"
    else:
        full = load_ds(args.dataset)
        ds = full.select(range(args.start, min(len(full), args.start + args.n)))
        pdict = json.load(open(os.path.join(ROOT, "graphrag-main/precomputed_paths_v2", PATHS[args.dataset])))
        items = [(ds[i]["question"], ds[i]["answer"], pdict.get(ds[i]["question"], [])) for i in range(len(ds))]
        dname = args.dataset.split("-")[-1]
    rel_map = json.load(open(args.rel_map)) if args.rel_map else None
    rand_syn = bool(rel_map) and any(isinstance(v, list) for v in rel_map.values())
    tag = (f"{args.name or args.model.split('/')[-1]}_{dname}_{args.cond}{'_cot' if args.cot else ''}{'_chat' if args.chat else ''}"
           f"{('_synr' if rand_syn else '_syn') if rel_map else ''}_cap{args.cap}")
    tdir = os.path.join(args.out_dir, tag)
    os.makedirs(tdir, exist_ok=True)

    preds, gold, timing, cons = [], [], [], []
    for qi, (q, a, paths) in enumerate(items):
        qf = os.path.join(tdir, f"q{args.start + qi:05d}.npz")
        if not args.no_trace and os.path.exists(qf):
            z = np.load(qf)
            preds.append(str(z["pred"]))
            gold.append(json.loads(str(z["gold"])))
            continue
        paths = [[tuple(t) for t in p] for p in paths]
        if rel_map:
            pick = ({r: random.Random(q).choice(v) if isinstance(v, list) else v for r, v in rel_map.items()}
                    if rand_syn else rel_map)
            paths = [[(h, pick.get(r, r), t) for h, r, t in p] for p in paths]
        if args.chat:
            ids, cat = build_chat_prompt(tok, q + ((" " + COT) if args.cot else ""), paths, args.cap)
        else:
            ids, cat = build_prompt(tok, q + ((" " + COT) if args.cot else ""), paths, args.cond, args.cap)
            if args.chat_fewshot:
                ids, cat = wrap_chat(tok, ids, cat)
        cache = make_prompt_cache(model)
        if eng is not None:
            eng.new_request()
            eng.phase = "prefill"
        ts = time.time()
        trace = not args.no_trace
        if trace:
            rec.start()
        logits = model(mx.array([ids]), cache=cache)[:, -1]
        if trace:
            pre, pre_la, c = rec.take(not args.no_lookahead, check=qi < args.check)
            if c is not None:
                cons.append(c)
        t_pre = time.time() - ts
        if eng is not None:
            eng.phase = "decode"
        gen, dec, dec_la = [], [], []
        for step in range(args.max_new_tokens):
            nxt = int(mx.argmax(logits, axis=-1).item())
            if nxt in eos:
                break
            gen.append(nxt)
            if not args.cot and first_line_done(tok.decode(gen, skip_special_tokens=True)):
                break
            if step == args.max_new_tokens - 1:
                break
            if trace:
                rec.start()
            logits = model(mx.array([[nxt]]), cache=cache)[:, -1]
            if trace:
                r, la, _ = rec.take(not args.no_lookahead)
                dec.append(r)
                dec_la.append(la)
        dt = time.time() - ts
        timing.append(dict(q=qi, prompt_len=len(ids), steps=len(gen), prefill_s=t_pre, total_s=dt))
        text = tok.decode(gen, skip_special_tokens=True)
        ans = text.split("Answer:")[-1].strip().split("\n")[0] if args.cot else extract_answer(text)
        preds.append(ans)
        gold.append(a)
        if trace:
            nl, k = len(pre), pre[0].shape[-1]
            D = len(dec)                                   # routing of gen[0..D-1] as decode inputs
            pidx = np.stack(pre)
            plaidx = np.stack(pre_la) if pre_la is not None else np.zeros((0, len(ids), k), np.int16)
            didx = np.stack([np.stack([dec[s][l][0] for s in range(D)]) for l in range(nl)]) if D \
                else np.zeros((nl, 0, k), np.int16)
            dla = (np.stack([np.stack([dec_la[s][l][0] for s in range(D)]) for l in range(nl - 1)])
                   if D and dec_la[0] is not None else np.zeros((nl - 1 if pre_la is not None else 0, 0, k), np.int16))
            np.savez_compressed(qf, prompt_ids=np.array(ids, np.int32), categories=np.array(cat),
                                gen_ids=np.array(gen[:D], np.int32), pre_idx=pidx, pre_w=np.zeros(0, np.float16),
                                dec_idx=didx, dec_w=np.zeros(0, np.float16), la_pre_idx=plaidx, la_dec_idx=dla,
                                layer_ids=np.array(rec.layer_ids), pred=np.array(ans),
                                gold=np.array(json.dumps(list(a) if not isinstance(a, str) else [a])),
                                question=np.array(q))
        if (qi + 1) % 25 == 0 or qi == len(items) - 1:
            m = evaluate_metrics(preds, gold)
            print(f"[{tag}] {qi + 1}/{len(items)} hit {m['hits_at_1']:.3f}  last: prompt {len(ids)} prefill "
                  f"{t_pre * 1e3:.0f}ms, {len(gen)} tok in {dt:.1f}s"
                  + (f"  self-consistency {np.mean(cons):.4f}" if cons else ""), flush=True)
    m = evaluate_metrics(preds, gold)
    json.dump({"model": args.model, "dataset": dname, "cond": args.cond, "cot": args.cot, "syn": bool(rel_map),
               "cap": args.cap, "hit": m["hits_at_1"], "f1": m["f1"], "predictions": preds, "answers": gold,
               "questions": [q for q, _a, _p in items], "timing": timing, "layer_ids": rec.layer_ids,
               "self_consistency": float(np.mean(cons)) if cons else None, "backend": "mlx", "batch": 1,
               "offload": dict(store=args.offload_store, slots=args.slots, policy=args.policy,
                               stats=eng.stats) if eng is not None else None},
              open(os.path.join(args.out_dir, f"{tag}.json"), "w"))
    print(f"[{tag}] Hit {m['hits_at_1']:.3f} F1 {m['f1']:.3f} n={len(preds)}", flush=True)


if __name__ == "__main__":
    main()
