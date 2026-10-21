"""MoE plan E0: reader accuracy and routing traces for large MoE models.

Every supported model (Mixtral, Qwen3-MoE, DeepSeek-V2, GLM-4.5) has an experts module whose
forward receives the FINAL routing decision (top_k_index, top_k_weights); a forward pre-hook on
it records exactly which experts each token used, whatever the router logic.

Also records the "one layer ahead" gate prediction used by Mixtral-offloading / HOBBIT-style
prefetchers: the router of MoE layer l+1 applied to the input hidden state of MoE layer l. Gate
weights are read straight from the checkpoint (works even when layers are CPU-offloaded) and each
model's own routing rule is re-implemented (softmax top-k / DeepSeek group-limited greedy /
GLM sigmoid + bias correction).

Questions are batched (left padding); greedy decoding with a KV cache. Per question we save one
npz: prompt ids + token categories, routing [layer, token, k] ids and weights for prompt and for
every generated token, the lookahead predictions, and the prediction / gold answer (so an
interrupted run resumes by skipping batches whose npz files all exist).

  python experiments/moe/collect_traces.py --model zai-org/GLM-4.5-Air --dataset rmanluo/RoG-cwq \
      --cond M0 --n 500 --batch 32 --out_dir outputs/moe/traces
Options: --cot (W2 long output), --rel_map map.json (Wsyn: relation paraphrases),
         --paths_jsonl file (Wtrain: {question, answer, paths} per line), --no_trace (accuracy only)
"""

import argparse
import gc
import json
import os
import re
import sys
import time

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "graphrag-main"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "common"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "exp2"))
from faithful_pfkv import extract_answer             # noqa: E402
from metrics import evaluate_metrics                  # noqa: E402
from table2_worker import load_ds                     # noqa: E402
from conditions import line_units, pre_suf, trie_text_lines   # noqa: E402
from run import PATHS, first_line_done                # noqa: E402

EXPERT_CLASSES = ("Experts", "NaiveMoe")
COT = "Let's think step by step, then give the final answer on a line starting with 'Answer:'."


def build_prompt(tok, q, paths, cond, cap):
    """Returns ids and per-token categories. cond: direct | M0 | T. Few-shot template."""
    P, S = pre_suf(tok, q, "fewshot")
    ids, cat = list(P), ["pre"] * len(P)
    if cond == "M0":
        for p in paths[:cap]:
            if p:
                lid, lcat, _ = line_units(tok, p)
                ids += lid; cat += lcat
    elif cond == "T":
        for s in trie_text_lines(paths, cap):
            t = tok.encode(s, add_special_tokens=False)
            ids += t; cat += ["path"] * len(t)
    ids += S; cat += ["question"] * len(S)
    return ids, cat


# ----------------------------------------------------------------------------- lookahead gates
class GateBank:
    """Router weights of every MoE layer, loaded from the checkpoint, plus each model's rule."""

    def __init__(self, model_name, cfg, layer_ids):
        self.cfg, self.mt = cfg, cfg.model_type
        self.k = cfg.num_experts_per_tok
        idx = json.load(open(hf_hub_download(model_name, "model.safetensors.index.json")))["weight_map"]
        sub = "block_sparse_moe" if self.mt == "mixtral" else "mlp"
        self.W, self.bias, self.cache = {}, {}, {}
        files = {}
        for L in layer_ids:
            for name, store in [(f"model.layers.{L}.{sub}.gate.weight", self.W),
                                (f"model.layers.{L}.{sub}.gate.e_score_correction_bias", self.bias)]:
                if name in idx:
                    files.setdefault(idx[name], []).append((name, L, store))
        for f, items in files.items():
            with safe_open(hf_hub_download(model_name, f), framework="pt") as fh:
                for name, L, store in items:
                    store[L] = fh.get_tensor(name)          # keep checkpoint dtype

    def _w(self, L, dev):
        key = (L, str(dev))
        if key not in self.cache:
            self.cache[key] = (self.W[L].to(dev), self.bias[L].to(dev) if L in self.bias else None)
        return self.cache[key]

    @torch.no_grad()
    def route(self, L, h):
        """Top-k expert ids that MoE layer L would pick for hidden states h [N, H].
        Matches each model's own numerics: Mixtral / Qwen3 compute the router GEMM in the model
        dtype then softmax in fp32; DeepSeek-V2 and GLM-4.5 compute it in fp32."""
        W, b = self._w(L, h.device)
        if self.mt in ("mixtral", "qwen3_moe"):
            logits = torch.nn.functional.linear(h, W.to(h.dtype))
            return logits.float().softmax(-1).topk(self.k, dim=-1).indices
        logits = torch.nn.functional.linear(h.float(), W.float())
        b = b.float() if b is not None else None
        if self.mt == "deepseek_v2":
            s = logits.softmax(-1)
            if getattr(self.cfg, "topk_method", "greedy") == "group_limited_greedy":
                G, tg = self.cfg.n_group, self.cfg.topk_group
                gs = s.view(s.shape[0], G, -1).max(-1).values
                gm = torch.zeros_like(gs).scatter_(1, gs.topk(tg, dim=-1).indices, 1)
                s = s.masked_fill(~gm.repeat_interleave(s.shape[1] // G, dim=1).bool(), 0.0)
            return s.topk(self.k, dim=-1).indices
        if self.mt == "glm4_moe":
            s = logits.sigmoid() + (b if b is not None else 0)
            G, tg = self.cfg.n_group, self.cfg.topk_group
            gs = s.view(s.shape[0], G, -1).topk(min(2, s.shape[1] // G), dim=-1).values.sum(-1)
            gm = torch.zeros_like(gs).scatter_(1, gs.topk(tg, dim=-1).indices, 1)
            s = s.masked_fill(~gm.repeat_interleave(s.shape[1] // G, dim=1).bool(), 0.0)
            return s.topk(self.k, dim=-1).indices
        raise ValueError(self.mt)


class Recorder:
    def __init__(self, model, gates):
        self.layers, self.layer_ids = [], []
        name2mod = dict(model.named_modules())
        for name, mod in model.named_modules():
            if type(mod).__name__.endswith(EXPERT_CLASSES):
                self.layers.append(name)
                self.layer_ids.append(int(re.search(r"layers\.(\d+)\.", name).group(1)))
        self.gates = gates(self.layer_ids) if gates else None
        for li, name in enumerate(self.layers):
            name2mod[name].register_forward_pre_hook(self._exp_hook(li), with_kwargs=True)
            if self.gates is not None and li + 1 < len(self.layers):
                block = name2mod[name.rsplit(".", 1)[0]]
                block.register_forward_pre_hook(self._block_hook(li), with_kwargs=True)
        self.buf = None

    def _exp_hook(self, li):
        def h(mod, args, kwargs):
            idx = kwargs.get("top_k_index", args[1] if len(args) > 1 else None)
            w = kwargs.get("top_k_weights", args[2] if len(args) > 2 else None)
            if self.buf is not None and idx is not None:
                self.buf[("r", li)] = (idx.detach().to("cpu", torch.int16), w.detach().float().to("cpu", torch.float16))
        return h

    def _block_hook(self, li):
        nxt = self.layer_ids[li + 1]

        def h(mod, args, kwargs):
            x = args[0] if args else kwargs.get("hidden_states")
            if self.buf is not None and x is not None:
                self.buf[("la", li)] = self.gates.route(nxt, x.reshape(-1, x.shape[-1])).to("cpu", torch.int16)
        return h

    def start(self):
        self.buf = {}

    def take(self):
        b, self.buf = self.buf, None
        n = len(self.layers)
        return [b[("r", i)] for i in range(n)], [b.get(("la", i)) for i in range(n - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="rmanluo/RoG-cwq")
    ap.add_argument("--paths_jsonl", default=None, help="Wtrain: jsonl of {question, answer, paths}")
    ap.add_argument("--rel_map", default=None, help="Wsyn: json {relation: paraphrase}")
    ap.add_argument("--cond", default="M0", choices=["direct", "M0", "T"])
    ap.add_argument("--cot", action="store_true", help="W2: long chain-of-thought output")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--max_memory", required=True, help="e.g. 0:130GiB,1:130GiB,2:130GiB,cpu:1100GiB")
    ap.add_argument("--no_trace", action="store_true", help="accuracy only")
    ap.add_argument("--no_lookahead", action="store_true")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    mm = {}
    for kv in args.max_memory.split(","):
        k, v = kv.split(":")
        mm[int(k) if k.isdigit() else k] = v
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(args.model)
    t0 = time.time()
    for attempt in range(3):
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto", device_map="auto",
                                                         max_memory=mm).eval()
            break
        except RuntimeError as e:
            # transformers 5 fuses per-expert weights while loading; the temporaries can OOM a GPU that the
            # device map already fills (seen with DeepSeek-V2.5). Retry with less GPU memory per device.
            if "conversion" not in str(e) or attempt == 2:
                raise
            mm = {k: (f"{int(v[:-3]) - 24}GiB" if isinstance(k, int) else v) for k, v in mm.items()}
            print(f"load failed during weight conversion; retrying with max_memory={mm}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    print(f"loaded in {time.time() - t0:.0f}s; devices: "
          f"{sorted({str(d) for d in getattr(model, 'hf_device_map', {}).values()})}", flush=True)
    gates = None if (args.no_trace or args.no_lookahead) else (lambda ids: GateBank(args.model, cfg, ids))
    rec = Recorder(model, gates)
    print(f"hooked {len(rec.layers)} expert modules (layers {rec.layer_ids[0]}..{rec.layer_ids[-1]}); "
          f"lookahead={'on' if rec.gates else 'off'}", flush=True)

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
    tag = (f"{args.model.split('/')[-1]}_{dname}_{args.cond}{'_cot' if args.cot else ''}"
           f"{('_synr' if rand_syn else '_syn') if rel_map else ''}_cap{args.cap}")
    tdir = os.path.join(args.out_dir, tag)
    os.makedirs(tdir, exist_ok=True)
    g_eos = getattr(model.generation_config, "eos_token_id", None)
    eos = {tok.eos_token_id} | (set(g_eos) if isinstance(g_eos, list) else ({g_eos} if g_eos is not None else set()))
    dev = next(model.parameters()).device
    nl = len(rec.layers)

    preds, gold, timing, n_syn_paths = [], [], [], 0
    for b0 in range(0, len(items), args.batch):
        chunk = items[b0:b0 + args.batch]
        qfiles = [os.path.join(tdir, f"q{args.start + b0 + i:05d}.npz") for i in range(len(chunk))]
        if not args.no_trace and all(os.path.exists(f) for f in qfiles):         # resume
            for f in qfiles:
                z = np.load(f, allow_pickle=False)
                preds.append(str(z["pred"])); gold.append(json.loads(str(z["gold"])))
            print(f"[{tag}] {b0 + len(chunk)}/{len(items)} (resumed)", flush=True)
            continue
        prompts, cats = [], []
        for q, _a, paths in chunk:
            paths = [[tuple(t) for t in p] for p in paths]
            if rel_map:
                if rand_syn:     # Wsyn-rand: each request draws its own variant per relation
                    import random as _r
                    rng = _r.Random(q)
                    pick = {r: rng.choice(v) if isinstance(v, list) else v for r, v in rel_map.items()}
                else:
                    pick = rel_map
                new = [[(h, pick.get(r, r), t) for h, r, t in p] for p in paths]
                n_syn_paths += sum(any(r in rel_map for _h, r, _t in p) for p in paths[:args.cap])
                paths = new
            ids, cat = build_prompt(tok, q + ((" " + COT) if args.cot else ""), paths, args.cond, args.cap)
            prompts.append(ids); cats.append(cat)
        B, Lmax = len(prompts), max(len(p) for p in prompts)
        inp = torch.full((B, Lmax), tok.pad_token_id, dtype=torch.long)
        am = torch.zeros((B, Lmax), dtype=torch.long)
        for i, p in enumerate(prompts):
            inp[i, Lmax - len(p):] = torch.tensor(p); am[i, Lmax - len(p):] = 1
        inp, am = inp.to(dev), am.to(dev)
        pos = (am.cumsum(-1) - 1).clamp(min=0)
        gen = [[] for _ in range(B)]
        done = [False] * B
        pre, pre_la, dec, dec_la = None, None, [[] for _ in range(B)], [[] for _ in range(B)]
        ts = time.time()
        with torch.no_grad():
            if not args.no_trace:
                rec.start()
            out = model(input_ids=inp, attention_mask=am, position_ids=pos, use_cache=True)
            if not args.no_trace:
                pre, pre_la = rec.take()
            cache, logits = out.past_key_values, out.logits[:, -1]
            cur_pos = pos[:, -1]
            step = 0
            for step in range(args.max_new_tokens):
                nxt = logits.argmax(-1)
                appended = [False] * B
                for i in range(B):
                    if not done[i]:
                        t = int(nxt[i])
                        if t in eos:
                            done[i] = True
                        else:
                            gen[i].append(t)
                            appended[i] = True
                            if not args.cot and first_line_done(tok.decode(gen[i], skip_special_tokens=True)):
                                done[i] = True
                if all(done):
                    break
                am = torch.cat([am, torch.ones((B, 1), dtype=am.dtype, device=dev)], dim=1)
                cur_pos = cur_pos + 1
                if not args.no_trace:
                    rec.start()
                out = model(input_ids=nxt[:, None], attention_mask=am, position_ids=cur_pos[:, None],
                            past_key_values=cache, use_cache=True)
                if not args.no_trace:
                    r, la = rec.take()
                    for i in range(B):                      # routing of the token just appended
                        if appended[i]:
                            dec[i].append([(x[0][i], x[1][i]) for x in r])
                            dec_la[i].append([x[i] if x is not None else None for x in la])
                cache, logits = out.past_key_values, out.logits[:, -1]
        dt = time.time() - ts
        timing.append(dict(batch=b0, size=B, prompt_len=Lmax, steps=step + 1, sec=dt))
        for i, (q, a, _p) in enumerate(chunk):
            text = tok.decode(gen[i], skip_special_tokens=True)
            ans = text.split("Answer:")[-1].strip().split("\n")[0] if args.cot else extract_answer(text)
            preds.append(ans); gold.append(a)
            if args.no_trace:
                continue
            n, off = len(prompts[i]), Lmax - len(prompts[i])
            k = pre[0][0].shape[-1]
            pidx = np.stack([pre[l][0].view(B, Lmax, -1)[i, off:].numpy() for l in range(nl)])
            pw = np.stack([pre[l][1].view(B, Lmax, -1)[i, off:].numpy() for l in range(nl)])
            la_ok = pre_la and all(x is not None for x in pre_la)
            plaidx = (np.stack([pre_la[l].view(B, Lmax, -1)[i, off:].numpy() for l in range(nl - 1)])
                      if la_ok else np.zeros((0, n, k), np.int16))
            D = min(len(gen[i]), len(dec[i]))
            didx = (np.stack([[dec[i][s][l][0].numpy() for s in range(D)] for l in range(nl)]) if D
                    else np.zeros((nl, 0, k), np.int16))
            dw = (np.stack([[dec[i][s][l][1].numpy() for s in range(D)] for l in range(nl)]) if D
                  else np.zeros((nl, 0, k), np.float16))
            dla = (np.stack([[dec_la[i][s][l].numpy() for s in range(D)] for l in range(nl - 1)]) if (D and la_ok)
                   else np.zeros((max(nl - 1, 0) if la_ok else 0, 0, k), np.int16))
            np.savez_compressed(qfiles[i], prompt_ids=np.array(prompts[i], np.int32), categories=np.array(cats[i]),
                                gen_ids=np.array(gen[i][:D], np.int32), pre_idx=pidx, pre_w=pw, dec_idx=didx,
                                dec_w=dw, la_pre_idx=plaidx, la_dec_idx=dla, layer_ids=np.array(rec.layer_ids),
                                pred=np.array(ans), gold=np.array(json.dumps(list(a) if not isinstance(a, str) else [a])),
                                question=np.array(q))
        print(f"[{tag}] {b0 + B}/{len(items)} batch {dt:.0f}s (prompt {Lmax}, steps {step + 1})", flush=True)

    m = evaluate_metrics(preds, gold)
    json.dump({"model": args.model, "dataset": dname, "cond": args.cond, "cot": args.cot, "syn": bool(rel_map),
               "cap": args.cap, "hit": m["hits_at_1"], "f1": m["f1"], "predictions": preds, "answers": gold,
               "questions": [q for q, _a, _p in items], "timing": timing, "layers": rec.layers,
               "layer_ids": rec.layer_ids, "syn_paths": n_syn_paths},
              open(os.path.join(args.out_dir, f"{tag}.json"), "w"))
    print(f"[{tag}] Hit {m['hits_at_1']:.3f} F1 {m['f1']:.3f} n={len(preds)}", flush=True)


if __name__ == "__main__":
    main()
