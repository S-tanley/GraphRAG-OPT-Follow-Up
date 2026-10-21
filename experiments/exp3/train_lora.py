"""LoRA fine-tuning of the reader under a given attention condition (Exp3, remedy 4).

The forward pass is the same masked computation used at inference (MaskedEngine helpers):
context rows use the condition's mask and pos_int; suffix+answer rows read the context keys
at pos_out and are causal among themselves. Loss on answer tokens only.

  python experiments/exp3/train_lora.py --cond M2-hyb --out outputs/exp3/lora_RoG_M2-hyb
"""

import argparse
import json
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "graphrag-main"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "common"))
from conditions import build             # noqa: E402
from masked_engine import MaskedEngine   # noqa: E402


def forward_loss(eng, ctx, suf, s0, target):
    dev = eng.dev
    n = len(ctx["ids"])
    tail = suf + target
    t = len(tail)
    mask_c = torch.as_tensor(ctx["mask"], device=dev, dtype=torch.bool)
    mask_t = torch.ones(t, n + t, device=dev, dtype=torch.bool)
    mask_t[:, n:] = torch.tril(mask_t[:, n:])
    ci, si = eng._cos_sin(ctx["pos_int"])
    co, so = eng._cos_sin(ctx["pos_out"])
    ct, st = eng._cos_sin(list(range(s0, s0 + t)))
    Hc = eng.core.embed_tokens(torch.tensor(ctx["ids"], device=dev))
    Ht = eng.core.embed_tokens(torch.tensor(tail, device=dev))
    for layer in eng.layers:
        qc, kc, vc = eng._qkv(layer, layer.input_layernorm(Hc))
        qt, kt, vt = eng._qkv(layer, layer.input_layernorm(Ht))
        ac = eng._attn(layer, eng._rope(qc, ci, si), eng._rope(kc, ci, si), vc, mask_c)
        K = torch.cat([eng._rope(kc, co, so), eng._rope(kt, ct, st)], dim=1)
        V = torch.cat([vc, vt], dim=1)
        at = eng._attn(layer, eng._rope(qt, ct, st), K, V, mask_t)
        Hc = Hc + ac
        Hc = Hc + layer.mlp(layer.post_attention_layernorm(Hc))
        Ht = Ht + at
        Ht = Ht + layer.mlp(layer.post_attention_layernorm(Ht))
    logits = eng.model.lm_head(eng.core.norm(Ht[len(suf) - 1:-1])).float()
    return F.cross_entropy(logits, torch.tensor(target, device=dev))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rmanluo/RoG")
    ap.add_argument("--cond", default="M2-hyb")
    ap.add_argument("--data", default=os.path.join(ROOT, "outputs/exp3/train_paths.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_train", type=int, default=-1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max_answers", type=int, default=5)
    args = ap.parse_args()

    torch.manual_seed(0)
    template = "rog" if "rog" in args.model.lower() else "fewshot"
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf" if template == "rog" else args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda")
    model.gradient_checkpointing_disable()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                             task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    eng = MaskedEngine(model.base_model.model)

    rows = [json.loads(l) for l in open(args.data)]
    if args.n_train > 0:
        rows = rows[:args.n_train]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    total = args.epochs * len(rows) // args.accum
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / max(1, total)))
    model.train()
    step, t0, run = 0, time.time(), 0.0
    for ep in range(args.epochs):
        random.Random(ep).shuffle(rows)
        for i, r in enumerate(rows):
            paths = [[tuple(t) for t in p] for p in r["paths"]]
            ctx, suf, s0, _ = build(args.cond, tok, r["question"], paths, template, 8)
            ans = r["answer"][:args.max_answers]
            text = ("\n".join(ans)) if template == "rog" else (" " + ans[0] + "\n")
            target = tok.encode(text, add_special_tokens=False) + ([tok.eos_token_id] if template == "rog" else [])
            loss = forward_loss(eng, ctx, suf, s0, target) / args.accum
            loss.backward()
            run += loss.item()
            if (i + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 50 == 0:
                    print(f"ep {ep} step {step}/{total} loss {run / 50:.4f} {time.time() - t0:.0f}s", flush=True)
                    run = 0.0
    model.save_pretrained(args.out)
    json.dump(vars(args), open(os.path.join(args.out, "train_args.json"), "w"), indent=1)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
