"""
Fine-tune the RoG relation-path planner on MetaQA (RoG's transfer recipe).
LoRA on the attention projections; loss only on the target relation sequence.
Saves a MERGED model (base+LoRA) to --out so precompute can load it directly.

  python finetune_planner.py --data metaqa_plan_train.jsonl --out RoG_metaqa_planner \
      --epochs 3 --bs 8 --lr 2e-4
"""
import argparse
import json
import math
import random

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

PROMPT = ("<s>[INST] Please generate a valid relation path that can be helpful for "
          "answering the following question: {q} [/INST]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="rmanluo/RoG")
    ap.add_argument("--data", default="metaqa_plan_train.jsonl")
    ap.add_argument("--out", default="RoG_metaqa_planner")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    dev = "cuda"

    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float16).to(dev)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    rows = [json.loads(l) for l in open(args.data)]
    random.shuffle(rows)

    def encode(ex):
        prompt = PROMPT.format(q=ex["question"])
        target = " " + "<SEP>".join(ex["rule"])     # planner emits the relation seq
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        t_ids = tok(target, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        ids = (p_ids + t_ids)[:args.max_len]
        labels = ([-100] * len(p_ids) + t_ids)[:args.max_len]
        return ids, labels

    enc = [encode(r) for r in rows]

    def collate(batch):
        m = max(len(x[0]) for x in batch)
        input_ids, labels, attn = [], [], []
        for ids, lab in batch:
            pad = m - len(ids)
            input_ids.append(ids + [tok.pad_token_id] * pad)
            labels.append(lab + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return (torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attn))

    dl = DataLoader(enc, batch_size=args.bs, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    total = max(1, len(dl) * args.epochs)
    from transformers import get_cosine_schedule_with_warmup
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=max(1, total // 10),
                                            num_training_steps=total)
    model.train()
    step = 0
    for ep in range(args.epochs):
        running = 0.0
        for input_ids, labels, attn in dl:
            input_ids, labels, attn = input_ids.to(dev), labels.to(dev), attn.to(dev)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            running += loss.item(); step += 1
            if step % 50 == 0:
                print(f"epoch {ep} step {step}/{total} loss {running/50:.4f}", flush=True)
                running = 0.0

    print("merging LoRA and saving...", flush=True)
    model = model.merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    print(f"saved merged planner -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
