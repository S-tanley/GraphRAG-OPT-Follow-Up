"""Wsyn-rand: several token-disjoint paraphrases per relation, so that requests sharing a schema
no longer share tokens (each request picks one variant at random in collect_traces.py).

Variant 0 is the hand-checked phrase from syn_map.json; extra variants come from Qwen3-8B and are
kept only if they overlap little with the original identifier AND with the variants already
kept (max token-overlap over all four target tokenizers).

  python experiments/moe/make_syn_variants.py --base outputs/moe/syn_map.json --out outputs/moe/syn_map_rand.json
"""

import argparse
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGETS = ["mistralai/Mixtral-8x22B-Instruct-v0.1", "Qwen/Qwen3-235B-A22B", "deepseek-ai/DeepSeek-V2.5",
           "zai-org/GLM-4.5-Air"]
PROMPT = ("Give 6 DIFFERENT short English phrasings (2-6 words each) of the Freebase relation below, "
          "each stating the relation from the subject to the object. Use different vocabulary in each "
          "phrasing and never reuse the identifier's own words. One phrasing per line, no numbering.\n"
          "Identifier: {rel}\nExample phrasing: {ex}")


def ids(tok, s):
    return set(tok.encode(" " + s, add_special_tokens=False))


def overlap(toks, a, b):
    return max(len(ids(t, a) & ids(t, b)) / max(1, len(ids(t, b))) for t in toks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max_overlap", type=float, default=0.25)
    args = ap.parse_args()
    base = json.load(open(args.base))
    toks = [AutoTokenizer.from_pretrained(m) for m in TARGETS]
    gtok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    gen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16).to("cuda").eval()
    out, short = {}, 0
    for rel, v0 in base.items():
        enc = gtok.apply_chat_template([{"role": "user", "content": PROMPT.format(rel=rel, ex=v0)}],
                                       add_generation_prompt=True, enable_thinking=False,
                                       return_tensors="pt", return_dict=True)
        with torch.no_grad():
            o = gen.generate(enc["input_ids"].to("cuda"), attention_mask=enc["attention_mask"].to("cuda"),
                             max_new_tokens=120, do_sample=False)
        cands = [re.sub(r"^[\-\*\d\.\)\s]+|[\"'.]", "", s).strip()
                 for s in gtok.decode(o[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).split("\n")]
        keep = [v0]
        for c in cands:
            if not c or len(c.split()) > 8 or len(keep) >= args.k:
                continue
            if overlap(toks, rel, c) <= args.max_overlap and all(overlap(toks, k_, c) <= args.max_overlap for k_ in keep):
                keep.append(c)
        short += len(keep) < 2
        out[rel] = keep
        print(f"{rel:55s} {keep}", flush=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}: {len(out)} relations, mean variants {sum(map(len, out.values())) / len(out):.2f}, "
          f"{short} with a single variant", flush=True)


if __name__ == "__main__":
    main()
