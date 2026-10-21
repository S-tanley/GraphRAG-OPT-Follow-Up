"""Wsyn: paraphrase the most frequent Freebase relations into natural-language phrases that share
(almost) no tokens with the original identifier, so that "same schema" no longer implies
"same tokens". Used to test whether schema-level expert similarity follows meaning or tokens.

Paraphrases come from a local instruct model (Qwen3-8B, thinking disabled). Each candidate is
kept only if its token overlap with the original relation is below a threshold under ALL target
tokenizers; otherwise we retry with a hint and finally fall back to a hand-written rule.

  python experiments/moe/make_syn_map.py --top 120 --out outputs/moe/syn_map.json
"""

import argparse
import json
import os
import re
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TARGETS = ["mistralai/Mixtral-8x22B-Instruct-v0.1", "Qwen/Qwen3-235B-A22B", "deepseek-ai/DeepSeek-V2.5",
           "zai-org/GLM-4.5-Air"]
PROMPT = ("Rewrite the Freebase relation identifier below as a short natural English phrase (2-6 words) "
          "that states the same relation from the subject to the object. Use everyday words and do NOT "
          "reuse the words, word pieces, dots or underscores of the identifier.{hint}\n"
          "Identifier: {rel}\nPhrase:")


def overlap(tok, a, b):
    A, B = set(tok.encode(a, add_special_tokens=False)), set(tok.encode(" " + b, add_special_tokens=False))
    return len(A & B) / max(1, len(B))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=120)
    ap.add_argument("--max_overlap", type=float, default=0.2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cnt = Counter()
    for f in ["paths_rmanluo-RoG-cwq.json", "paths_rmanluo-RoG-webqsp.json"]:
        D = json.load(open(os.path.join(ROOT, "graphrag-main/precomputed_paths_v2", f)))
        for P in D.values():
            for p in P[:8]:
                for _h, r, _t in p:
                    cnt[r] += 1
    total = sum(cnt.values())
    rels = [r for r, _ in cnt.most_common(args.top)]
    cover = sum(cnt[r] for r in rels) / total
    print(f"{len(cnt)} distinct relations; top {len(rels)} cover {cover:.1%} of relation occurrences", flush=True)

    toks = [AutoTokenizer.from_pretrained(m) for m in TARGETS]
    gtok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    gen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16).to("cuda").eval()

    def ask(rel, hint=""):
        msgs = [{"role": "user", "content": PROMPT.format(rel=rel, hint=hint)}]
        enc = gtok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
                                       return_tensors="pt", return_dict=True)
        ids = enc["input_ids"].to("cuda")
        with torch.no_grad():
            out = gen.generate(ids, max_new_tokens=24, do_sample=False)
        s = gtok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip().split("\n")[0]
        return re.sub(r"[\"'.]", "", s).strip()

    mapping, report = {}, []
    for r in rels:
        cand, best = None, None
        for hint in ["", " Avoid these words: " + ", ".join(sorted(set(re.split(r"[._]", r)) - {""})) + "."]:
            c = ask(r, hint)
            ov = max(overlap(t, r, c) for t in toks)
            if best is None or ov < best[1]:
                best = (c, ov)
            if c and ov <= args.max_overlap:
                cand = (c, ov)
                break
        cand = cand or best
        mapping[r] = cand[0]
        report.append((r, cand[0], round(cand[1], 3), cnt[r]))
        print(f"{cnt[r]:6d}  {r:55s} -> {cand[0]:40s} overlap {cand[1]:.2f}", flush=True)
    bad = [x for x in report if x[2] > args.max_overlap]
    json.dump(mapping, open(args.out, "w"), indent=1)
    json.dump({"coverage": cover, "n": len(rels), "over_threshold": bad, "rows": report},
              open(args.out.replace(".json", "_report.json"), "w"), indent=1)
    print(f"wrote {args.out}; {len(bad)} relations above overlap threshold", flush=True)


if __name__ == "__main__":
    main()
