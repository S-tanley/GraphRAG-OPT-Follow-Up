"""K0: turn retrieval_timeline.jsonl into token-id prompts for one generator model.

Prompt = the GraphRAG-Opt few-shot M0 prompt (one line per retrieved path, all paths up to the
retrieval cap) inside the model's chat template with thinking off -- the format used for the MoE
traces. Also stores the same prompt with no paths (question-only, the part left to prefill if every
path segment's KV were ready when retrieval ends) and each path line's token count.

  python experiments/kv_prefetch/build_prompts.py --model Qwen/Qwen3-8B \
      --timeline outputs/kv_prefetch/retrieval_timeline.jsonl --out outputs/kv_prefetch/prompts_Qwen3-8B.jsonl
"""
import argparse, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "experiments", "moe"))
from collect_traces import build_prompt, line_units  # noqa: E402
from collect_traces_v2 import wrap_chat               # noqa: E402
from transformers import AutoTokenizer                # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--timeline", default="outputs/kv_prefetch/retrieval_timeline.jsonl")
ap.add_argument("--out", required=True)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model)
with open(a.out, "w") as f:
    for line in open(a.timeline):
        r = json.loads(line)
        paths = [[tuple(t) for t in p] for p in r["paths"]]
        full, _ = wrap_chat(tok, *build_prompt(tok, r["question"], paths, "M0", len(paths)))
        qonly, _ = wrap_chat(tok, *build_prompt(tok, r["question"], [], "M0", 0))
        seg = [len(line_units(tok, p)[0]) for p in paths if p]
        f.write(json.dumps(dict(dataset=r["dataset"], idx=r["idx"], hop=r["hop"], answer=r["answer"],
                                full_ids=full, qonly_ids=qonly, seg_tokens=seg)) + "\n")
print("wrote", a.out)
