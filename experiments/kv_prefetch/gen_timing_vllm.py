"""K0: batch-1 prefill / decode latency of each K0 prompt on vLLM (run in .venv-vllm).

Per request, sequentially (no other load): (a) full prompt, max_tokens=1 -> prefill (TTFT);
(b) full prompt, greedy until newline / EOS / 64 tokens -> prefill + decode; (c) question-only prompt,
max_tokens=1 -> what is left to prefill if the path segments' KV were ready. Prefix caching off, so
every call does a full prefill. Output: one json line per request with the three times.

  CUDA_VISIBLE_DEVICES=0 .venv-vllm/bin/python experiments/kv_prefetch/gen_timing_vllm.py \
      --model Qwen/Qwen3-8B --prompts outputs/kv_prefetch/prompts_Qwen3-8B.jsonl --out ...
"""
import argparse, json, time

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--prompts", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tp", type=int, default=1)
ap.add_argument("--gpu_mem", type=float, default=0.8)
ap.add_argument("--max_new", type=int, default=64)
a = ap.parse_args()

rows = [json.loads(l) for l in open(a.prompts)]
llm = LLM(a.model, tensor_parallel_size=a.tp, gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=False,
          max_model_len=8192, seed=0)
one = SamplingParams(temperature=0, max_tokens=1)
gen = SamplingParams(temperature=0, max_tokens=a.max_new, stop=["\n"])


def timed(ids, sp):
    t0 = time.perf_counter()
    o = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)[0].outputs[0]
    return (time.perf_counter() - t0) * 1000, o


for r in rows[:20]:  # warm-up (CUDA graphs, allocator)
    timed(r["full_ids"], gen)
with open(a.out, "w") as f:
    for i, r in enumerate(rows):
        pf, _ = timed(r["full_ids"], one)
        tot, o = timed(r["full_ids"], gen)
        qo, _ = timed(r["qonly_ids"], one)
        text = o.text.strip()
        hit = any(str(x).lower() in text.lower() for x in r["answer"]) if r["answer"] else False
        f.write(json.dumps(dict(dataset=r["dataset"], idx=r["idx"], hop=r["hop"], prompt_len=len(r["full_ids"]),
                                qonly_len=len(r["qonly_ids"]), seg_tokens=r["seg_tokens"], prefill_ms=pf,
                                total_ms=tot, qonly_prefill_ms=qo, out_tokens=len(o.token_ids), hit=hit,
                                text=text)) + "\n")
        if i % 200 == 0:
            print(f"{i}/{len(rows)} len {len(r['full_ids'])} prefill {pf:.1f} total {tot:.1f} ms", flush=True)
print("done", a.out)
