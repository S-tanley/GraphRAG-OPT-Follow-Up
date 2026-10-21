"""K0 on the edge (Mac Studio, MLX): the same prompts and retrieval results as rag_bench.py on the H200, timed
on Apple silicon. Run in .venv-mac.

  gen       MLX model, batch 1: prefill of the full prompt, decode of the short answer (greedy, <= 32 tokens,
            stops at a newline), prefill of `rest` (what remains after retrieval if passage KVs are ready) and of
            `early` (prefix [+ question], computable during retrieval). Same Prompter / configs as rag_bench.py.
  retrieve  bge-large query encoding + exact search over the per-dataset corpus (MPS and CPU), bge-reranker-v2-m3
            over the dense top-50 (MPS). The LLM shares the GPU with these on a Mac.

  .venv-mac/bin/python experiments/kv_prefetch/rag_mac.py gen --model mlx-community/Qwen3-8B-4bit --n 100
  .venv-mac/bin/python experiments/kv_prefetch/rag_mac.py retrieve --n 200
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_bench import QPREF, Prompter, em_f1, jl, ptext  # noqa: E402

ROOT = "outputs/kv_prefetch/rag"
DATASETS = ["hotpotqa", "musique", "2wiki"]


def cmd_gen(a):
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    model, tok = load(a.model)
    pr = Prompter(tok._tokenizer if hasattr(tok, "_tokenizer") else tok)
    mname = a.model.split("/")[-1]
    eos = set(tok.eos_token_ids) if hasattr(tok, "eos_token_ids") else {tok.eos_token_id}

    def prefill(ids):
        cache = make_prompt_cache(model)
        t0 = time.perf_counter()
        logits = model(mx.array(ids)[None], cache=cache)
        mx.eval(logits)
        return (time.perf_counter() - t0) * 1e3, cache, logits

    def run(ids):
        pf, cache, logits = prefill(ids)
        t0 = time.perf_counter()
        out = []
        nxt = mx.argmax(logits[:, -1, :], axis=-1)
        for _ in range(32):
            t = int(nxt.item())
            if t in eos:
                break
            out.append(t)
            if "\n" in tok.decode(out):
                break
            logits = model(nxt[:, None], cache=cache)
            nxt = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(nxt)
        return pf, (time.perf_counter() - t0) * 1e3, tok.decode(out).strip().split("\n")[0]

    for name in DATASETS:
        corpus = jl(f"{ROOT}/{name}/corpus.jsonl")
        qs = {q["qid"]: q for q in jl(f"{ROOT}/{name}/questions.jsonl")}
        R = jl(f"{ROOT}/{name}/retrieval.jsonl")[:a.n]
        wiki = {}
        if os.path.exists(f"{ROOT}/{name}/retrieval_wiki.jsonl"):
            wiki = {r["qid"]: r["wiki_passages"] for r in jl(f"{ROOT}/{name}/retrieval_wiki.jsonl")}
        os.makedirs(f"{ROOT}/{name}/gen_mac", exist_ok=True)
        for r in R[:3]:                                             # warm-up
            run(pr.build(qs[r["qid"]]["question"], [corpus[j] for j in r["dense"][:5]], "qlast")[0])
        for ret, k, layout in [tuple(x.split(":")) for x in a.configs.split(",")]:
            k = int(k)
            if ret == "wiki" and not wiki:
                continue
            path = f"{ROOT}/{name}/gen_mac/{mname}_{ret}_k{k}_{layout}.jsonl"
            if os.path.exists(path):
                continue
            with open(path, "w") as f:
                for r in R:
                    q = qs[r["qid"]]
                    if ret == "wiki" and r["qid"] not in wiki:
                        continue
                    ps = wiki[r["qid"]][:k] if ret == "wiki" else [corpus[j] for j in r[ret][:k]]
                    ids, early, segs, rest = pr.build(q["question"], ps, layout)
                    pf, dec, pred = run(ids)
                    rs = prefill(rest)[0]
                    eg = prefill(early)[0]
                    em, f1 = em_f1(pred, q["answers"])
                    f.write(json.dumps(dict(qid=r["qid"], len=len(ids), early_len=len(early), rest_len=len(rest),
                                            prefill_ms=pf, decode_ms=dec, rest_ms=rs, early_ms=eg, em=em, f1=f1,
                                            pred=pred)) + "\n")
            G = jl(path)
            print(f"{name:9} {mname} {ret}:k{k}:{layout:6} len {np.mean([g['len'] for g in G]):6.0f} prefill "
                  f"{np.median([g['prefill_ms'] for g in G]):7.1f} decode {np.median([g['decode_ms'] for g in G]):6.1f} "
                  f"rest {np.median([g['rest_ms'] for g in G]):5.1f} early {np.median([g['early_ms'] for g in G]):5.1f} ms | "
                  f"EM {np.mean([g['em'] for g in G]):.3f}", flush=True)


def cmd_kg(a):
    """KG path prompts (build_prompts.py output): prefill of the full prompt, decode of the answer, prefill of the
    question-only prompt (what is left if every path segment's KV were ready)."""
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    model, tok = load(a.model)
    eos = set(tok.eos_token_ids) if hasattr(tok, "eos_token_ids") else {tok.eos_token_id}

    def prefill(ids):
        cache = make_prompt_cache(model)
        t0 = time.perf_counter()
        logits = model(mx.array(ids)[None], cache=cache)
        mx.eval(logits)
        return (time.perf_counter() - t0) * 1e3, cache, logits

    rows = [json.loads(l) for l in open(a.prompts)]
    rows = [r for r in rows if r["hop"] in (3, 4)]
    per = {}
    for r in rows:
        key = (r["dataset"], r["hop"])
        if len(per.setdefault(key, [])) < a.n:
            per[key].append(r)
    with open(a.out, "w") as f:
        for _, rs in sorted(per.items()):
            for r in rs[:2]:
                prefill(r["full_ids"])                            # warm-up
            for r in rs:
                pf, cache, logits = prefill(r["full_ids"])
                t0 = time.perf_counter()
                nxt, out = mx.argmax(logits[:, -1, :], axis=-1), []
                for _ in range(32):
                    t = int(nxt.item())
                    if t in eos:
                        break
                    out.append(t)
                    if "\n" in tok.decode(out):
                        break
                    logits = model(nxt[:, None], cache=cache)
                    nxt = mx.argmax(logits[:, -1, :], axis=-1)
                dec = (time.perf_counter() - t0) * 1e3
                qo = prefill(r["qonly_ids"])[0]
                f.write(json.dumps(dict(dataset=r["dataset"], idx=r["idx"], hop=r["hop"], prompt_len=len(r["full_ids"]),
                                        qonly_len=len(r["qonly_ids"]), prefill_ms=pf, decode_ms=dec,
                                        qonly_prefill_ms=qo)) + "\n")
    print("wrote", a.out)


def cmd_retrieve(a):
    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    dev = "mps"
    from huggingface_hub import snapshot_download
    p = snapshot_download("BAAI/bge-large-en-v1.5")
    etok = AutoTokenizer.from_pretrained(p)
    enc = AutoModel.from_pretrained(p, dtype=torch.float16).to(dev).eval()
    p = snapshot_download("BAAI/bge-reranker-v2-m3")
    rtok = AutoTokenizer.from_pretrained(p)
    rr = AutoModelForSequenceClassification.from_pretrained(p, dtype=torch.float16).to(dev).eval()

    @torch.no_grad()
    def encode(texts, maxlen):
        b = etok(texts, padding=True, truncation=True, max_length=maxlen, return_tensors="pt").to(dev)
        return torch.nn.functional.normalize(enc(**b).last_hidden_state[:, 0].float(), dim=-1)

    for name in DATASETS:
        corpus = jl(f"{ROOT}/{name}/corpus.jsonl")
        texts = [ptext(x) for x in corpus]
        emb_path = f"{ROOT}/{name}/emb_mac.npy"
        if not os.path.exists(emb_path):
            order = np.argsort([len(t) for t in texts])
            E = np.zeros((len(texts), 1024), np.float16)
            for i in range(0, len(texts), 128):
                idx = order[i:i + 128]
                E[idx] = encode([texts[j] for j in idx], 256).half().cpu().numpy()
            np.save(emb_path, E)
        torch.mps.empty_cache()                                     # corpus encoding leaves a large MPS cache
        E = np.load(emb_path)
        Eg, Ec = torch.from_numpy(E).to(dev), E.astype(np.float32)
        qs = jl(f"{ROOT}/{name}/questions.jsonl")[:a.n]
        rows = []
        for i, q in enumerate(qs[:5] + qs):
            T = {}
            torch.mps.synchronize(); t0 = time.perf_counter()
            qe = encode([QPREF + q["question"]], 64)
            torch.mps.synchronize(); T["encode"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            top = torch.topk((Eg @ qe.half().T).view(-1), 50).indices.cpu().numpy()
            T["search_mps"] = time.perf_counter() - t0
            qc = qe.cpu().numpy()[0]
            t0 = time.perf_counter()
            np.argpartition(-(Ec @ qc), 50)[:50]
            T["search_cpu"] = time.perf_counter() - t0
            torch.mps.synchronize(); t0 = time.perf_counter()
            with torch.no_grad():
                for c0 in (0, 25):                                  # two batches of 25 pairs
                    b = rtok([[q["question"], texts[j]] for j in top[c0:c0 + 25]], padding=True, truncation=True,
                             max_length=512, return_tensors="pt").to(dev)
                    rr(**b).logits.view(-1).float().cpu()
            torch.mps.synchronize(); T["rerank50"] = time.perf_counter() - t0
            if i >= 5:
                rows.append({k: v * 1e3 for k, v in T.items()})
        med = {k: float(np.median([r[k] for r in rows])) for k in rows[0]}
        json.dump(dict(n=len(rows), median_ms=med), open(f"{ROOT}/{name}/retrieval_mac.json", "w"), indent=1)
        print(name, " ".join(f"{k} {v:.1f}ms" for k, v in med.items()), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "retrieve", "kg"])
    ap.add_argument("--prompts", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default="mlx-community/Qwen3-8B-4bit")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--configs", default="dense:5:qlast,dense:5:qfirst,dense:10:qlast,dense:20:qlast,rerank:5:qlast,"
                                          "wiki:5:qlast")
    a = ap.parse_args()
    {"gen": cmd_gen, "retrieve": cmd_retrieve, "kg": cmd_kg}[a.cmd](a)


if __name__ == "__main__":
    main()
