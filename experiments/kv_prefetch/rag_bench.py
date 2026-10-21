"""K0-v2: retrieval vs prefill vs decode on text multi-hop RAG (HotpotQA, MuSiQue, 2Wiki), batch 1.
Run in .venv-vllm (vLLM 0.11, faiss-cpu); data from rag_prep.py.

  embed     bge-large-en-v1.5 embeddings of each corpus (fp16 .npy)
  retrieve  per question: query encoding + search on
              small corpus (the HippoRAG-style per-dataset corpus, 6-12k passages): GPU exact, CPU exact
              large corpus (1.04M passages of all three datasets): GPU exact, CPU faiss flat, CPU faiss IVF
            + bge-reranker-v2-m3 over the dense top-50 (small corpus). Saves the top-20 ids of dense and rerank.
  gen       vLLM, one request at a time. For each (retriever, k, prompt layout); retriever "wiki" uses the
            21M-passage results of wiki_index.py:
              prefill_ms (full prompt, 1 token), total_ms (greedy answer), rest_ms (prefill of only the tokens that
              would remain after retrieval if every passage's KV were ready: question + tail for q-last, tail only
              for q-first, whose prefix+question can be prefilled while retrieval runs), EM / F1.
            ircot: IRCoT-style interleaving (LLM thought <= 64 tokens -> dense retrieval with the thought, <= 3
              rounds), with per-step retrieval / LLM times, then the final answer.

  CUDA_VISIBLE_DEVICES=0,1 .venv-vllm/bin/python experiments/kv_prefetch/rag_bench.py embed
  CUDA_VISIBLE_DEVICES=0,1 .venv-vllm/bin/python experiments/kv_prefetch/rag_bench.py retrieve --n 500
  CUDA_VISIBLE_DEVICES=0,1 .venv-vllm/bin/python experiments/kv_prefetch/rag_bench.py gen --model <snapshot> --n 300
"""
import argparse
import json
import os
import re
import string
import time
from collections import Counter

import numpy as np
import torch

ROOT = "outputs/kv_prefetch/rag"
DATASETS = ["hotpotqa", "musique", "2wiki"]
QPREF = "Represent this sentence for searching relevant passages: "
HUB = os.path.expanduser("~/.cache/huggingface/hub")


def snap(repo):
    d = f"{HUB}/models--{repo.replace('/', '--')}/snapshots"
    return os.path.join(d, sorted(os.listdir(d))[0])


def jl(path):
    return [json.loads(l) for l in open(path)]


def sync(dev):
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize(dev)


# ----------------------------------------------------------------------------- encoder / reranker
class Encoder:
    def __init__(self, dev):
        from transformers import AutoModel, AutoTokenizer
        p = snap("BAAI/bge-large-en-v1.5")
        self.tok = AutoTokenizer.from_pretrained(p)
        self.m = AutoModel.from_pretrained(p, torch_dtype=torch.float16).to(dev).eval()
        self.dev = dev

    @torch.no_grad()
    def __call__(self, texts, maxlen=256):
        b = self.tok(texts, padding=True, truncation=True, max_length=maxlen, return_tensors="pt").to(self.dev)
        h = self.m(**b).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(h.float(), dim=-1)


class Reranker:
    def __init__(self, dev):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        p = snap("BAAI/bge-reranker-v2-m3")
        self.tok = AutoTokenizer.from_pretrained(p)
        self.m = AutoModelForSequenceClassification.from_pretrained(p, torch_dtype=torch.float16).to(dev).eval()
        self.dev = dev

    @torch.no_grad()
    def __call__(self, q, passages):
        b = self.tok([[q, x] for x in passages], padding=True, truncation=True, max_length=512,
                     return_tensors="pt").to(self.dev)
        return self.m(**b).logits.view(-1).float()


def ptext(p):
    return f"{p['title']}: {p['text']}"


def cmd_embed(a):
    enc = Encoder("cuda:0")
    for name in DATASETS + ["large"]:
        out = f"{ROOT}/{name}/emb.npy"
        if os.path.exists(out):
            continue
        corpus = jl(f"{ROOT}/{name}/corpus.jsonl")
        texts = [ptext(p) for p in corpus]
        order = np.argsort([len(t) for t in texts])           # length-sorted batches
        E = np.zeros((len(texts), 1024), np.float16)
        t0 = time.time()
        for i in range(0, len(texts), 512):
            idx = order[i:i + 512]
            E[idx] = enc([texts[j] for j in idx]).half().cpu().numpy()
        np.save(out, E)
        print(f"{name}: {len(texts)} passages embedded in {time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------- retrieval timing
def cmd_retrieve(a):
    import faiss
    faiss.omp_set_num_threads(a.cpu_threads)
    torch.set_num_threads(a.cpu_threads)
    dev = "cuda:0"
    enc, rr = Encoder(dev), Reranker(dev)
    big = np.load(f"{ROOT}/large/emb.npy")
    big_gpu = torch.from_numpy(big).to(dev)
    flat = faiss.IndexFlatIP(1024)
    flat.add(big.astype(np.float32))
    ivf_path = f"{ROOT}/large/ivf4096.faiss"
    if os.path.exists(ivf_path):
        ivf = faiss.read_index(ivf_path)
    else:
        q = faiss.IndexFlatIP(1024)
        ivf = faiss.IndexIVFFlat(q, 1024, 4096, faiss.METRIC_INNER_PRODUCT)
        samp = big[np.random.default_rng(0).choice(len(big), 200000, replace=False)].astype(np.float32)
        ivf.train(samp)
        ivf.add(big.astype(np.float32))
        faiss.write_index(ivf, ivf_path)
    ivf.nprobe = a.nprobe
    for name in DATASETS:
        corpus = jl(f"{ROOT}/{name}/corpus.jsonl")
        texts = [ptext(p) for p in corpus]
        E = np.load(f"{ROOT}/{name}/emb.npy")
        Eg, Ec = torch.from_numpy(E).to(dev), E.astype(np.float32)
        qs = jl(f"{ROOT}/{name}/questions.jsonl")[:a.n]
        out = open(f"{ROOT}/{name}/retrieval.jsonl", "w")
        for i, q in enumerate(qs + qs[:10]):                   # first 10 again at the end; first pass is warm-up
            T = {}
            sync(dev); t0 = time.perf_counter()
            qe = enc([QPREF + q["question"]], maxlen=64)
            sync(dev); T["encode"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            s = (Eg @ qe.half().T).view(-1)
            top_small = torch.topk(s, 50).indices.cpu().numpy()
            T["small_gpu"] = time.perf_counter() - t0
            qc = qe.cpu().numpy()
            t0 = time.perf_counter()
            np.argpartition(-(Ec @ qc[0]), 50)[:50]
            T["small_cpu"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            torch.topk((big_gpu @ qe.half().T).view(-1), 50).indices.cpu()
            T["large_gpu"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            flat.search(qc, 50)
            T["large_cpu_flat"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            ivf.search(qc, 50)
            T["large_cpu_ivf"] = time.perf_counter() - t0
            sync(dev); t0 = time.perf_counter()
            sc = rr(q["question"], [texts[j] for j in top_small])
            order = top_small[torch.argsort(sc, descending=True).cpu().numpy()]
            sync(dev); T["rerank50"] = time.perf_counter() - t0
            if i < 10:
                continue
            gold = set(q["gold_pids"])
            rec = {f"recall@{k}_{m}": len(gold & set(map(int, ids[:k]))) / max(1, len(gold))
                   for m, ids in (("dense", top_small), ("rerank", order)) for k in (5, 10)}
            out.write(json.dumps(dict(qid=q["qid"], ms={k: v * 1e3 for k, v in T.items()},
                                      dense=list(map(int, top_small[:20])), rerank=list(map(int, order[:20])),
                                      **rec)) + "\n")
        out.close()
        R = jl(f"{ROOT}/{name}/retrieval.jsonl")
        ms = {k: np.median([r["ms"][k] for r in R]) for k in R[0]["ms"]}
        rc = {k: np.mean([r[k] for r in R]) for k in R[0] if k.startswith("recall")}
        print(name, " ".join(f"{k} {v:.2f}ms" for k, v in ms.items()), "|",
              " ".join(f"{k} {v:.2f}" for k, v in rc.items()), flush=True)


# ----------------------------------------------------------------------------- generation timing
def norm_ans(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em_f1(pred, golds):
    p = norm_ans(pred)
    em = max(float(p == norm_ans(g)) for g in golds)
    f1 = 0.0
    for g in golds:
        gt, pt = norm_ans(g).split(), p.split()
        c = sum((Counter(gt) & Counter(pt)).values())
        if c:
            pr, rc = c / len(pt), c / len(gt)
            f1 = max(f1, 2 * pr * rc / (pr + rc))
    return em, f1


INSTR = "Answer the question based on the passages. Reply with only the short answer (a few words), no explanation."


class Prompter:
    """Token ids of the chat prompt split into the pieces whose KV could be ready at different times."""

    def __init__(self, tok):
        self.tok = tok
        mark = "@@U@@"
        try:
            full = tok.apply_chat_template([{"role": "user", "content": mark}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            full = tok.apply_chat_template([{"role": "user", "content": mark}], tokenize=False,
                                           add_generation_prompt=True)
        self.head, self.tail = full.split(mark)

    def enc(self, s):
        return self.tok.encode(s, add_special_tokens=False)

    def build(self, question, passages, layout):
        """Returns (ids, early, segs, rest): early = ids known before retrieval (prefix, + question for q-first);
        segs = per-passage ids; rest = ids that remain to prefill once early and segs are cached."""
        ps = [self.enc(f"[{i + 1}] {ptext(p)}\n") for i, p in enumerate(passages)]
        if layout == "qfirst":
            early = self.enc(self.head + INSTR + f"\nQuestion: {question}\n\nPassages:\n")
            rest = self.enc("\nAnswer:" + self.tail)
        else:
            early = self.enc(self.head + INSTR + "\n\nPassages:\n")
            rest = self.enc(f"\nQuestion: {question}\nAnswer:" + self.tail)
        ids = early + [t for s in ps for t in s] + rest
        return ids, early, ps, rest


def cmd_gen(a):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    llm = LLM(a.model, tensor_parallel_size=a.tp, gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=False,
              max_model_len=12288, seed=0)
    tok = llm.get_tokenizer()
    pr = Prompter(tok)
    mname = os.path.basename(os.path.dirname(os.path.dirname(a.model.rstrip("/")))).replace("models--", "")
    one = SamplingParams(temperature=0, max_tokens=1)
    ans = SamplingParams(temperature=0, max_tokens=32, stop=["\n"])
    thought = SamplingParams(temperature=0, max_tokens=64, stop=["\n"])

    def run(ids, sp):
        t0 = time.perf_counter()
        o = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)[0].outputs[0]
        return (time.perf_counter() - t0) * 1e3, o

    enc = Encoder("cuda:1") if a.ircot and torch.cuda.device_count() > 1 and a.tp == 1 else None
    if a.ircot and enc is None:
        enc = Encoder("cuda:0")
    for name in DATASETS:
        corpus = jl(f"{ROOT}/{name}/corpus.jsonl")
        qs = {q["qid"]: q for q in jl(f"{ROOT}/{name}/questions.jsonl")}
        R = jl(f"{ROOT}/{name}/retrieval.jsonl")[:a.n]
        os.makedirs(f"{ROOT}/{name}/gen", exist_ok=True)
        for r in R[:5]:                                         # warm-up
            run(pr.build(qs[r["qid"]]["question"], [corpus[j] for j in r["dense"][:5]], "qlast")[0], ans)
        wiki = {}
        if os.path.exists(f"{ROOT}/{name}/retrieval_wiki.jsonl"):
            wiki = {r["qid"]: r["wiki_passages"] for r in jl(f"{ROOT}/{name}/retrieval_wiki.jsonl")}
        for ret, k, layout in [tuple(x.split(":")) for x in a.configs.split(",")]:
            k = int(k)
            path = f"{ROOT}/{name}/gen/{mname}_{ret}_k{k}_{layout}.jsonl"
            if os.path.exists(path) or (ret == "wiki" and not wiki):
                continue
            with open(path, "w") as f:
                for r in R:
                    q = qs[r["qid"]]
                    if ret == "wiki" and r["qid"] not in wiki:
                        continue
                    ps = wiki[r["qid"]][:k] if ret == "wiki" else [corpus[j] for j in r[ret][:k]]
                    ids, early, segs, rest = pr.build(q["question"], ps, layout)
                    pf, _ = run(ids, one)
                    tot, o = run(ids, ans)
                    rs, _ = run(rest, one)
                    eg, _ = run(early, one)
                    em, f1 = em_f1(o.text.strip(), q["answers"])
                    f.write(json.dumps(dict(qid=r["qid"], len=len(ids), early_len=len(early),
                                            seg_lens=[len(s) for s in segs], rest_len=len(rest), prefill_ms=pf,
                                            total_ms=tot, rest_ms=rs, early_ms=eg, out_tokens=len(o.token_ids),
                                            em=em, f1=f1, pred=o.text.strip())) + "\n")
            G = jl(path)
            print(f"{name:9} {mname} {ret}:k{k}:{layout:6} len {np.mean([g['len'] for g in G]):6.0f} "
                  f"prefill {np.median([g['prefill_ms'] for g in G]):6.1f} decode "
                  f"{np.median([g['total_ms'] - g['prefill_ms'] for g in G]):6.1f} rest {np.median([g['rest_ms'] for g in G]):5.1f} ms"
                  f" | EM {np.mean([g['em'] for g in G]):.3f} F1 {np.mean([g['f1'] for g in G]):.3f}", flush=True)
        if a.ircot:
            path = f"{ROOT}/{name}/gen/{mname}_ircot.jsonl"
            if os.path.exists(path):
                continue
            E = torch.from_numpy(np.load(f"{ROOT}/{name}/emb.npy")).to(enc.dev)
            with open(path, "w") as f:
                for r in R:
                    q = qs[r["qid"]]
                    steps, got, thoughts = [], list(r["dense"][:4]), []
                    for step in range(3):
                        body = pr.build(q["question"], [corpus[j] for j in got], "qlast")[0]
                        cue = pr.enc("Thought so far: " + " ".join(thoughts) + "\nNext thought (one sentence; "
                                     "if you know the answer, write 'So the answer is: X'):" + pr.tail)
                        ids = body[:-len(pr.enc(pr.tail))] + cue
                        tl, o = run(ids, thought)
                        th = o.text.strip()
                        thoughts.append(th)
                        if "answer is" in th.lower():
                            steps.append(dict(llm_ms=tl, retr_ms=0.0, len=len(ids)))
                            break
                        sync(enc.dev); t0 = time.perf_counter()
                        qe = enc([QPREF + th], maxlen=64)
                        top = torch.topk((E @ qe.half().T).view(-1), 8).indices.cpu().tolist()
                        sync(enc.dev); rt = (time.perf_counter() - t0) * 1e3
                        got += [j for j in top if j not in got][:2]
                        steps.append(dict(llm_ms=tl, retr_ms=rt, len=len(ids)))
                    ids = pr.build(q["question"], [corpus[j] for j in got], "qlast")[0]
                    pf, _ = run(ids, one)
                    tot, o = run(ids, ans)
                    em, f1 = em_f1(o.text.strip(), q["answers"])
                    f.write(json.dumps(dict(qid=r["qid"], steps=steps, final_len=len(ids), final_prefill_ms=pf,
                                            final_total_ms=tot, em=em, f1=f1, pred=o.text.strip())) + "\n")
            G = jl(path)
            print(f"{name:9} {mname} ircot: steps {np.mean([len(g['steps']) for g in G]):.2f} "
                  f"LLM {np.median([sum(s['llm_ms'] for s in g['steps']) for g in G]):6.1f} retr "
                  f"{np.median([sum(s['retr_ms'] for s in g['steps']) for g in G]):5.1f} final "
                  f"{np.median([g['final_total_ms'] for g in G]):6.1f} ms | EM {np.mean([g['em'] for g in G]):.3f} "
                  f"F1 {np.mean([g['f1'] for g in G]):.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["embed", "retrieve", "gen"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--cpu_threads", type=int, default=16)
    ap.add_argument("--nprobe", type=int, default=64)
    ap.add_argument("--model", default=None)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu_mem", type=float, default=0.5)
    ap.add_argument("--configs", default="dense:5:qlast,dense:5:qfirst,dense:10:qlast,dense:10:qfirst,"
                                          "dense:20:qlast,rerank:5:qlast,rerank:5:qfirst")
    ap.add_argument("--ircot", action="store_true")
    a = ap.parse_args()
    {"embed": cmd_embed, "retrieve": cmd_retrieve, "gen": cmd_gen}[a.cmd](a)


if __name__ == "__main__":
    main()
