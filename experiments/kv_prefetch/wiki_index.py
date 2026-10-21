"""K0-v2, large-datastore setting: 21M-passage Wikipedia (facebook/wiki_dpr psgs_w100, DPR multiset embeddings),
the datastore of TeleRAG / RAG serving papers. Run in .venv-vllm (faiss-cpu, pyarrow).

  build     embeddings -> float32 memmap (64 GB); faiss IVF4096,Flat (inner product) trained on 300k vectors
  retrieve  per question of rag_prep.py's datasets: DPR question encoding (GPU) + CPU IVF search for several
            nprobe values and thread counts (+ optional GPU exact search); saves the top-20 passages
            (title, text) of nprobe=256 for the generation step (rag_bench.py gen, retriever "wiki")

  .venv-vllm/bin/python experiments/kv_prefetch/wiki_index.py build
  CUDA_VISIBLE_DEVICES=0 .venv-vllm/bin/python experiments/kv_prefetch/wiki_index.py retrieve --n 300
"""
import argparse
import glob
import json
import os
import time

import numpy as np

ROOT = "outputs/kv_prefetch/rag"
WD = f"{ROOT}/wiki"
HUB = os.path.expanduser("~/.cache/huggingface/hub")
D = 768


def parquet_files():
    return sorted(glob.glob(f"{HUB}/datasets--facebook--wiki_dpr/snapshots/*/data/psgs_w100/multiset/train-*.parquet"))


def cmd_build(a):
    import faiss
    import pyarrow.parquet as pq
    os.makedirs(WD, exist_ok=True)
    files = parquet_files()
    sizes = [pq.ParquetFile(f).metadata.num_rows for f in files]
    N = sum(sizes)
    emb_path = f"{WD}/emb.f32"
    if not os.path.exists(emb_path + ".done"):
        mm = np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float32, shape=(N, D))
        i0, t0 = 0, time.time()
        for f, n in zip(files, sizes):
            col = pq.read_table(f, columns=["embeddings"]).column("embeddings").combine_chunks()
            mm[i0:i0 + n] = np.asarray(col.values.to_numpy(zero_copy_only=False), np.float32).reshape(n, D)
            i0 += n
        mm.flush()
        del mm
        json.dump(dict(N=N, files=files, sizes=sizes), open(f"{WD}/meta.json", "w"))
        open(emb_path + ".done", "w").write("ok")
        print(f"embeddings {N} x {D} written in {time.time() - t0:.0f}s", flush=True)
    E = np.load(emb_path, mmap_mode="r")
    faiss.omp_set_num_threads(a.threads)
    idx_path = f"{WD}/ivf4096.faiss"
    if not os.path.exists(idx_path):
        # k-means and the assignment of all 21M vectors run on the GPU (faiss-cpu's training ran single-threaded);
        # faiss only builds the inverted lists (add_core with the precomputed assignment). Same index as
        # IndexIVFFlat(IndexFlatIP, 4096) trained by faiss: inner-product assignment, centroids = cluster means.
        import torch
        t0 = time.time()
        g = torch.Generator().manual_seed(0)
        samp = torch.from_numpy(np.ascontiguousarray(
            E[np.sort(np.random.default_rng(0).choice(len(E), 300000, replace=False))])).cuda()
        C = samp[torch.randperm(len(samp), generator=g)[:4096].cuda()].clone()
        for _ in range(20):
            a = torch.cat([(samp[i:i + 50000] @ C.T).argmax(1) for i in range(0, len(samp), 50000)])
            S = torch.zeros_like(C).index_add_(0, a, samp)
            n = torch.bincount(a, minlength=4096).clamp_min(1).unsqueeze(1).float()
            C = torch.where(n > 1, S / n, C)
        quant = faiss.IndexFlatIP(D)
        quant.add(C.cpu().numpy())
        ivf = faiss.IndexIVFFlat(quant, D, 4096, faiss.METRIC_INNER_PRODUCT)
        ivf.is_trained = True
        for i in range(0, len(E), 1_000_000):
            x = np.ascontiguousarray(E[i:i + 1_000_000])
            a = (torch.from_numpy(x).cuda() @ C.T).argmax(1).cpu().numpy().astype(np.int64)
            ivf.add_core(len(x), faiss.swig_ptr(x), None, faiss.swig_ptr(a))
        faiss.write_index(ivf, idx_path)
        print(f"IVF4096 built in {time.time() - t0:.0f}s, ntotal {ivf.ntotal}", flush=True)


def passage(meta, files_cache, gid):
    import pyarrow.parquet as pq
    i = gid
    for f, n in zip(meta["files"], meta["sizes"]):
        if i < n:
            if f not in files_cache:
                files_cache[f] = pq.read_table(f, columns=["title", "text"])
            t = files_cache[f]
            return dict(title=str(t.column("title")[i]), text=str(t.column("text")[i]))
        i -= n
    raise IndexError(gid)


def cmd_retrieve(a):
    import faiss
    import torch
    from transformers import AutoTokenizer, DPRQuestionEncoder
    meta = json.load(open(f"{WD}/meta.json"))
    ivf = faiss.read_index(f"{WD}/ivf4096.faiss")
    p = sorted(glob.glob(f"{HUB}/models--facebook--dpr-question_encoder-multiset-base/snapshots/*"))[0]
    tok = AutoTokenizer.from_pretrained(p)
    enc = DPRQuestionEncoder.from_pretrained(p, dtype=torch.float16).cuda().eval()
    Eg = None
    if a.gpu_exact:
        E = np.load(f"{WD}/emb.f32", mmap_mode="r")
        Eg = torch.empty((len(E), D), dtype=torch.float16, device="cuda")
        for i in range(0, len(E), 2_000_000):
            Eg[i:i + 2_000_000] = torch.from_numpy(np.ascontiguousarray(E[i:i + 2_000_000])).half().cuda()
    files_cache = {}
    for name in ("hotpotqa", "musique", "2wiki"):
        qs = [json.loads(l) for l in open(f"{ROOT}/{name}/questions.jsonl")][:a.n]
        out = open(f"{ROOT}/{name}/retrieval_wiki.jsonl", "w")
        for qi, q in enumerate(qs[:5] + qs):                          # 5 warm-up queries
            T = {}
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.no_grad():
                b = tok(q["question"], return_tensors="pt", truncation=True, max_length=64).to("cuda")
                qe = enc(**b).pooler_output.float()
            torch.cuda.synchronize(); T["encode"] = time.perf_counter() - t0
            qc = qe.cpu().numpy()
            ids = None
            for th in a.threads:
                faiss.omp_set_num_threads(th)
                for npb in a.nprobe:
                    ivf.nprobe = npb
                    t0 = time.perf_counter()
                    _, I = ivf.search(qc, 20)
                    T[f"ivf_np{npb}_t{th}"] = time.perf_counter() - t0
                    if npb == max(a.nprobe):
                        ids = I[0]
            if Eg is not None:
                torch.cuda.synchronize(); t0 = time.perf_counter()
                torch.topk((Eg @ qe.half().T).view(-1), 20)
                torch.cuda.synchronize(); T["gpu_exact"] = time.perf_counter() - t0
            if qi < 5:
                continue
            ps = [passage(meta, files_cache, int(g)) for g in ids]
            out.write(json.dumps(dict(qid=q["qid"], ms={k: v * 1e3 for k, v in T.items()}, wiki=list(range(20)),
                                      wiki_passages=ps)) + "\n")
        out.close()
        R = [json.loads(l) for l in open(f"{ROOT}/{name}/retrieval_wiki.jsonl")]
        print(name, " ".join(f"{k} {np.median([r['ms'][k] for r in R]):.1f}ms" for k in R[0]["ms"]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "retrieve"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--threads", type=int, nargs="+", default=[16])
    ap.add_argument("--nprobe", type=int, nargs="+", default=[32, 128, 256])
    ap.add_argument("--gpu_exact", action="store_true")
    a = ap.parse_args()
    if a.cmd == "build":
        a.threads = a.threads[0] if isinstance(a.threads, list) else a.threads
        a.threads = max(a.threads, 32)
        cmd_build(a)
    else:
        cmd_retrieve(a)


if __name__ == "__main__":
    main()
