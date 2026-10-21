"""K0-v2 data: multi-hop RAG benchmarks as plain jsonl (run in the main .venv, which has `datasets`).

For HotpotQA (distractor), MuSiQue (answerable + unanswerable dev) and 2WikiMultihopQA: the first N dev questions and
the HippoRAG-style corpus = every paragraph attached to those N questions (deduplicated). Also a large corpus = all
paragraphs of train + dev of the three datasets, to time retrieval over a million-scale datastore.

  python experiments/kv_prefetch/rag_prep.py --n 1000 --out outputs/kv_prefetch/rag
Output: <out>/<ds>/questions.jsonl {qid, question, answers, gold_pids}, <out>/<ds>/corpus.jsonl {pid, title, text},
        <out>/large/corpus.jsonl
"""
import argparse
import json
import os

from datasets import load_dataset

SETS = {"hotpotqa": ("hotpotqa/hotpot_qa", "distractor"), "musique": ("dgslibisey/MuSiQue", None),
        "2wiki": ("framolfese/2WikiMultihopQA", None)}


def paragraphs(name, r):
    if name == "musique":
        return [(p["title"], p["paragraph_text"], bool(p.get("is_supporting", False))) for p in r["paragraphs"]]
    c, sup = r["context"], set(r["supporting_facts"]["title"])
    return [(t, " ".join(s).strip(), t in sup) for t, s in zip(c["title"], c["sentences"])]


def answers(name, r):
    a = [r["answer"]]
    if name == "musique":
        a += list(r.get("answer_aliases") or [])
    return [x for x in a if x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default="outputs/kv_prefetch/rag")
    ap.add_argument("--no_large", action="store_true")
    a = ap.parse_args()
    big = {}
    for name, (hf, cfg) in SETS.items():
        ds = load_dataset(hf, cfg) if cfg else load_dataset(hf)
        dev = ds["validation"]
        os.makedirs(f"{a.out}/{name}", exist_ok=True)
        corpus, qs = {}, []
        for i in range(min(a.n, len(dev))):
            r = dev[i]
            gold = []
            for t, x, sup in paragraphs(name, r):
                pid = corpus.setdefault((t, x), len(corpus))
                if sup:
                    gold.append(pid)
            if name == "musique" and not r["answerable"]:
                continue                                  # keep the corpus (distractors), skip unanswerable questions
            qs.append(dict(qid=r["id"], question=r["question"], answers=answers(name, r), gold_pids=gold))
        with open(f"{a.out}/{name}/corpus.jsonl", "w") as f:
            for (t, x), pid in sorted(corpus.items(), key=lambda kv: kv[1]):
                f.write(json.dumps(dict(pid=pid, title=t, text=x)) + "\n")
        with open(f"{a.out}/{name}/questions.jsonl", "w") as f:
            for q in qs:
                f.write(json.dumps(q) + "\n")
        print(f"{name}: {len(qs)} questions, corpus {len(corpus)} passages", flush=True)
        if not a.no_large:
            for sp in ds:
                for r in ds[sp]:
                    for t, x, _ in paragraphs(name, r):
                        big.setdefault((t, x), None)
            print(f"  large corpus so far: {len(big)}", flush=True)
    if not a.no_large:
        os.makedirs(f"{a.out}/large", exist_ok=True)
        with open(f"{a.out}/large/corpus.jsonl", "w") as f:
            for pid, (t, x) in enumerate(big):
                f.write(json.dumps(dict(pid=pid, title=t, text=x)) + "\n")
        print(f"large: {len(big)} passages")


if __name__ == "__main__":
    main()
