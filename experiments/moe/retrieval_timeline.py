"""MoE plan Phase 0.5 (Q0): how long does each stage of RoG-plan retrieval take, i.e. how early
is each piece of retrieval information available before prefill can start?

Replicates graph_utils.get_paths_rog_plan stage by stage (same planner prompt, beam search,
greedy rule parsing, bfs_with_rule grounding, prefix fallback) and times:
  adj      build the question's adjacency from its subgraph triples  (a real KG system keeps this
           prebuilt, so it is reported separately and excluded from the window)
  plan     RoG planner generation (GPU)          -> after it: relation schema known
  parse    map generated text to KG relations
  ground   bfs_with_rule grounding + fallbacks    -> during it: candidate entities appear
  prompt   build the reader prompt (M0 and T)    -> after it: prefill can start

  python experiments/moe/retrieval_timeline.py --dataset rmanluo/RoG-cwq --n 200 --out ...
"""

import argparse
import json
import os
import statistics as st
import sys
import time
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "graphrag-main"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "common"))
from graph_utils import _bfs_with_rule, _greedy_parse_rule   # noqa: E402
from faithful_pfkv import build_prompt                      # noqa: E402
from conditions import trie_text_lines                      # noqa: E402
from table2_worker import load_ds                           # noqa: E402

INSTR = "Please generate a valid relation path that can be helpful for answering the following question: "


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_retrieval(planner, tok, q_entities, graph_triples, question, hop, max_paths, num_beam):
    t = {}
    t0 = time.perf_counter()
    adj, rels = defaultdict(list), set()
    for h, r, tt in graph_triples:
        r = r.strip()
        adj[h].append((r, tt))
        adj[tt].append((r, h))
        rels.add(r)
    rel_sorted = sorted(rels, key=len, reverse=True)
    t["adj"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    ids = tok(f"<s>[INST] {INSTR}{question} [/INST]", return_tensors="pt").to(planner.device)
    with torch.no_grad():
        out = planner.generate(**ids, max_new_tokens=80, num_beams=num_beam, num_return_sequences=num_beam,
                               do_sample=False, pad_token_id=tok.eos_token_id)
    sync()
    t["plan"] = time.perf_counter() - t1
    gen_len = out.shape[1] - ids["input_ids"].shape[1]

    t2 = time.perf_counter()
    rules = []
    for j in range(out.shape[0]):
        gen = tok.decode(out[j, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        rule = _greedy_parse_rule(gen, rel_sorted, hop)
        if rule:
            rules.append(rule)
    t["parse"] = time.perf_counter() - t2

    t3 = time.perf_counter()
    paths, seen, first_path_at = [], set(), None

    def add(p):
        nonlocal first_path_at
        key = tuple(p)
        if p and key not in seen:
            seen.add(key)
            paths.append(p)
            if first_path_at is None:
                first_path_at = time.perf_counter() - t3

    done = False
    for rule in rules:
        for s in q_entities:
            for p in _bfs_with_rule(adj, s, rule):
                add(p)
        if len(paths) >= max_paths:
            done = True
            break
    if not done and len(paths) < max_paths:
        for rule in rules:
            for L in range(min(len(rule), hop) - 1, 0, -1):
                for s in q_entities:
                    for p in _bfs_with_rule(adj, s, rule[:L]):
                        add(p)
            if len(paths) >= max_paths:
                break
    paths = paths[:max_paths]
    t["ground"] = time.perf_counter() - t3
    t["first_path"] = first_path_at if first_path_at is not None else t["ground"]
    return paths, rules, t, gen_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rmanluo/RoG-cwq")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--hop", type=int, default=3)
    ap.add_argument("--num_beam", type=int, default=4)
    ap.add_argument("--caps", default="8,30")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ds = load_ds(args.dataset).select(range(args.n))
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True)
    planner = AutoModelForCausalLM.from_pretrained("rmanluo/RoG", dtype=torch.float16).to("cuda").eval()
    # warm-up
    timed_retrieval(planner, tok, ds[0]["q_entity"], ds[0]["graph"], ds[0]["question"], args.hop, 8, args.num_beam)

    rows = []
    for cap in [int(c) for c in args.caps.split(",")]:
        for i in range(len(ds)):
            ex = ds[i]
            paths, rules, t, gen_len = timed_retrieval(planner, tok, ex["q_entity"], ex["graph"], ex["question"],
                                                       args.hop, cap, args.num_beam)
            tp = time.perf_counter()
            full, _, _, _ = build_prompt(tok, ex["question"], paths, cap, "rog")
            tok.encode("".join(trie_text_lines(paths, cap)), add_special_tokens=False)
            t["prompt"] = time.perf_counter() - tp
            rows.append(dict(cap=cap, n_triples=len(ex["graph"]), n_paths=len(paths), n_rules=len(rules),
                             gen_len=gen_len, prompt_tokens=len(full), **t))
        print(f"cap {cap}: done {len(ds)}", flush=True)
    json.dump(rows, open(args.out, "w"))

    def q(v, p):
        v = sorted(v)
        return v[min(len(v) - 1, int(p * len(v)))]
    for cap in sorted({r["cap"] for r in rows}):
        R = [r for r in rows if r["cap"] == cap]
        print(f"\n== {args.dataset} cap={cap} n={len(R)} (ms: mean / p50 / p90)")
        for k in ["adj", "plan", "parse", "ground", "first_path", "prompt"]:
            v = [1000 * r[k] for r in R]
            print(f"  {k:10s} {st.mean(v):8.1f} {q(v, .5):8.1f} {q(v, .9):8.1f}")
        win_q = [1000 * (r["plan"] + r["parse"] + r["ground"] + r["prompt"]) for r in R]
        win_s = [1000 * (r["parse"] + r["ground"] + r["prompt"]) for r in R]
        print(f"  window from question arrival (excl. adj): {st.mean(win_q):.1f} / {q(win_q, .5):.1f} / {q(win_q, .9):.1f}")
        print(f"  window from schema known:                 {st.mean(win_s):.1f} / {q(win_s, .5):.1f} / {q(win_s, .9):.1f}")
        print(f"  planner generated tokens (mean): {st.mean(r['gen_len'] for r in R):.1f}; "
              f"subgraph triples (mean): {st.mean(r['n_triples'] for r in R):.0f}; prompt tokens (mean): {st.mean(r['prompt_tokens'] for r in R):.0f}")


if __name__ == "__main__":
    main()
