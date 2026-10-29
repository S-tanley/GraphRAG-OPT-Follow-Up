"""Multi-user simulation: how much does overlapping KV computation with retrieval help when many requests share
one GPU (continuous batching) and a pool of retrieval workers?

Requests arrive as a Poisson process. Each request: retrieval on one of W CPU workers (service time and the
arrival times of its segments -- KG paths -- from the measured traces), then prefill and a short decode on the GPU.
GPU: iterations of a continuous-batching engine; iteration time = base + c_pf * prefill tokens + c_dec * decoding
sequences (calibrated on vLLM / H200 and MLX / Mac measurements). Each iteration takes all decoding sequences,
then post-retrieval prefill (FIFO, chunked to a token budget), then -- only with what is left of the budget and
at most lp_cap tokens -- low-priority precompute work.

Policies
  P0  retrieve, then prefill the whole prompt
  P1  + prefix and question prefilled as soon as the request arrives (question-first prompt), low priority
  P2  + every segment (KG path / passage) prefilled as soon as it arrives, low priority; after retrieval only the
      tail, any segment not yet precomputed, and a recompute fraction r of the precomputed segment tokens remain
  P3  P2 + cross-request segment KV pool: a segment is already cached with probability h (no compute at all)
Text RAG with a reranker: dense candidates are known after t_dense; P2/P3 precompute the dense top-m candidates
during the rerank (on the same GPU); final passages outside them are prefilled after retrieval.

  python experiments/kv_prefetch/mu_sim.py --scenario kg_server --qps 2 4 8 12 --out out.json
"""
import argparse
import heapq
import json
import os
from collections import deque

import numpy as np

HW = {
    # vLLM on H200, Qwen3-8B: ~4.7 ms per decode iteration at small batch, 23 us per prefill token (batch-1 fits)
    "h200_8b": dict(base=4.5, c_pf=0.023, c_dec=0.03, budget=2048),
    # MLX on M5 Max, Qwen3-8B-4bit: ~8 ms per decode token, 0.33-0.36 ms per prefill token
    "mac_8b": dict(base=8.0, c_pf=0.34, c_dec=0.4, budget=1024),
    # vLLM TP=2 on 2x H200, Llama-3.1-70B: ~20 ms per decode iteration, 137 us per prefill token
    "h200_70b": dict(base=20.0, c_pf=0.137, c_dec=0.1, budget=2048),
    # MLX on M5 Max, Qwen3-30B-A3B-4bit (MoE): ~6 ms per decode token, ~0.23 ms per prefill token
    "mac_30b": dict(base=6.0, c_pf=0.23, c_dec=0.3, budget=1024),
}


# ----------------------------------------------------------------------------- workloads
def kg_requests(timeline, prompts, hops=(4,), datasets=None):
    P = {(r["dataset"], r["idx"], r["hop"]): r for r in map(json.loads, open(prompts))}
    out = []
    for r in map(json.loads, open(timeline)):
        if r["hop"] not in hops or (datasets and r["dataset"] not in datasets):
            continue
        p = P.get((r["dataset"], r["idx"], r["hop"]))
        if p is None:
            continue
        seg = p["seg_tokens"]
        arr = r["arrivals_ms"][:len(seg)]
        if len(arr) < len(seg):
            seg = seg[:len(arr)]
        early = len(p["qonly_ids"])
        tail = 10
        full = len(p["full_ids"])
        out.append(dict(retr=r["retrieval_ms"], seg_t=arr, seg_n=seg, early=early, tail=tail,
                        other=max(0, full - early - sum(seg) - tail), dec=8, rerank=0.0, cand=None))
    return out


def text_requests(root, ds, gen_file, retr_key, rerank, m_cand=10, rerank_sep=False):
    R = {r["qid"]: r for r in map(json.loads, open(f"{root}/{ds}/retrieval.jsonl"))}
    W = {}
    if os.path.exists(f"{root}/{ds}/retrieval_wiki.jsonl"):
        W = {r["qid"]: r for r in map(json.loads, open(f"{root}/{ds}/retrieval_wiki.jsonl"))}
    out = []
    for g in map(json.loads, open(f"{root}/{ds}/gen/{gen_file}")):
        r = R[g["qid"]]
        seg = g["seg_lens"]
        if retr_key == "rerank":
            t_dense = r["ms"]["encode"] + r["ms"]["small_gpu"]
            final, dense = set(r["rerank"][:len(seg)]), r["dense"][:m_cand]
            hit = [j in set(dense) for j in r["rerank"][:len(seg)]]
            cand = dict(t=t_dense, n=m_cand, avg_tok=float(np.mean(seg)), hit=hit)
            if rerank_sep:                       # reranker on its own device: part of the retrieval latency
                retr, rr = t_dense + r["ms"]["rerank50"], 0.0
            else:                                # reranker on the LLM's GPU: high-priority GPU work
                retr, rr = t_dense, r["ms"]["rerank50"]
        else:
            w = W.get(g["qid"])
            if w is None:
                continue
            retr, rr, cand = w["ms"]["encode"] + w["ms"][retr_key], 0.0, None
        out.append(dict(retr=retr, seg_t=[retr] * len(seg), seg_n=seg, early=g["early_len"], tail=g["rest_len"],
                        other=max(0, g["len"] - g["early_len"] - sum(seg) - g["rest_len"]), dec=max(2, g["out_tokens"]),
                        rerank=rr, cand=cand))
    return out


# ----------------------------------------------------------------------------- simulator
class Req:
    __slots__ = ("id", "spec", "t_arr", "t_rstart", "t_rdone", "hp_left", "lp_items", "pre_done", "dec_left",
                 "t_first", "state", "seg_done", "rerank_left")


def simulate(specs, policy, hw, qps, workers, n_req, r_recomp=0.15, hit_rate=0.0, lp_cap=512, seed=0):
    rng = np.random.default_rng(seed)
    t_arr = np.cumsum(rng.exponential(1000.0 / qps, n_req))
    reqs = []
    for i in range(n_req):
        q = Req()
        q.id, q.spec, q.t_arr = i, specs[int(rng.integers(len(specs)))], float(t_arr[i])
        q.state, q.hp_left, q.dec_left, q.t_first, q.t_rdone = "wait_cpu", 0, 0, None, None
        q.seg_done = [False] * len(q.spec["seg_n"])
        q.pre_done = False
        q.rerank_left = 0.0
        q.lp_items = deque()
        reqs.append(q)
    events = [(q.t_arr, 0, "arrive", q.id) for q in reqs]
    heapq.heapify(events)
    seq = 1
    cpu_free, cpu_q = workers, deque()
    hp_q, lp_q, decoding = deque(), deque(), []
    t, gpu_busy_until, busy_time = 0.0, 0.0, 0.0
    lp_tokens_done = 0

    def push(time, kind, rid):
        nonlocal seq
        seq += 1
        heapq.heappush(events, (time, seq, kind, rid))

    def start_retrieval(q, now):
        nonlocal cpu_free
        cpu_free -= 1
        q.t_rstart = now
        s = q.spec
        if policy in ("P2", "P3"):
            for j, (ta, n) in enumerate(zip(s["seg_t"], s["seg_n"])):
                if s["cand"] is None:
                    push(now + ta, "seg", q.id * 1000 + j)
            if s["cand"] is not None:
                push(now + s["cand"]["t"], "cand", q.id)
        push(now + s["retr"], "rdone", q.id)

    def finish_retrieval(q, now):
        s = q.spec
        q.t_rdone = now
        if s["rerank"] > 0:                                   # reranker runs on the GPU (high priority work)
            q.rerank_left = s["rerank"]
            q.state = "rerank"
            hp_q.append(q)
            return
        ready_prefill(q, now)

    def ready_prefill(q, now):
        s = q.spec
        segs = sum(s["seg_n"])
        if policy == "P0":
            left = s["early"] + segs + s["other"] + s["tail"]
        elif policy == "P1":
            left = (0 if q.pre_done else s["early"]) + segs + s["other"] + s["tail"]
        else:
            if s["cand"] is not None:
                done = [h and q.seg_done[0] for h in s["cand"]["hit"]]   # candidates precomputed as a block
            else:
                done = q.seg_done
            pre = sum(n for n, d in zip(s["seg_n"], done) if d)
            left = (0 if q.pre_done else s["early"]) + (segs - pre) + s["other"] + s["tail"] + r_recomp * pre
            # drop any low-priority work still queued for this request
            q.lp_items.clear()
        q.hp_left = int(round(left))
        q.state = "prefill"
        hp_q.append(q)

    def add_lp(q, kind, tokens, j=None):
        if tokens <= 0:
            if kind == "pre":
                q.pre_done = True
            elif kind == "seg":
                q.seg_done[j] = True
            return
        item = [q, kind, tokens, j]
        q.lp_items.append(item)
        lp_q.append(item)

    done_ttft, n_done = [], 0
    while events or hp_q or decoding or (lp_q and any(it[0].state in ("wait_cpu", "retrieving", "rerank") for it in lp_q)):
        # process all events up to the GPU's next free time
        next_ev = events[0][0] if events else float("inf")
        if t < gpu_busy_until:
            t = min(gpu_busy_until, next_ev) if next_ev < gpu_busy_until else gpu_busy_until
        while events and events[0][0] <= t + 1e-9:
            et, _, kind, rid = heapq.heappop(events)
            if kind == "arrive":
                q = reqs[rid]
                if policy != "P0":
                    add_lp(q, "pre", q.spec["early"])
                if cpu_free > 0:
                    q.state = "retrieving"
                    start_retrieval(q, et)
                else:
                    cpu_q.append(q)
            elif kind == "seg":
                q = reqs[rid // 1000]
                j = rid % 1000
                if q.state == "retrieving":
                    if policy == "P3" and rng.random() < hit_rate:
                        q.seg_done[j] = True
                    else:
                        add_lp(q, "seg", q.spec["seg_n"][j], j)
            elif kind == "cand":
                q = reqs[rid]
                if q.state in ("retrieving", "rerank"):
                    c = q.spec["cand"]
                    add_lp(q, "seg", int(c["n"] * c["avg_tok"] * (1 - (hit_rate if policy == "P3" else 0))), 0)
            elif kind == "rdone":
                q = reqs[rid]
                cpu_free += 1
                finish_retrieval(q, et)
                if cpu_q:
                    nq = cpu_q.popleft()
                    nq.state = "retrieving"
                    start_retrieval(nq, et)
        if t < gpu_busy_until - 1e-9:
            continue
        # ---- one GPU iteration
        budget = hw["budget"] - len(decoding)
        pf_tokens, work_ms, finished_prefill = 0, 0.0, []
        for q in list(hp_q):
            if budget <= 0:
                break
            if q.state == "rerank":
                work_ms += q.rerank_left
                q.rerank_left = 0
                hp_q.remove(q)
                ready_prefill(q, t)
                continue
            take = min(q.hp_left, budget)
            q.hp_left -= take
            budget -= take
            pf_tokens += take
            if q.hp_left <= 0:
                hp_q.remove(q)
                finished_prefill.append(q)
        lp_budget = min(budget, lp_cap)
        lp_used = 0
        while lp_q and lp_budget > 0:
            item = lp_q[0]
            q, kind, tok, j = item
            if q.state not in ("wait_cpu", "retrieving", "rerank") or item not in q.lp_items:
                lp_q.popleft()
                continue
            take = min(tok, lp_budget)
            item[2] -= take
            lp_budget -= take
            lp_used += take
            if item[2] <= 0:
                lp_q.popleft()
                q.lp_items.remove(item)
                if kind == "pre":
                    q.pre_done = True
                else:
                    q.seg_done[j] = True
        if not (pf_tokens or lp_used or decoding or work_ms):
            # idle: jump to the next event
            if events:
                t = max(t, events[0][0])
                continue
            break
        dt = hw["base"] + hw["c_pf"] * (pf_tokens + lp_used) + hw["c_dec"] * len(decoding) + work_ms
        lp_tokens_done += lp_used
        busy_time += dt
        t_end = t + dt
        for q in decoding[:]:
            q.dec_left -= 1
            if q.dec_left <= 0:
                decoding.remove(q)
                q.state = "done"
                n_done += 1
        for q in finished_prefill:
            q.t_first = t_end
            done_ttft.append(t_end - q.t_arr)
            q.dec_left = q.spec["dec"] - 1
            q.state = "decode"
            if q.dec_left > 0:
                decoding.append(q)
            else:
                n_done += 1
        gpu_busy_until = t_end
        t = t_end
    tt = np.array(done_ttft[n_req // 10:])                   # drop warm-up
    span = t - reqs[n_req // 10].t_arr
    return dict(policy=policy, qps=qps, ttft_p50=float(np.percentile(tt, 50)), ttft_p90=float(np.percentile(tt, 90)),
                ttft_p99=float(np.percentile(tt, 99)), ttft_mean=float(tt.mean()), gpu_util=busy_time / t,
                lp_tokens_per_req=lp_tokens_done / n_req, completed_per_s=len(tt) / (span / 1000))


SCEN = {
    "kg_server": dict(hw="h200_8b", workers=16,
                      build=lambda a: kg_requests(f"{a.root}/retrieval_timeline.jsonl", f"{a.root}/prompts_mac_Qwen3-8B.jsonl")),
    "kg_mac": dict(hw="mac_8b", workers=8,
                   build=lambda a: kg_requests(f"{a.root}/retrieval_timeline_mac.jsonl", f"{a.root}/prompts_mac_Qwen3-8B.jsonl")),
    "kg_server_70b": dict(hw="h200_70b", workers=16,
                          build=lambda a: kg_requests(f"{a.root}/retrieval_timeline.jsonl", f"{a.root}/prompts_mac_Qwen3-8B.jsonl")),
    "kg_mac_30b": dict(hw="mac_30b", workers=8,
                       build=lambda a: kg_requests(f"{a.root}/retrieval_timeline_mac.jsonl", f"{a.root}/prompts_mac_Qwen3-8B.jsonl")),
    "rerank_server": dict(hw="h200_8b", workers=64,
                          build=lambda a: sum((text_requests(f"{a.root}/rag", d, "Qwen--Qwen3-8B_rerank_k5_qfirst.jsonl", "rerank", True)
                                               for d in ("hotpotqa", "musique", "2wiki")), [])),
    "rerank_sep_server": dict(hw="h200_8b", workers=64,
                              build=lambda a: sum((text_requests(f"{a.root}/rag", d, "Qwen--Qwen3-8B_rerank_k5_qfirst.jsonl", "rerank", True, rerank_sep=True)
                                                   for d in ("hotpotqa", "musique", "2wiki")), [])),
    "wiki_server": dict(hw="h200_8b", workers=6,
                        build=lambda a: sum((text_requests(f"{a.root}/rag", d, "Qwen--Qwen3-8B_wiki_k5_qfirst.jsonl", "ivf_np128_t32", False)
                                             for d in ("hotpotqa", "2wiki")), [])),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=list(SCEN))
    ap.add_argument("--root", default="outputs/kv_prefetch")
    ap.add_argument("--qps", type=float, nargs="+", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--r", type=float, default=0.15, help="recompute fraction of precomputed segment tokens")
    ap.add_argument("--hit", type=float, default=0.35, help="P3 cross-request segment hit rate")
    ap.add_argument("--lp_cap", type=int, default=512)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sc = SCEN[a.scenario]
    specs = sc["build"](a)
    hw = HW[sc["hw"]]
    print(f"{a.scenario}: {len(specs)} request specs; mean retrieval {np.mean([s['retr'] for s in specs]):.1f} ms, "
          f"mean prompt {np.mean([s['early'] + sum(s['seg_n']) + s['other'] + s['tail'] for s in specs]):.0f} tokens", flush=True)
    rows = []
    for qps in a.qps:
        base = None
        for pol in ("P0", "P1", "P2", "P3"):
            r = simulate(specs, pol, hw, qps, sc["workers"], a.n, a.r, a.hit, a.lp_cap)
            r["scenario"] = a.scenario
            rows.append(r)
            base = base or r
            print(f"  qps {qps:5.1f} {pol}: TTFT p50 {r['ttft_p50']:7.1f} p90 {r['ttft_p90']:7.1f} p99 {r['ttft_p99']:7.1f} ms "
                  f"(p50 {r['ttft_p50'] / base['ttft_p50'] - 1:+.0%}, p99 {r['ttft_p99'] / base['ttft_p99'] - 1:+.0%})  "
                  f"GPU {r['gpu_util']:.0%}  done/s {r['completed_per_s']:.2f}", flush=True)
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
