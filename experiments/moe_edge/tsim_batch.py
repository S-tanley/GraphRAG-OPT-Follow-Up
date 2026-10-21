"""Time-stepped simulator for scenario S2 / E2: segment-wise regrouping of decode batches with continuous
batching (decode side only, as in prefill/decode disaggregation).

N requests are decoding concurrently (a finished one is replaced from the queue); one engine step runs a
group of B of them for one token. Every S steps the active requests are re-partitioned into groups by the
predicted expert profile of their next segment; each group then runs S steps (KV stays in unified memory,
so switching groups costs nothing). Per step and layer the group needs the union of its members' experts
(memory-bound batch decode: each needed expert is read once per step). The expert cache and the
SSD loader are the ones of tsim.py (demand > lookahead > background), with one-layer-ahead prefetch of the
union of the members' lookahead predictions.

Grouping keys: random | static (groups formed at admission, kept) | temporal (previous segment's profile)
| oracle (actual next-segment profile) | cache (cache-affinity scheduling: in each round every request runs
one segment, and the next group is the B requests whose predicted next segment is most resident).

  python experiments/moe_edge/tsim_batch.py --trace outputs/moe_edge/traces/Qwen3-30B-A3B-4bit-cf_cwq_M0_cot_chat_cap8 \
      --hw qwen3_30b_4bit --E 128 --cache_frac 0.5 --N 32 --B 8 --S 16,64 --out out.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "moe"))
from e1_analysis import load_run  # noqa: E402
from segment_regroup import partition, profile  # noqa: E402
from tsim import HW, Sim  # noqa: E402


def run(reqs, key, N, B, S, hw, C, E, evict, seed=0):
    rng = np.random.default_rng(seed)
    L, k = reqs[0]["dec"].shape[0], reqs[0]["dec"].shape[-1]
    sim = Sim(L, E, k, C, hw)
    sim.evict_mode = evict
    queue = list(range(len(reqs)))
    rng.shuffle(queue)
    pos, active, static_groups = {}, [], []
    st = dict(t=0.0, toks=0, stall=0.0, union=0)

    def _run_group(g):
        for s in range(S):
            act = [r for r in g if pos[r] + s < reqs[r]["dec"].shape[1]]
            if not act:
                break
            st["toks"] += len(act)
            for l in range(L):
                t_gate = st["t"] + hw["nonexp_layer"] / hw["mem_bw"]
                if l + 1 < L:
                    la = [reqs[r]["la_dec"][l, pos[r] + s] for r in act if reqs[r]["la_dec"] is not None]
                    if la:
                        sim.issue((l + 1) * E + np.unique(np.concatenate(la)), 1, t_gate)
                u = np.unique(np.concatenate([reqs[r]["dec"][l, pos[r] + s] for r in act]))
                st["union"] += len(u)
                keys = l * E + u
                t_ready = sim.need(keys, t_gate)
                st["stall"] += t_ready - t_gate
                st["t"] = t_ready + len(u) * hw["eb"] / hw["mem_bw"]
                sim.freq[keys] += 1

    while queue or active:
        while len(active) < N and queue:
            r = queue.pop(0)
            active.append(r)
            pos[r] = 0
        if key == "static":
            static_groups = [[r for r in g if r in active] for g in static_groups]
            placed = {r for g in static_groups for r in g}
            new = [r for r in active if r not in placed]
            for g in static_groups:
                while len(g) < B and new:
                    g.append(new.pop(0))
            while new:
                static_groups.append(new[:B])
                new = new[B:]
            groups = [g for g in static_groups if g]
        else:
            vecs = None
            if key == "oracle":
                vecs = {r: profile(reqs[r]["dec"], pos[r], pos[r] + S, E) for r in active}
            elif key == "temporal":
                vecs = {r: profile(reqs[r]["dec"], pos[r] - S, pos[r], E) for r in active}
            groups = partition(active, B, vecs, rng) if key != "cache" else None
        if key == "cache":
            # cache-affinity scheduling: each round every active request runs one segment; the next group is
            # the B remaining requests whose predicted next segment (previous-segment profile) is most resident
            remaining, groups = list(active), []
            while remaining:
                res = sim.res.reshape(L, E).astype(np.float64)
                sc = []
                for r in remaining:
                    p = profile(reqs[r]["dec"], pos[r] - S, pos[r], E)
                    sc.append(float(p @ res.ravel()) if p is not None else 0.0)
                order = np.argsort(-np.array(sc))[:B]
                g = [remaining[i] for i in order]
                remaining = [r for r in remaining if r not in g]
                groups.append(g)
                _run_group(g)
        for g in (groups if key != "cache" else []):
            _run_group(g)
        for r in active:
            pos[r] += S
        active = [r for r in active if pos[r] < reqs[r]["dec"].shape[1]]
    return dict(key=key, N=N, B=B, S=S, C=C, evict=evict, decode_tok_s=st["toks"] / st["t"],
                stall_frac=st["stall"] / st["t"], experts_per_layer_token=st["union"] / st["toks"] / L,
                GB_loaded_per_token=sim.stats["loaded"] * hw["eb"] / 1e9 / st["toks"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--hw", required=True)
    ap.add_argument("--E", type=int, required=True)
    ap.add_argument("--cache_frac", type=float, default=0.5)
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--S", default="16,64")
    ap.add_argument("--keys", default="random,static,temporal,cache,oracle")
    ap.add_argument("--evict", default="lru")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    hw = HW[args.hw]
    reqs = [r for r in load_run(args.trace)[:args.n] if r["dec"].shape[1] > 0]
    L = reqs[0]["dec"].shape[0]
    C = int(args.cache_frac * L * args.E)
    print(f"{args.trace}: {len(reqs)} requests, mean decode {np.mean([r['dec'].shape[1] for r in reqs]):.0f} steps, "
          f"cache {args.cache_frac:.0%} ({C} experts), N={args.N} B={args.B}", flush=True)
    rows = []
    for S in [int(x) for x in args.S.split(",")]:
        base = None
        for key in args.keys.split(","):
            r = run(reqs, key, args.N, args.B, S, hw, C, args.E, args.evict)
            rows.append(r)
            base = base or r
            print(f"  S={S:3d} {key:9s} decode {r['decode_tok_s']:7.1f} tok/s ({r['decode_tok_s'] / base['decode_tok_s'] - 1:+.0%})"
                  f"  experts/layer/token {r['experts_per_layer_token']:.2f}  stall {r['stall_frac']:.0%}"
                  f"  loaded {r['GB_loaded_per_token']:.3f} GB/token", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(rows, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
