"""Cross-model headline tables for the phase-A report (markdown).
  .venv-mac/bin/python experiments/moe_edge/report_tables.py OLMoE-1B-7B-0924-Instruct Qwen3-30B-A3B-4bit-cf DeepSeek-V2-Lite-Chat
"""
import json
import os
import sys

TR = "outputs/moe_edge/traces"
PA = "outputs/moe_edge/phase_a"
models = sys.argv[1:]
NONRET = lambda p: not any(x in p for x in ("OURS", "RPP_R", "KNN_R", "CAND", "ORACLE"))   # no retrieval / oracle


def jl(f):
    return json.load(open(f)) if os.path.exists(f) else None


print("### E0: reader accuracy (Hit@1)\n")
print("| model | CWQ M0 | CWQ Direct | WEB M0 | WEB Direct | Wtrain | W2c (mean / median decode steps) |")
print("|---|---|---|---|---|---|---|")
for m in models:
    g = lambda t: (jl(f"{TR}/{m}_{t}.json") or {}).get("hit")
    w2 = jl(f"{TR}/{m}_cwq_M0_cot_chat_cap8.json")
    import numpy as np
    st = [t["steps"] for t in w2["timing"]] if w2 else []
    f = lambda x: "–" if x is None else f"{x:.3f}"
    print(f"| {m} | {f(g('cwq_M0_cap8'))} | {f(g('cwq_direct_cap8'))} | {f(g('webqsp_M0_cap8'))} | "
          f"{f(g('webqsp_direct_cap8'))} | {f(g('train_M0_cap8'))} | "
          + (f"{np.mean(st):.0f} / {np.median(st):.0f}" if st else "–") + " |")

print("\n### E1 at admission (T0): recall@2k, paired bootstrap of OURS_ADMIT − BASE_ADMIT (pt, 95% CI)\n")
print("| model | workload | GLOBAL | KNN_T | BASE_ADMIT (best token) | RPP_R | KNN_R | OURS_ADMIT | OURS − BASE |")
print("|---|---|---|---|---|---|---|---|---|")
for m in models:
    R = jl(f"outputs/moe_edge/e1/{m}.json")
    if not R:
        continue
    B = str(2 * R["k"])
    for w in ("W1-cwq", "W1-webqsp", "W2c-cwq"):
        if w not in R:
            continue
        r, b = R[w]["recall"], R[w]["bootstrap_diff"].get(f"OURS_ADMIT-BASE_ADMIT@{B}")
        print(f"| {m} | {w} | " + " | ".join(f"{r[n][B]:.3f}" for n in ("GLOBAL", "KNN_T", "BASE_ADMIT", "RPP_R", "KNN_R", "OURS_ADMIT"))
              + (" | {:+.1f} [{:+.1f}, {:+.1f}] |".format(*[x * 100 for x in b]) if b else " | – |"))

print("\n### E1 after prefill (T1, decode steps >= 1): recall@2k\n")
print("| model | workload | GLOBAL | FIRSTTOK (no retrieval) | CANDFIRST | OURS_ADMIT (T0) | ORACLE | CANDFIRST − FIRSTTOK |")
print("|---|---|---|---|---|---|---|---|")
for m in models:
    P = jl(f"{PA}/{m}/e1_phases.json")
    if not P:
        continue
    for w in ("W1-cwq", "W1-webqsp"):
        if w not in P:
            continue
        r = P[w]["recall"]
        k2 = sorted(r["GLOBAL"], key=int)[1]
        b = P[w]["bootstrap"].get(f"CANDFIRST-FIRSTTOK@{k2}")
        print(f"| {m} | {w} | " + " | ".join(f"{r[n][k2]:.3f}" if n in r else "–" for n in ("GLOBAL", "FIRSTTOK", "CANDFIRST", "OURS_ADMIT", "ORACLE"))
              + (" | {:+.1f} [{:+.1f}, {:+.1f}] |".format(*[x * 100 for x in b]) if b else " | – |"))

print("\n### E2a: regrouping decode batches after prefill (W1-cwq, steps >= 1): share of the oracle gain\n")
print("| model | B | oracle: fewer loads/token | BASE_ADMIT | OURS_ADMIT | FIRSTTOK | CANDFIRST | BASE+FIRSTTOK | OURS+FIRSTTOK |")
print("|---|---|---|---|---|---|---|---|---|")
for m in models:
    for f in (f"{PA}/{m}/regroup_pred_skip0.json", f"{PA}/{m}/regroup_pred.json"):
        G = jl(f)
        if G:
            break
    if not G:
        continue
    for row in G["rows"]:
        base, orc = row["random"], row["oracle (actual decode)"]
        sh = lambda n: f"{(base - row[n]) / (base - orc):.0%}" if n in row and base != orc else "–"
        print(f"| {m} | {row['B']} | {1 - orc / base:.0%} | {sh('BASE_ADMIT')} | {sh('OURS_ADMIT')} | {sh('FIRSTTOK (no retrieval)')} | "
              f"{sh('CANDFIRST (new)')} | {sh('BASE_ADMIT+FIRSTTOK (no retrieval)')} | {sh('OURS_ADMIT+FIRSTTOK')} |")

print("\n### E2b: segment regrouping of long decodes (W2c, N=32, B=8): fewer expert loads per token vs random\n")
print("| model | S | oracle | previous-segment profile | static groups |")
print("|---|---|---|---|---|")
for m in models:
    S = jl(f"{PA}/{m}/segment_regroup.json")
    for row in S or []:
        print(f"| {m} | {row['S']} | {1 - row['oracle'] / row['random']:+.1%} | {1 - row['temporal'] / row['random']:+.1%} | "
              f"{1 - row['static'] / row['random']:+.1%} |")

print("\n### E3/E4 (S1 single user): decode stall and end-to-end latency, best non-retrieval policy vs retrieval vs oracle\n")
print("| model | workload | cache | best non-retrieval: decode stall ms/req (policy) | best retrieval prior | T0 ORACLE | e2e: best non-retr. → best retr. | TTFT: lfu-prefill vs moeinf-prefill |")
print("|---|---|---|---|---|---|---|---|")
for m in models:
    for w in ("W1-cwq", "W1-webqsp", "W2"):
        for cf in ("0.25", "0.5", "0.75"):
            rows = jl(f"{PA}/{m}/tsim_{w}_{cf}.json")
            if not rows:
                continue
            rows = rows + (jl(f"{PA}/{m}/tsim_extra_{w}_{cf}.json") or [])
            nr = [r for r in rows if NONRET(r["policy"])]
            rt = [r for r in rows if not NONRET(r["policy"]) and "ORACLE" not in r["policy"]]
            orc = [r for r in rows if "T0:ORACLE" in r["policy"]]
            b = min(nr, key=lambda r: r["stall_dec_ms_per_req"])
            o = min(rt, key=lambda r: r["stall_dec_ms_per_req"]) if rt else None
            be = min(nr, key=lambda r: r.get("e2e_ms", 1e18))
            oe = min(rt, key=lambda r: r.get("e2e_ms", 1e18)) if rt else None
            oc = min(orc, key=lambda r: r["stall_dec_ms_per_req"]) if orc else None
            t_sw = next((r["ttft_ms"] for r in rows if r["policy"] == "sw-moeinf/la1/none"), None)
            t_mi = next((r["ttft_ms"] for r in rows if r["policy"] == "moeinf/none/none"), None)
            print(f"| {m} | {w} | {cf} | {b['stall_dec_ms_per_req']:.1f} ({b['policy']}) | "
                  + (f"{o['stall_dec_ms_per_req']:.1f} ({o['stall_dec_ms_per_req'] / b['stall_dec_ms_per_req'] - 1:+.0%}, {o['policy']})" if o else "–")
                  + (f" | {oc['stall_dec_ms_per_req']:.1f} ({oc['stall_dec_ms_per_req'] / b['stall_dec_ms_per_req'] - 1:+.0%})" if oc else " | –")
                  + (f" | {be['e2e_ms']:.0f} → {oe['e2e_ms']:.0f} ({oe['e2e_ms'] / be['e2e_ms'] - 1:+.1%})" if oe and "e2e_ms" in be else " | –")
                  + (f" | {t_sw:.0f} vs {t_mi:.0f} ms" if t_sw and t_mi else " | –") + " |")
