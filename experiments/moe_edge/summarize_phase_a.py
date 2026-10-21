"""Markdown tables of the phase-A outputs of one model (outputs/moe_edge/phase_a/<model>/).
  .venv-mac/bin/python experiments/moe_edge/summarize_phase_a.py OLMoE-1B-7B-0924-Instruct
"""
import glob, json, os, sys

m = sys.argv[1]
d = f"outputs/moe_edge/phase_a/{m}"
print(f"## {m}\n")
f = f"outputs/moe_edge/e1/{m}.json"
if os.path.exists(f):
    R = json.load(open(f))
    k = R["k"]; B = str(2 * k)
    print(f"### E1 (T0), recall@{B} and paired bootstrap (pt)\n")
    print("| workload | GLOBAL | RPP | KNN_T | BASE_ADMIT | RELBAG | RPP_R | KNN_R | OURS_ADMIT | OURS-BASE [95% CI] | RPP_R-RPP [95% CI] | KNN_R-KNN_T [95% CI] |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for w in ("W1-cwq", "W1-webqsp", "Wsyn-cwq", "Wsynr-cwq", "W2-cwq", "W2c-cwq"):
        if w not in R:
            continue
        r, b = R[w]["recall"], R[w]["bootstrap_diff"]
        cell = lambda key: "{:+.1f} [{:+.1f}, {:+.1f}]".format(*[x * 100 for x in b[key]]) if key in b else "–"
        print(f"| {w} | " + " | ".join(f"{r[n][B]:.3f}" if n in r else "–" for n in
              ("GLOBAL", "RPP", "KNN_T", "BASE_ADMIT", "RELBAG", "RPP_R", "KNN_R", "OURS_ADMIT"))
              + f" | {cell('OURS_ADMIT-BASE_ADMIT@' + B)} | {cell('RPP_R-RPP@' + B)} | {cell('KNN_R-KNN_T@' + B)} |")
    print()
for fn in sorted(glob.glob(f"{d}/tsim_*.json")):
    rows = json.load(open(fn))
    name = os.path.basename(fn)[5:-5]
    base = next((r for r in rows if r["policy"] == "sw-moeinf/la1/none"), None)
    print(f"### tsim {name} (C = {rows[0]['C']})\n")
    print("| policy | TTFT ms | TPOT ms | e2e ms | decode stall ms/req | vs sw-moeinf/la1 | first-8 stall ms | GB/req |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        rel = f"{r['stall_dec_ms_per_req'] / base['stall_dec_ms_per_req'] - 1:+.0%}" if base and base['stall_dec_ms_per_req'] > 0 else "–"
        print(f"| {r['policy']} | {r['ttft_ms']:.0f} | {r['tpot_ms']:.2f} | {r.get('e2e_ms', float('nan')):.0f} | {r['stall_dec_ms_per_req']:.1f} | {rel} | "
              f"{r['first8_stall_ms']:.1f} | {r['GB_loaded_per_req']:.1f} |")
    print()
