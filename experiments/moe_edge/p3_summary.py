"""Tables for anchor P3 (tsim_ahead.py output): decode stall per token by policy, per hardware x cache size.

  python experiments/moe_edge/p3_summary.py outputs/moe/p3_w1.json [outputs/moe/p3_cot.json ...]
"gap closed" = (stall of la1 - stall) / (stall of la1 - stall of ORACLE with the same H): the share of what
perfect H-layer-ahead knowledge could remove over the one-layer-ahead gate that a predictor actually removes.
"""
import json
import re
import sys
from collections import defaultdict

ORDER = ["lru/none", "moeinf/none", "moeinf/la1"]


def key(p):
    if p in ORDER:
        return (0, ORDER.index(p), "", 0)
    m = re.match(r"moeinf/ah(\d+):(\w+)", p)
    rank = {"GATE": 1, "TOKGATE": 2, "AV": 3, "AVG": 4, "MIX": 5, "ORACLE": 6}
    return (1, rank.get(m.group(2), 9), p, int(m.group(1)))


for f in sys.argv[1:]:
    rows = json.load(open(f))
    by = defaultdict(dict)
    for r in rows:
        by[(r["hw"], r["cache_frac"])][r["policy"]] = r
    print(f"\n#### {f}  ({rows[0]['n']} requests)")
    for (hw, frac), d in sorted(by.items()):
        la1 = d["moeinf/la1"]["stall_ms_per_tok"]
        print(f"\n{hw}, cache {frac:.0%}\n")
        print("| policy | stall ms/tok | TPOT ms | decode loads/tok | wasted | gap closed |")
        print("|---|---|---|---|---|---|")
        for p in sorted(d, key=key):
            r = d[p]
            m = re.match(r"moeinf/ah(\d+):", p)
            gap = ""
            if m and "ORACLE" not in p:
                orc = d.get(f"moeinf/ah{m.group(1)}:ORACLE:m8")
                if orc and la1 - orc["stall_ms_per_tok"] > 1e-9:
                    gap = f"{(la1 - r['stall_ms_per_tok']) / (la1 - orc['stall_ms_per_tok']):.0%}"
            print(f"| {p} | {r['stall_ms_per_tok']:.2f} | {r['tpot_ms']:.2f} | {r['loads_per_tok']:.1f} | "
                  f"{r['wasted_frac']:.0%} | {gap} |")
