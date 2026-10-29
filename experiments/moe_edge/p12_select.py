"""Multi-user plan 1.2: pick prefetch parameters on the calibration workload (W1) only.

For every (predictor, eviction) family, among the given policies, the one configuration with the smallest mean
regret over all (hardware, cache) settings, regret = stall / best stall of that family in that setting - 1.

  python experiments/moe_edge/p12_select.py outputs/moe/mu/p12/p12b_w1.json [more.json ...]
"""
import json
import sys
from collections import defaultdict


def family(policy):
    ev, pref = policy.split("/")
    return ev, pref.split(":")[1]


def select(rows, show=5):
    S = sorted({(r["hw"], r["cache_frac"]) for r in rows})
    st = {(r["hw"], r["cache_frac"], r["policy"]): r["stall_ms_per_tok"] for r in rows}
    fams = defaultdict(set)
    for r in rows:
        fams[family(r["policy"])].add(r["policy"])
    out = {}
    for fam, pols in sorted(fams.items()):
        best = {s: min(st[(s[0], s[1], p)] for p in pols if (s[0], s[1], p) in st) for s in S}
        scored = []
        for p in pols:
            if not all((s[0], s[1], p) in st for s in S):
                continue
            reg = [st[(s[0], s[1], p)] / best[s] - 1 for s in S]
            scored.append((sum(reg) / len(reg), max(reg), p, reg))
        scored.sort()
        out[fam] = scored[0][2]
        print(f"== {fam[1]} / {fam[0]}  (settings: " + ", ".join(f"{h} {f:.0%}" for h, f in S) + ")")
        for m, mx, p, reg in scored[:show]:
            print(f"   {p:42} mean {m:+.2%} max {mx:+.2%}  " + " ".join(f"{x:+.1%}" for x in reg))
    return out


if __name__ == "__main__":
    rows = sum((json.load(open(f)) for f in sys.argv[1:]), [])
    print(json.dumps({f"{k[1]}/{k[0]}": v for k, v in select(rows, show=0).items()}, indent=1))
