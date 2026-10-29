"""Multi-user plan 3.3: the section-5 ablation table from the full sweep (tsim_mu --mode pool, p33c_full.json).

Rows (plan section 5), all with LFU as the flowing-region policy:
  base      sync + LFU                                   select fcfs,   no subst,  rho 0,  no prefetch, C2 off
  subst     + use-time substitution (<= 10%)             fcfs,          rate:0.1,  rho 0,  none,        off
  batch     + similarity batching (P-anchor, k=2)        anchor,        rate:0.1,  rho 0,  none,        off
  res       + resident (EMA, rho .5) / flowing           anchor,        rate:0.1,  rho .5, none,        off
  pref      + multi-layer prefetch (anchor+GATE, adapt.) anchor,        rate:0.1,  rho .5, ah8,         off
  full      + C2 protection                              anchor,        rate:0.1,  rho .5, ah8,         on
  -batch    full without batching                        fcfs,          rate:0.1,  rho .5, ah8,         on
  -subst    full without substitution                    anchor,        none,      rho .5, ah8,         on
Throughput is reported relative to `subst` (the H6 reference "sync + LFU + substitution only") at the same
(hw, cache, N, B); H6 asks full >= +15% at the same p99.

H6 at the same p99 (section 4): among the full-stack family (prefetch, resident, C2, subst; select fcfs or anchor with
any wait bound k), the best throughput whose p99 is at most 2% above the reference's, (a) at the same B (B limited, e.g. by
KV memory) and (b) over all B <= N (the frontier; offload favours the largest B).

  python experiments/moe_edge/p33_tables.py outputs/moe/mu/p33/p33c_full.json
  python experiments/moe_edge/p33_tables.py outputs/moe/mu/p33/p33d_fcfs.json outputs/moe/mu/p33/p33d_anchor.json --B 16 32
"""
import argparse
import json
from collections import defaultdict

import numpy as np

PF = "lfu/ah8:AVG:m8:p0.3:g0.5:kl0.15"
ROWS = [("base", "lfu/none", "fcfs", "none", 0.0, False),
        ("subst", "lfu/none", "fcfs", "rate:0.1", 0.0, False),
        ("batch", "lfu/none", "anchor", "rate:0.1", 0.0, False),
        ("res", "lfu/none", "anchor", "rate:0.1", 0.5, False),
        ("pref", PF, "anchor", "rate:0.1", 0.5, False),
        ("full", PF, "anchor", "rate:0.1", 0.5, True),
        ("-batch", PF, "fcfs", "rate:0.1", 0.5, True),
        ("-subst", PF, "anchor", "none", 0.5, True)]


TOL = 1.02                       # "same p99": at most 2% above the reference (simulation noise of a 30 s horizon)


def gmean(x):
    return float(np.exp(np.mean(np.log(x))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--B", type=int, nargs="+", default=None, help="ablation table: only these B")
    ap.add_argument("--klim", type=int, default=2, help="ablation table: wait bound of anchor batching")
    a = ap.parse_args()
    runs = [r for f in a.files for r in json.load(open(f))]
    warm = sorted({r.get("warm", "self") for r in runs})
    print(f"runs {len(runs)}, warm start {warm}\n")
    idx = {(r["hw"], r["cache_frac"], r["n"], r["B"], r["policy"], r["select"], r["subst"], r["rho"], r["protect"]): r
           for r in runs if r["select"] == "fcfs" or r["klim"] == a.klim}
    settings = sorted({k[:4] for k in idx if a.B is None or k[3] in a.B})
    get = {name: (lambda s, p=p, sel=sel, sub=sub, rho=rho, pr=pr: idx.get(s + (p, sel, sub, rho, pr)))
           for name, p, sel, sub, rho, pr in ROWS}
    # 1. per (hw, cache): geometric mean over N x B of throughput and p99 relative to `subst`; plus raw numbers
    print("## Ablation (geo-mean over N in {64,128} x B in {16,32}; ratios vs `subst` at the same setting)\n")
    for hw in sorted({s[0] for s in settings}):
        print(f"### {hw}\n")
        print("| row | cache | tok/s (abs, mean) | tput vs subst | p50 vs subst | p99 vs subst | changed | loads/tok "
              "| pf waste |")
        print("|---|---|---|---|---|---|---|---|---|")
        for f in sorted({s[1] for s in settings}):
            ss = [s for s in settings if s[:2] == (hw, f)]
            for name, *_ in ROWS:
                rr = [(get[name](s), get["subst"](s)) for s in ss]
                rr = [(a, b) for a, b in rr if a and b]
                if not rr:
                    continue
                t = gmean([a["throughput_tok_s"] / b["throughput_tok_s"] for a, b in rr])
                p50 = gmean([a["tpot_p50_ms"] / b["tpot_p50_ms"] for a, b in rr])
                p99 = gmean([a["tpot_p99_ms"] / b["tpot_p99_ms"] for a, b in rr])
                ch = np.mean([a.get("subst_change_rate", 0.0) for a, _ in rr])
                lt = np.mean([a["loads_per_token"] for a, _ in rr])
                w = np.mean([a["pf_waste_frac"] for a, _ in rr])
                ab = np.mean([a["throughput_tok_s"] for a, _ in rr])
                print(f"| {name} | {f:.2f} | {ab:.1f} | {100 * (t - 1):+.1f}% | {100 * (p50 - 1):+.1f}% | "
                      f"{100 * (p99 - 1):+.1f}% | {100 * ch:.1f}% | {lt:.2f} | {100 * w:.0f}% |")
        print()
    # 2. H6 per setting: full vs subst at the same (N, B); "same p99" holds when full's p99 is not higher
    print("## H6 per setting (full vs subst, same N and B)\n")
    print("| hw | cache | N | B | subst tok/s | full tok/s | gain | p99 subst / full (ms) | p99 ok | best row | best gain |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    gains = defaultdict(list)
    for s in settings:
        a, b = get["full"](s), get["subst"](s)
        if not (a and b):
            continue
        g = a["throughput_tok_s"] / b["throughput_tok_s"] - 1
        ok = a["tpot_p99_ms"] <= b["tpot_p99_ms"] * TOL
        best = max(((n, get[n](s)) for n, *_ in ROWS if get[n](s) and n not in ("base", "-subst")),
                   key=lambda x: x[1]["throughput_tok_s"])
        gains[(s[0], s[1])].append((g, ok))
        print(f"| {s[0]} | {s[1]:.2f} | {s[2]} | {s[3]} | {b['throughput_tok_s']:.1f} | {a['throughput_tok_s']:.1f} | "
              f"{100 * g:+.1f}% | {b['tpot_p99_ms']:.0f} / {a['tpot_p99_ms']:.0f} | {'yes' if ok else 'no'} | "
              f"{best[0]} | {100 * (best[1]['throughput_tok_s'] / b['throughput_tok_s'] - 1):+.1f}% |")
    print("\n## H6 summary: settings with full >= +15% and p99 not higher\n")
    for k, v in sorted(gains.items()):
        n_ok = sum(g >= 0.15 and ok for g, ok in v)
        print(f"- {k[0]} cache {k[1]:.2f}: {n_ok}/{len(v)}; gains {', '.join(f'{100 * g:+.0f}%' for g, _ in v)}")
    # 3. one-component contributions: each added component vs the row before it, and the leave-one-out rows
    print("\n## Contribution of each step (geo-mean throughput ratio over all N x B, per hw x cache)\n")
    steps = [("subst", "base"), ("batch", "subst"), ("res", "batch"), ("pref", "res"), ("full", "pref"),
             ("full", "-batch"), ("full", "-subst")]
    print("| hw | cache | " + " | ".join(f"{a} / {b}" for a, b in steps) + " |")
    print("|---|---|" + "---|" * len(steps))
    for hw, f in sorted({s[:2] for s in settings}):
        ss = [s for s in settings if s[:2] == (hw, f)]
        cells = []
        for a, b in steps:
            rr = [(get[a](s), get[b](s)) for s in ss]
            rr = [x["throughput_tok_s"] / y["throughput_tok_s"] for x, y in rr if x and y]
            cells.append(f"{100 * (gmean(rr) - 1):+.1f}%" if rr else "–")
        print(f"| {hw} | {f:.2f} | " + " | ".join(cells) + " |")
    h6_same_p99(runs)


def h6_same_p99(runs):
    fam = lambda r: (r["policy"] == PF and r["subst"] == "rate:0.1" and r["rho"] == 0.5 and r["protect"])
    ref = lambda r: (r["policy"] == "lfu/none" and r["subst"] == "rate:0.1" and r["rho"] == 0.0 and not r["protect"]
                     and r["select"] == "fcfs")
    tag = lambda r: "fcfs" if r["select"] == "fcfs" else f"{r['select']} k{r['klim']}"
    groups = defaultdict(list)
    for r in runs:
        groups[(r["hw"], r["cache_frac"], r["n"])].append(r)
    print("\n## H6 at the same p99: full-stack family vs `subst` (fcfs)\n")
    print("(a) same B: best full-stack point with p99 <= the reference's; (b) frontier over all B <= N: at the "
          "reference's best-throughput p99, best of each side\n")
    def best_under(rs, p99):
        ok = [r for r in rs if r["tpot_p99_ms"] <= p99 * TOL]
        return max(ok, key=lambda r: r["throughput_tok_s"]) if ok else None

    def gain(x, y):
        return x["throughput_tok_s"] / y["throughput_tok_s"] - 1 if x and y else None

    pct = lambda g: f"{100 * g:+.1f}%" if g is not None else "–"
    point = lambda r: f"B{r['B']} {tag(r)} {r['throughput_tok_s']:.1f}" if r else "none"
    print("| hw | cache | N | B | ref tok/s | ref p99 ms | best full, same B, p99 <= ref | gain |")
    print("|---|---|---|---|---|---|---|---|")
    summ = defaultdict(list)
    front = []
    for (hw, f, n), rs in sorted(groups.items()):
        R = [r for r in rs if ref(r)]
        F = [r for r in rs if fam(r)]
        if not R or not F:
            continue
        for b in sorted({r["B"] for r in R}):
            r0 = next(r for r in R if r["B"] == b)
            best = best_under([r for r in F if r["B"] == b], r0["tpot_p99_ms"])
            summ[(hw, f)].append(gain(best, r0))
            print(f"| {hw} | {f:.2f} | {n} | {b} | {r0['throughput_tok_s']:.1f} | {r0['tpot_p99_ms']:.0f} | "
                  f"{point(best)} | {pct(gain(best, r0))} |")
        rb = max(R, key=lambda r: r["throughput_tok_s"])          # the reference's best point over all B
        front.append((hw, f, n, rb, best_under(F, rb["tpot_p99_ms"])))
    print("\nsame-B summary (settings with gain >= +15% / all):")
    for k, v in sorted(summ.items()):
        vv = [x for x in v if x is not None]
        print(f"- {k[0]} cache {k[1]:.2f}: {sum(x >= 0.15 for x in vv)}/{len(v)}; median "
              f"{100 * np.median(vv) if vv else float('nan'):+.1f}%")
    print("\n(b) frontier over all B <= N: the reference's best point vs the full family's best at p99 <= its p99\n")
    print("| hw | cache | N | ref best (p99 ms) | full best | gain |")
    print("|---|---|---|---|---|---|")
    for hw, f, n, rb, fb in front:
        print(f"| {hw} | {f:.2f} | {n} | {point(rb)} ({rb['tpot_p99_ms']:.0f}) | {point(fb)} | {pct(gain(fb, rb))} |")


if __name__ == "__main__":
    main()
