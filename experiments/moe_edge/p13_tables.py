"""Multi-user plan 1.3 tables: use-time substitution sweep vs one-layer-ahead prefetch (tsim_mu seq outputs).

  python experiments/moe_edge/p13_tables.py outputs/moe/mu/p13/p13_cot_v2.json outputs/moe/mu/p04_cot.json \
      > outputs/moe/mu/p13/p13_tables.md
"""
import json
import sys

rows = json.load(open(sys.argv[1]))
p04 = {(r["hw"], r["cache_frac"], r["policy"]): r for r in json.load(open(sys.argv[2])) if r["protect"]}
S = [("h200_pcie", 0.25), ("h200_pcie", 0.5), ("h200_pcie", 0.75), ("mac_ssd", 0.25), ("mac_ssd", 0.5), ("mac_ssd", 0.75)]
NAME = {"none": "不预取", "GATEs": "校准 GATE（自适应）", "AVG": "anchor+GATE（自适应）"}
pname = lambda p: p.split("/")[0] + " · " + NAME["none" if p.endswith("none") else p.split(":")[1]]
subs = ["none", "rate:0.02", "rate:0.05", "rate:0.1", "tok:1", "tok:2", "skip:rate:0.05"]
lab = {"none": "不顶替", "rate:0.02": "改动 ≤ 2%", "rate:0.05": "≤ 5%", "rate:0.1": "≤ 10%", "tok:1": "每 token 每层 ≤ 1",
       "tok:2": "≤ 2", "skip:rate:0.05": "只跳过 ≤ 5%（对照）"}
la1 = {s: min(p04[(s[0], s[1], ev + "/la1")]["stall_ms_per_tok"] for ev in ("lru", "moeinf")) for s in S}
print("### 1.3：使用时顶替（CoT 50，C2 开）。每格：decode 停顿 ms/token（改动率）/ 相对提前 1 层\n")
for hw, f in S:
    print(f"\n**{hw} · 缓存 {f:.0%}**（提前 1 层：{la1[(hw, f)]:.1f}）\n")
    print("| 预取 | " + " | ".join(lab[s] for s in subs) + " |")
    print("|---" * (len(subs) + 1) + "|")
    g = [r for r in rows if r["hw"] == hw and r["cache_frac"] == f]
    for pol in sorted({r["policy"] for r in g}):
        cells = []
        for sb in subs:
            r = next(x for x in g if x["policy"] == pol and x["subst"] == sb)
            cells.append(f"{r['stall_ms_per_tok']:.1f}（{r.get('subst_change_rate', 0):.1%}）/ "
                         f"{r['stall_ms_per_tok'] / la1[(hw, f)] - 1:+.0%}")
        print(f"| {pname(pol)} | " + " | ".join(cells) + " |")
print("\n### 判据（改动率 ≤ 10% 时比提前 1 层降 ≥ 30%）\n")
print("| 硬件 | 缓存 | 最好的配置（改动 ≤ 10%） | 停顿 | 相对提前 1 层 | 改动 ≤ 5% 时最好 | 判据 |")
print("|---|---|---|---|---|---|---|")
for hw, f in S:
    g = [r for r in rows if r["hw"] == hw and r["cache_frac"] == f and r.get("subst_change_rate", 0) <= 0.1 + 1e-9
         and not r["subst"].startswith("skip")]
    b = min(g, key=lambda r: r["stall_ms_per_tok"])
    b5 = min((r for r in g if r.get("subst_change_rate", 0) <= 0.05 + 1e-9), key=lambda r: r["stall_ms_per_tok"])
    d = b["stall_ms_per_tok"] / la1[(hw, f)] - 1
    print(f"| {hw} | {f:.0%} | {pname(b['policy'])} + {b['subst']}（{b.get('subst_change_rate', 0):.1%}） | "
          f"{b['stall_ms_per_tok']:.1f} | {d:+.0%} | {b5['stall_ms_per_tok']:.1f}（{b5['stall_ms_per_tok'] / la1[(hw, f)] - 1:+.0%}，"
          f"{b5['subst']}） | {'达到' if d <= -0.3 else '没达到'} |")
print("\n### 一组固定配置跑全部 6 个设置（最差 / 各设置，相对提前 1 层）\n")
print("| 配置 | 改动 ≤ 5% | 改动 ≤ 10% |")
print("|---|---|---|")
for pol in sorted({r["policy"] for r in rows}):
    cells = []
    for sb in ("rate:0.05", "rate:0.1"):
        d = [next(r for r in rows if (r["hw"], r["cache_frac"], r["policy"], r["subst"]) == (h, f, pol, sb))["stall_ms_per_tok"]
             / la1[(h, f)] - 1 for h, f in S]
        cells.append(f"{max(d):+.0%} / " + " ".join(f"{x:+.0%}" for x in d))
    print(f"| {pname(pol)} | " + " | ".join(cells) + " |")
