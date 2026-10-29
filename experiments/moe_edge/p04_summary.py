"""Tables for the multi-user plan 0.4 (P3 main table rerun under tsim_mu): best configuration per predictor family,
C2 effect, waste breakdown, GATE with a threshold. Reads tsim_mu.py seq outputs.

  python experiments/moe_edge/p04_summary.py outputs/moe/mu/p04_cot.json > outputs/moe/mu/p04_cot_tables.md
"""
import json
import sys
from collections import defaultdict

FAM = [("不预取", lambda p: p.endswith("/none")),
       ("提前 1 层", lambda p: p.endswith("/la1")),
       ("GATE 多层", lambda p: ":GATE:" in p and ":p" not in p),
       ("TOK+GATE 阈值", lambda p: ":TOKGATE:" in p and ":p" in p),
       ("GATE 阈值（校准）", lambda p: ":GATEs:" in p),
       ("anchor+GATE", lambda p: ":AVG:" in p),
       ("anchor+GATE（校准）", lambda p: ":AVGs:" in p),
       ("完美预测", lambda p: ":ORACLE:" in p)]
BASE = ("不预取", "提前 1 层", "GATE 多层", "TOK+GATE 阈值", "GATE 阈值（校准）")
OURS = ("anchor+GATE", "anchor+GATE（校准）")
W = ("hit", "late", "early", "corrected", "final_wrong", "end")


def best(rows, fam):
    c = [r for r in rows if dict(FAM)[fam](r["policy"])]
    return min(c, key=lambda r: r["stall_ms_per_tok"]) if c else None


def main():
    rows = json.load(open(sys.argv[1]))
    groups = defaultdict(list)
    for r in rows:
        groups[(r["hw"], r["cache_frac"], r["protect"])].append(r)
    keys = sorted({(h, f) for h, f, _ in groups})
    c2s = sorted({k[2] for k in groups})
    for c2 in c2s:
        print(f"\n### decode 停顿（ms/token），C2 {'开' if c2 else '关'}；括号里是最优配置\n")
        print("| 硬件 | 缓存 | " + " | ".join(n for n, _ in FAM) + " | 最好的 anchor 比最强基线：停顿 / TPOT | 比提前 1 层 |")
        print("|---" * (len(FAM) + 4) + "|")
        for hw, f in keys:
            g = groups.get((hw, f, c2))
            if not g:
                continue
            b = {n: best(g, n) for n, _ in FAM}
            cells = [f"{b[n]['stall_ms_per_tok']:.1f} ({b[n]['policy']})" if b[n] else "–" for n, _ in FAM]
            if not any(b[n] for n in OURS) or not b["提前 1 层"]:
                print(f"| {hw} | {f:.0%} | " + " | ".join(cells) + " | – | – |")
                continue
            sb = min((b[n] for n in BASE if b[n]), key=lambda r: r["stall_ms_per_tok"])
            so = min((b[n] for n in OURS if b[n]), key=lambda r: r["stall_ms_per_tok"])
            d_st = so["stall_ms_per_tok"] / sb["stall_ms_per_tok"] - 1
            d_tp = so["tpot_ms"] / sb["tpot_ms"] - 1
            d_la = so["stall_ms_per_tok"] / b["提前 1 层"]["stall_ms_per_tok"] - 1
            print(f"| {hw} | {f:.0%} | " + " | ".join(cells) + f" | {d_st:+.0%} / {d_tp:+.0%}（基线 {sb['policy']}） | {d_la:+.0%} |")
    print("\n### C2（保护已缓存的被预测 expert）的效果：同一策略 C2 开 / 关的停顿比，按预测器家族取中位数\n")
    print("| 硬件 | 缓存 | " + " | ".join(n for n, _ in FAM if n not in ("不预取",)) + " |")
    print("|---" * (len(FAM) + 1) + "|")
    for hw, f in keys:
        if (hw, f, True) not in groups or (hw, f, False) not in groups:
            continue
        off = {r["policy"]: r for r in groups[(hw, f, False)]}
        on = {r["policy"]: r for r in groups[(hw, f, True)]}
        cells = []
        for n, fn in FAM[1:]:
            rat = sorted(on[p]["stall_ms_per_tok"] / off[p]["stall_ms_per_tok"] - 1 for p in off if fn(p) and p in on
                         and off[p]["stall_ms_per_tok"] > 0)
            cells.append(f"{rat[len(rat) // 2]:+.1%}" if rat else "–")
        print(f"| {hw} | {f:.0%} | " + " | ".join(cells) + " |")
    print("\n### 浪费分解（C2 关）：预取加载中各类的比例；P3 口径的浪费 = 被换出时没用过的 / 全部 decode 加载\n")
    print("| 硬件 | 缓存 | 配置 | 停顿 | 预取加载 / token | 命中目标层 | 之后用上 | 目标层前被换出 | 早期预测被修正 | 最终预测也错 | 结束时未用 | 预取浪费合计 | P3 口径 |")
    print("|---" * 13 + "|")
    for hw, f in keys:
        g = groups[(hw, f, False)]
        b = {n: best(g, n) for n, _ in FAM}
        for n in ("提前 1 层", "GATE 多层", "TOK+GATE 阈值", "GATE 阈值（校准）", "anchor+GATE", "anchor+GATE（校准）"):
            r = b[n]
            if not r or not r["pf_loads"]:
                continue
            fr = [r[f"pf_{w}_frac"] for w in W]
            print(f"| {hw} | {f:.0%} | {r['policy']} | {r['stall_ms_per_tok']:.1f} | {r['pf_loads'] / r['toks']:.1f} | "
                  + " | ".join(f"{x:.0%}" for x in fr) + f" | {r['pf_waste_frac']:.0%} | {r['wasted_frac']:.0%} |")
    print("\n### 按首次预测距离 h 的预取浪费率（C2 关，最好的 anchor 配置）：(修正 + 最终也错 + 提前换出 + 结束未用) / 该 h 的预取加载\n")
    print("| 硬件 | 缓存 | 配置 | " + " | ".join(f"h={h}" for h in range(1, 9)) + " |")
    print("|---" * 11 + "|")
    for hw, f in keys:
        g = groups[(hw, f, False)]
        r = min((best(g, n) for n in OURS if best(g, n)), key=lambda x: x["stall_ms_per_tok"])
        bh = r["pf_by_h"]
        cells = []
        for h in range(1, 9):
            n = bh["loads"][h]
            wst = sum(bh[w][h] for w in ("early", "corrected", "final_wrong", "end"))
            cells.append(f"{wst / n:.0%} ({n / r['toks']:.1f}/tok)" if n else "–")
        print(f"| {hw} | {f:.0%} | {r['policy']} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
