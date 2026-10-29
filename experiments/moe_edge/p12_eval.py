"""Multi-user plan 1.2 stage D: the W1-selected prefetch configurations on the CoT test workload.

Criteria (plan 1.2): the adaptive threshold reaches >= 80% of the effect of the post-hoc best fixed threshold
(effect = stall of one-layer-ahead prefetch - stall), and prefetch waste <= 15%. The post-hoc reference takes, per
setting, the best fixed-threshold configuration of the same (predictor, eviction) family on CoT itself (stage A plus
the phase-0 sweep, C2 on). Also the ablation table (1.2.4): each mechanism removed from the final configuration.

  python experiments/moe_edge/p12_eval.py outputs/moe/mu/p12/p12d_cot.json outputs/moe/mu/p12/p12a_cot.json \
      outputs/moe/mu/p04_cot.json > outputs/moe/mu/p12/p12d_tables.md
"""
import json
import re
import sys

FINAL = {("GATEs", "lru"): "lru/ah8:GATEs:m8:p0.7:kl0.05", ("GATEs", "moeinf"): "moeinf/ah8:GATEs:m8:p0.3:kl0.1:hl0.2",
         ("AVG", "lru"): "lru/ah8:AVG:m8:p0.7:g0.5:kl0.1:hl0.2", ("AVG", "moeinf"): "moeinf/ah8:AVG:m8:p0.3:kl0.1:hl0.2"}
W1FIXED = {("GATEs", "lru"): "lru/ah8:GATEs:m8:p0.7", ("GATEs", "moeinf"): "moeinf/ah8:GATEs:m8:p0.7",
           ("AVG", "lru"): "lru/ah8:AVG:m8:p0.9:g0.5", ("AVG", "moeinf"): "moeinf/ah8:AVG:m8:p0.7"}
NAME = {"GATEs": "校准 GATE", "AVG": "anchor+GATE"}


def fam(p):
    ev, pref = p.split("/")
    return (pref.split(":")[1] if ":" in pref else pref), ev


def is_fixed(p):
    return re.search(r":(k|kl|hl|d)[\d.]", p) is None and ":p" in p


def main():
    test = json.load(open(sys.argv[1]))
    ref = sum((json.load(open(f)) for f in sys.argv[2:]), [])
    ref = [r for r in ref if r.get("protect", True)]                     # C2 on, as in the test runs
    S = sorted({(r["hw"], r["cache_frac"]) for r in test})
    st = {}
    for r in test + ref:
        st.setdefault((r["hw"], r["cache_frac"], r["policy"]), r)
    print("### 1.2 判据（CoT 50，C2 开；参数全部在 W1 上选定）\n")
    print("| 家族 | 硬件 | 缓存 | 提前 1 层 | 事后最优固定阈值（CoT 上挑） | W1 选的固定阈值 | **W1 选的自适应** | 自适应 / 事后最优的收益 | 预取浪费 |")
    print("|---|---|---|---|---|---|---|---|---|")
    summary = {}
    for (pred, ev), fp in FINAL.items():
        ratios = []
        for hw, f in S:
            la1 = st[(hw, f, f"{ev}/la1")]["stall_ms_per_tok"]
            fixed = [r for (h2, f2, p), r in st.items() if (h2, f2) == (hw, f) and fam(p) == (pred, ev) and is_fixed(p)]
            post = min(fixed, key=lambda r: r["stall_ms_per_tok"])
            a = st[(hw, f, fp)]
            w = st[(hw, f, W1FIXED[(pred, ev)])]
            eff_post = la1 - post["stall_ms_per_tok"]
            ratio = (la1 - a["stall_ms_per_tok"]) / eff_post if eff_post > 0 else float("nan")
            ratios.append(ratio)
            print(f"| {NAME[pred]} / {ev} | {hw} | {f:.0%} | {la1:.1f} | {post['stall_ms_per_tok']:.1f}（{post['policy'].split('/', 1)[1]}） | "
                  f"{w['stall_ms_per_tok']:.1f} | **{a['stall_ms_per_tok']:.1f}** | {ratio:.0%} | {a['pf_waste_frac']:.0%} |")
        summary[(pred, ev)] = ratios
    print("\n收益比（自适应 / 事后最优）：" + "；".join(f"{NAME[p]} / {e}：最低 {min(v):.0%}，平均 {sum(v) / len(v):.0%}"
                                          for (p, e), v in summary.items()))
    print("\n### 1.2.4 消融（CoT；相对最终配置的停顿变化，6 个设置的平均 / 最差）\n")
    print("| 家族 | 最终配置 | 去掉自适应（只留 τ0） | 换成 W1 选的固定阈值 | 去掉按 h 加阈值 | 去掉 γ | 加冻结窗口 d=1 |")
    print("|---|---|---|---|---|---|---|")
    for (pred, ev), fp in FINAL.items():
        parts = fp.split(":")
        drop = lambda pre: ":".join(x for x in parts if not x.startswith(pre))
        variants = [drop("kl"), W1FIXED[(pred, ev)], drop("hl") if ":hl" in fp else None, drop("g") if ":g" in fp else None,
                    fp + ":d1:x" + ("0.9" if pred == "GATEs" else "1.0")]
        cells = []
        for v in variants:
            if v is None or v == fp:
                cells.append("–")
                continue
            d = [st[(h, f, v)]["stall_ms_per_tok"] / st[(h, f, fp)]["stall_ms_per_tok"] - 1 for h, f in S]
            cells.append(f"{sum(d) / len(d):+.1%} / {max(d):+.1%}")
        print(f"| {NAME[pred]} / {ev} | `{fp.split('/', 1)[1]}` | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
