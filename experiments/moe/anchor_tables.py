"""Markdown tables of anchor_offline_v2.py results: every predictor x h = 1..8, best per column in bold.

  python experiments/moe/anchor_tables.py outputs/moe/anchor_v3_glm.json cwq_M0_cap8 "title"
"""
import json
import sys

SIMS = ["route1-tok", "route2-tok", "route3-tok", "proj-tok", "proj-glob", "proj-hier", "logit-tok", "logit-glob",
        "logit-hier"]
GROUPS = [("基线", ["TOKID", "GATE", "TOK+GATE"]),
          ("anchor 单独 · 最相似 1 条（A1）", [f"A1-{s}" for s in SIMS]),
          ("anchor 单独 · 前 8 条加权投票（AV）", [f"AV-{s}" for s in SIMS]),
          ("anchor 投票 + GATE", [f"AV+GATE-{s}" for s in SIMS])]
H = range(1, 9)


def table(res, wl, title):
    r = res[wl]["recall"]
    allm = [m for _, ms in GROUPS for m in ms]
    best = {h: max(r[f"{m}|{h}"][0] for m in allm if f"{m}|{h}" in r) for h in H}
    out = [f"**{title}**", "", "| 方法 | " + " | ".join(f"h={h}" for h in H) + " |", "|---|" + "---|" * len(H)]
    for g, ms in GROUPS:
        out.append(f"| *{g}* |" + " |" * len(H))
        for m in ms:
            cells = []
            for h in H:
                v = r.get(f"{m}|{h}")
                if not v:
                    cells.append("–")
                    continue
                s = f".{round(v[0] * 1000):03d}"
                cells.append(f"**{s}**" if abs(v[0] - best[h]) < 1e-9 else s)
            out.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    print(table(json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]))
