"""Summarise sim.py outputs as markdown tables.

  python experiments/moe/sim_summary.py outputs/moe/sim/GLM-4.5-Air_W1-cwq_cache.json ...
"""

import json
import sys
from collections import defaultdict

# non-oracle policies that use no retrieval information (token-based priors count as baselines)
BASELINES = {"LRU", "LFU", "MOEINF", "PRED:GLOBAL:0", "PRED:RPP:0", "PRED:BASE_ADMIT:0", "PRED:BASE_ADMIT:4",
             "PRED:PREFILL:0", "HYB:GLOBAL:0:1", "HYB:RPP:0:1", "HYB:BASE_ADMIT:0:1", "HYB:BASE_ADMIT:4:1"}


def cache_table(rows, name):
    print(f"\n### {name}（cache：单请求顺序执行）\n")
    by = defaultdict(list)
    for r in rows:
        by[r["cap_frac"]].append(r)
    for cf, rs in sorted(by.items()):
        base = min((r for r in rs if r["policy"] in BASELINES), key=lambda r: r["decode_GB_per_req"])
        print(f"缓存容量 {cf:.0%}（最好的非 oracle 基线：{base['policy']}，decode {base['decode_GB_per_req']:.2f} GB/请求）\n")
        print("| 策略 | 预测 recall@2k | prefill GB/请求 | decode GB/请求 | 相对最好基线 | decode 命中率 | 前 8 步未命中/请求 |")
        print("|---|---|---|---|---|---|---|")
        for r in sorted(rs, key=lambda r: r["decode_GB_per_req"]):
            rel = r["decode_GB_per_req"] / base["decode_GB_per_req"] - 1
            rec = r.get("pred_recall_2k")
            print(f"| {r['policy']} | {'–' if rec is None else f'{rec:.3f}'} | {r['prefill_GB_per_req']:.2f} | {r['decode_GB_per_req']:.2f} | {rel:+.1%} | "
                  f"{r['decode_hit']:.3f} | {r['decode_miss_first8_per_req']:.1f} |")
        print()


def batch_table(rows, name):
    print(f"\n### {name}（batch：continuous batching + 准入策略）\n")
    print("| 缓存 | Bmax | 分组 | τ | 策略 | 吞吐 tok/s | decode tok/s | decode GB/token | 平均延迟 s | P99 延迟 s | 平均等待 s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["cap_frac"], r["Bmax"], r.get("group", 1), r["tau"], r["policy"])):
        print(f"| {r['cap_frac']:.0%} | {r['Bmax']} | {r.get('group', 1)} | {r['tau']} | {r['policy']} | {r['throughput_tok_s']:.1f} | "
              f"{r.get('decode_tok_s', float('nan')):.1f} | {r.get('decode_GB_per_token', float('nan')):.3f} | "
              f"{r['lat_mean_s']:.2f} | {r['lat_p99_s']:.2f} | {r['wait_mean_s']:.2f} |")


for f in sys.argv[1:]:
    rows = json.load(open(f))
    name = f.split("/")[-1].replace(".json", "")
    (cache_table if "_cache" in f else batch_table)(rows, name)
