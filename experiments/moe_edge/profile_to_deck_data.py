"""Turn the two profile JSONs into the compact data the profiling slides embed.
Timeline: ops end to end with their durations; per layer, the unattributed gap (measured end-to-end time
minus the timeline sum, if positive) is appended as an 'overhead' segment split evenly over the 3 layers.
SSD reads are placed inside their wait op at the same offset they had in the real run.
  .venv-mac/bin/python experiments/moe_edge/profile_to_deck_data.py > outputs/moe_edge/profile/deck_data.json
"""
import json, sys

GROUP = {"input_norm": "norm", "post_norm": "norm", "qkv_proj": "proj", "o_proj_residual": "proj",
         "qk_norm_rope_kv": "attn", "attention": "attn", "router": "router", "route_to_host": "host",
         "expert_load_wait": "io", "experts": "experts", "combine_residual": "combine"}
out = {"configs": []}
for mode in ("resident", "offload"):
    d = json.load(open(f"outputs/moe_edge/profile/{mode}.json"))
    out.setdefault("peaks", {})[mode] = d["peaks"]
    out["model"] = d["model"]; out["cache_frac"] = 0.5; out["prompt_tokens"] = d["prompt_tokens"]
    for c in d["configs"]:
        pk = d["peaks"]
        segs, t = [], 0.0
        tot = sum(e["dur"] for e in c["events"])
        gap = c["nosync_total_s"] - tot
        io_by_op = []
        for li in range(3):
            for e in [e for e in c["events"] if e["layer"] == li]:
                fp = pk["flops_q4"] if e["flop_peak"] == "q4" else pk["flops_bf16"]
                bw = e["bytes"] / e["dur"] if e["dur"] > 0 else 0
                fl = e["flops"] / e["dur"] if e["dur"] > 0 else 0
                seg = dict(n=e["name"], g=GROUP[e["name"]], l=li, s=round(t * 1e3, 5), d=round(e["dur"] * 1e3, 5),
                           b=round(e["bytes"]), f=round(e["flops"]), bw=round(bw / 1e9, 2),
                           mu=round(min(1.0, bw / pk["mem_bw"]), 4) if e["kind"] == "gpu" else 0,
                           cu=round(min(1.0, fl / fp), 4) if e["kind"] == "gpu" else 0)
                if "distinct_experts" in e: seg["U"] = e["distinct_experts"]
                if "loaded" in e: seg["ld"] = e["loaded"]
                if e["kind"] == "io":
                    io_by_op.append((e["wall_t0"], e["wall_t1"], t))
                segs.append(seg)
                t += e["dur"]
            if gap > 0:
                segs.append(dict(n="overhead", g="overhead", l=li, s=round(t * 1e3, 5), d=round(gap / 3 * 1e3, 5),
                                 b=0, f=0, bw=0, mu=0, cu=0))
                t += gap / 3
        io = []
        for r in c["io"]:
            for w0, w1, syn in io_by_op:
                if w0 - 1e-4 <= r["wall_t0"] <= w1 + 1e-4:
                    io.append([round((syn + r["wall_t0"] - w0) * 1e3, 5), round((syn + r["wall_t1"] - w0) * 1e3, 5), r["bytes"]])
                    break
        out["configs"].append(dict(mode=mode, phase=c["phase"], batch=c["batch"], segs=segs, io=io,
                                   timeline_ms=round(tot * 1e3, 4), e2e_ms=round(c["nosync_total_s"] * 1e3, 4),
                                   e2e_all_ms=[round(x * 1e3, 3) for x in c["all_nosync_s"]]))
json.dump(out, sys.stdout, separators=(",", ":"))
