#!/bin/bash
# Run every (workload, cap, policy) cache simulation and every batch config as separate processes.
# Usage: bash experiments/moe/run_sim_parallel.sh <model> [P]
M=$1; P=${2:-40}
cd ~/mbzuai-projects/graphrag-followup
OUT=outputs/moe/sim/parts; mkdir -p $OUT
POL="LRU LFU MOEINF PRED:GLOBAL:0 PRED:RPP:0 PRED:OURS_ADMIT:0 PRED:PREFILL:0 PRED:BASE_ADMIT:4 PRED:OURS_ADMIT:4 PRED:ORACLE:0 PRED:ORACLE:4 BELADY"
jobs=()
for w in W1-cwq W1-webqsp; do
  for c in 0.1 0.3 0.5 0.7; do for p in $POL; do
    jobs+=("--model $M --workload $w --mode cache --caps $c --policies $p --out $OUT/${M}_${w}_cache_${c}_${p//:/-}.json")
  done; done
  for c in 0.3 0.5; do for b in 4 8 16 32; do for p in FCFS AFF:BASE_ADMIT AFF:OURS_ADMIT AFF:ORACLE; do
    jobs+=("--model $M --workload $w --mode batch --caps $c --bmax $b --taus 0,8,32 --policies $p --out $OUT/${M}_${w}_batch_${c}_${b}_${p//:/-}.json")
  done; done; done
done
printf '%s\n' "${jobs[@]}" | CUDA_VISIBLE_DEVICES= xargs -P $P -I{} sh -c ".venv/bin/python experiments/moe/sim.py {} > /dev/null 2>&1"
# merge parts
.venv/bin/python - "$M" <<PY
import glob, json, sys
m = sys.argv[1]
for w in ["W1-cwq", "W1-webqsp"]:
    for mode in ["cache", "batch"]:
        rows = []
        for f in sorted(glob.glob(f"outputs/moe/sim/parts/{m}_{w}_{mode}_*.json")):
            rows += json.load(open(f))
        json.dump(rows, open(f"outputs/moe/sim/{m}_{w}_{mode}.json", "w"), indent=1)
        print(m, w, mode, len(rows), "rows")
PY
echo SIM DONE $M
