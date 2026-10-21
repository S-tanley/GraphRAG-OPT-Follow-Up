#!/bin/bash
# Round 2 of the trace-driven simulations (after the GLM round 1):
#  - cache (E3/E4): adds HYB:<pred>:<Q>:<lam> (MOEINF + predicted prior), with the token-based prior
#    (BASE_ADMIT / RPP) as the fair counterpart of the retrieval prior (OURS_ADMIT), and MIX<a>
#    predictors (a*ORACLE + (1-a)*GLOBAL) to map predictor recall -> bytes saved;
#  - batch (E2): W1 and the decode-heavy W2, admission grouping (MoE-Gen-style) on/off.
# Usage: bash experiments/moe/run_sim2.sh <model> [P]
M=$1; P=${2:-40}
cd ~/mbzuai-projects/graphrag-followup
OUT=outputs/moe/sim/parts2; mkdir -p $OUT
POL="LRU LFU MOEINF PRED:GLOBAL:0 PRED:RPP:0 PRED:BASE_ADMIT:0 PRED:OURS_ADMIT:0 PRED:PREFILL:0
     HYB:GLOBAL:0:1 HYB:RPP:0:1 HYB:BASE_ADMIT:0:1 HYB:OURS_ADMIT:0:1 HYB:BASE_ADMIT:4:1 HYB:OURS_ADMIT:4:1
     PRED:MIX10:0 HYB:MIX10:0:1 PRED:MIX25:0 HYB:MIX25:0:1 PRED:ORACLE:0 HYB:ORACLE:0:1 BELADY"
WLS=""
for w in W1-cwq W1-webqsp W2-cwq Wsynr-cwq; do
  [ -f outputs/moe/e1/${M}_${w}_pred.npz ] && WLS="$WLS $w"
done
echo "workloads:$WLS"
jobs=()
for w in $WLS; do
  for c in 0.1 0.3 0.5 0.7; do for p in $POL; do
    f=$OUT/${M}_${w}_cache_${c}_${p//:/-}.json
    [ -f $f ] || jobs+=("--model $M --workload $w --mode cache --caps $c --policies $p --out $f")
  done; done
  case $w in W1-cwq|W2-cwq)
    for c in 0.3 0.5; do for b in 8 32; do for g in 1 8; do for p in FCFS AFF:BASE_ADMIT AFF:OURS_ADMIT AFF:ORACLE; do
      f=$OUT/${M}_${w}_batch_${c}_${b}_g${g}_${p//:/-}.json
      [ -f $f ] || jobs+=("--model $M --workload $w --mode batch --caps $c --bmax $b --group $g --taus 0,8,32 --policies $p --out $f")
    done; done; done; done;;
  esac
done
echo "${#jobs[@]} jobs"
printf '%s\n' "${jobs[@]}" | CUDA_VISIBLE_DEVICES= xargs -P $P -I{} sh -c ".venv/bin/python experiments/moe/sim.py {} > /dev/null 2>&1"
.venv/bin/python - "$M" $WLS <<PY
import glob, json, sys
m = sys.argv[1]
for w in sys.argv[2:]:
    for mode in ["cache", "batch"]:
        rows = []
        for f in sorted(glob.glob(f"outputs/moe/sim/parts2/{m}_{w}_{mode}_*.json")):
            rows += json.load(open(f))
        if rows:
            json.dump(rows, open(f"outputs/moe/sim/{m}_{w}_{mode}_r2.json", "w"), indent=1)
            print(m, w, mode, len(rows), "rows")
PY
echo SIM2 DONE $M
