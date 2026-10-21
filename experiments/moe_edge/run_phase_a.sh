#!/bin/bash
# Phase A analyses for one model whose traces are in outputs/moe_edge/traces (or --traces dir).
# Usage: bash experiments/moe_edge/run_phase_a.sh <model tag prefix> <E> <tsim hw profile> [traces dir]
M=$1; E=$2; HWP=$3; TR=${4:-outputs/moe_edge/traces}
cd "$(dirname "$0")/../.."
PY=.venv-mac/bin/python
O=outputs/moe_edge/phase_a/$M; mkdir -p $O outputs/moe_edge/e1
echo "== [$(date +%H:%M)] E1 (T0 predictors, export)"
$PY experiments/moe/e1_analysis.py --model $M --traces $TR --E $E --out outputs/moe_edge/e1 --export > $O/e1.log 2>&1
grep -E "^== |diff OURS_ADMIT-BASE_ADMIT@|diff RPP_R-RPP@" $O/e1.log
echo "== [$(date +%H:%M)] E1 by phase (T1)"
$PY experiments/moe_edge/e1_phases.py --model $M --traces $TR --e1dir outputs/moe_edge/e1 --E $E --out $O/e1_phases.json > $O/e1_phases.log 2>&1
cat $O/e1_phases.log
echo "== [$(date +%H:%M)] decode regrouping after prefill"
$PY experiments/moe/regroup_pred.py --prefix $TR/$M --preds outputs/moe_edge/e1/${M}_W1-cwq_pred.npz --E $E --out $O/regroup_pred.json 2>&1 | tail -5
if [ -f $TR/${M}_cwq_M0_cot_chat_cap8.json ]; then
  echo "== [$(date +%H:%M)] W2 segment regrouping"
  $PY experiments/moe_edge/segment_regroup.py --trace $TR/${M}_cwq_M0_cot_chat_cap8 --E $E --N 32 --B 8 --S 8,16,32,64 --out $O/segment_regroup.json 2>&1
fi
echo "== [$(date +%H:%M)] time-stepped simulation (S1)"
POL="lru/none/none,moeinf/none/none,moeinf/la1/none,hyb/la1/T0:BASE_ADMIT,hyb/la1/T0:OURS_ADMIT,hyb/la1/T0:ORACLE,sw-moeinf/la1/none,sw-hyb/la1/T0:GLOBAL,sw-hyb/la1/T0:BASE_ADMIT,sw-hyb/la1/T0:OURS_ADMIT,sw-hyb/la1/T0:RPP_R,sw-hyb/la1/T0:KNN_T,sw-hyb/la1/T0:KNN_R,hyb/la1/T0:KNN_T,hyb/la1/T0:KNN_R,sw-hyb/la1/T1:FIRSTTOK,sw-hyb/la1/T1:CANDFIRST,sw-hyb/la1/T01:OURS_ADMIT:CANDFIRST,sw-hyb/la1/T01:BASE_ADMIT:FIRSTTOK,sw-hyb/la1/T0:ORACLE,sw-hyb/la1/T1:ORACLE"
for w in cwq webqsp; do
  P=outputs/moe_edge/e1/${M}_W1-${w}_pred.npz
  for cf in 0.25 0.5 0.75; do
    $PY experiments/moe_edge/tsim.py --trace $TR/${M}_${w}_M0_cap8 --train $TR/${M}_train_M0_cap8 --preds $P \
        --hw $HWP --E $E --cache_frac $cf --policy $POL --out $O/tsim_W1-${w}_${cf}.json 2>&1
  done
done
if [ -f $TR/${M}_cwq_M0_cot_chat_cap8.json ]; then
  for cf in 0.25 0.5 0.75; do
    $PY experiments/moe_edge/tsim.py --trace $TR/${M}_cwq_M0_cot_chat_cap8 --train $TR/${M}_train_M0_cap8 \
        --preds outputs/moe_edge/e1/${M}_W2c-cwq_pred.npz --hw $HWP --E $E --cache_frac $cf --n 100 \
        --policy "lru/none/none,moeinf/none/none,sw-moeinf/la1/none,sw-hyb/la1/seg16,sw-hyb/la1/seg32,sw-hyb/la1/T01:OURS_ADMIT:CANDFIRST,sw-hyb/la1/T0:ORACLE" \
        --out $O/tsim_W2_${cf}.json 2>&1
  done
fi
echo "== [$(date +%H:%M)] PHASE A DONE $M"
