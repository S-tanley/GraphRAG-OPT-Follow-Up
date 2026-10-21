#!/bin/bash
# Extra S1 policies for phase A (fairness): retrieval / token priors WITHOUT lookahead prefetch (at small caches
# the loader is saturated and la1 hurts, and the best non-retrieval policy is moeinf without la1).
# Usage: bash experiments/moe_edge/run_tsim_extra.sh <model> <E> <hw>
M=$1; E=$2; HWP=$3; TR=outputs/moe_edge/traces
cd "$(dirname "$0")/../.."
POL="hyb/none/T0:BASE_ADMIT,hyb/none/T0:KNN_T,hyb/none/T0:OURS_ADMIT,hyb/none/T0:KNN_R,hyb/none/T0:RPP_R,hyb/none/T0:ORACLE"
for w in cwq webqsp; do
  for cf in 0.25 0.5 0.75; do
    .venv-mac/bin/python experiments/moe_edge/tsim.py --trace $TR/${M}_${w}_M0_cap8 --train $TR/${M}_train_M0_cap8 \
        --preds outputs/moe_edge/e1/${M}_W1-${w}_pred.npz --hw $HWP --E $E --cache_frac $cf --policy $POL \
        --out outputs/moe_edge/phase_a/$M/tsim_extra_W1-${w}_${cf}.json
  done
done
