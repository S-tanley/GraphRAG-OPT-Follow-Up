#!/bin/bash
# Exp0: reproduce the GraphRAG-Opt ablation numbers and test the missing-newline issue.
# RoG x CWQ, seed-42 500-question subset, max_paths=8, max_new_tokens=16 (ablation setting).
# Usage (from ~/mbzuai-projects/graphrag-followup):  GPU=0 bash experiments/exp0/run.sh
set -u
ROOT=$(pwd)
PY=$ROOT/.venv/bin/python
SRC=$ROOT/graphrag-main
OUT=$ROOT/outputs/exp0
mkdir -p "$OUT" "$ROOT/logs"
export CUDA_VISIBLE_DEVICES=${GPU:-0} GROPT_SRC=$SRC PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PF=$SRC/precomputed_paths/paths_rmanluo-RoG-cwq.json
COMMON="--model rmanluo/RoG --dataset rmanluo/RoG-cwq --n_samples 500 --max_paths 8 --max_new_tokens 16"
LOG=$ROOT/logs/exp0.log

run() {  # name, then command
  local name=$1; shift
  if [ -f "$OUT/$name.json" ]; then echo "skip $name" | tee -a "$LOG"; return; fi
  echo "[$(date '+%H:%M:%S')] start $name" | tee -a "$LOG"
  "$@" >> "$LOG" 2>&1 && echo "[$(date '+%H:%M:%S')] done $name" | tee -a "$LOG"
}

cd "$SRC"
# A. manual-engine ablation: expect full=0.510, none=0.442
run abl_template $PY ablation_kgqa.py $COMMON --paths_file $PF --policies full,none \
    --encode_prefix template --out_json $OUT/abl_template.json
# B/C. GraphRAG-R vs Opt with reuse_budget=0 (must be prediction-identical)
run t2_graphrag_r $PY table2_worker.py $COMMON --paths_file $PF --method graphrag_r \
    --out_json $OUT/t2_graphrag_r.json
run t2_opt_b0 $PY table2_worker.py $COMMON --paths_file $PF --method graphrag_opt \
    --reuse_budget 0 --recompute_k 0 --out_json $OUT/t2_opt_b0.json
# D/E. selective reuse, budget 4 and 8, without / with the trailing newline
for b in 4 8; do
  run t2_opt_b$b $PY table2_worker.py $COMMON --paths_file $PF --method graphrag_opt \
      --reuse_budget $b --recompute_k 0 --out_json $OUT/t2_opt_b$b.json
  run t2_opt_b${b}_nl env GROPT_PATH_NEWLINE=1 $PY table2_worker.py $COMMON --paths_file $PF \
      --method graphrag_opt --reuse_budget $b --recompute_k 0 --out_json $OUT/t2_opt_b${b}_nl.json
done
# F. reuse-all through the HF engine, without / with the trailing newline
cd "$ROOT"
run hf_reuseall $PY experiments/exp0/reuse_all_hf.py $COMMON --paths_file $PF --out_json $OUT/hf_reuseall.json
run hf_reuseall_nl env GROPT_PATH_NEWLINE=1 $PY experiments/exp0/reuse_all_hf.py $COMMON --paths_file $PF \
    --out_json $OUT/hf_reuseall_nl.json

$PY experiments/exp0/score.py "$OUT" | tee "$OUT/summary.txt"
echo "[$(date '+%H:%M:%S')] ALL DONE" | tee -a "$LOG"
