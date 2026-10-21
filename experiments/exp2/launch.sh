#!/bin/bash
# Launch Exp2 workers on one GPU (default GPU0), one process per (model, dataset).
# Usage: GPU=0 bash experiments/exp2/launch.sh "rmanluo/RoG meta-llama/Llama-2-7b-hf" "rmanluo/RoG-cwq rmanluo/RoG-webqsp" CONDS [OUT] [CAP]
cd ~/mbzuai-projects/graphrag-followup
MODELS=$1; DATASETS=$2; CONDS=$3; OUT=${4:-outputs/exp2}; CAP=${5:-8}
mkdir -p logs "$OUT"
for m in $MODELS; do for d in $DATASETS; do
  name=${PREFIX:-exp2}_$(basename $m)_$(echo $d | sed 's#.*-##')_cap$CAP
  tmux new -d -s "$name" "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${GPU:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python experiments/exp2/run.py --model $m --dataset $d --conds $CONDS --max_paths $CAP --out_dir $OUT > logs/$name.log 2>&1"
  echo "launched $name"
done; done
