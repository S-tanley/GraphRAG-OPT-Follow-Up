#!/usr/bin/env bash
# Reviewer-B comprehensive throughput/memory batch sweep (isolated).
# Usage: run_sweep.sh <GPU> <model> <tag>   (loops CWQ + WEB)
set -u
source /home/am137688/anaconda3/etc/profile.d/conda.sh && conda activate graphrag
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
cd /home/am137688/amir/graphRag/graphrag_kv
GPU=$1; MODEL=$2; TAG=$3
BATCHES="1 2 4 8 16 32 64 128 256 512"
for DS in rmanluo/RoG-cwq rmanluo/RoG-webqsp; do
  DSTAG=$(echo $DS | sed 's#.*/##; s/RoG-//')
  echo "=== $TAG $DSTAG on GPU$GPU ==="
  CUDA_VISIBLE_DEVICES=$GPU python reviewer_b_gen/mem_throughput.py \
    --model "$MODEL" --dataset "$DS" \
    --batches $BATCHES --n_queries 512 --max_paths 8 --decode_tokens 16 \
    --out "reviewer_b_gen/results/sweep_${TAG}_${DSTAG}.csv" \
    2>&1 | grep -aE "B=|Wrote|OutOfMemory|topical|Error|Traceback" | grep -av "Loading weights"
done
echo "DONE $TAG"
