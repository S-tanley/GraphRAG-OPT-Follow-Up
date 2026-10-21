#!/bin/bash
# Phase 0 + 1 for one model on GPU0+GPU1 (+host RAM): reader check (direct, no trace) and
# M0 routing traces on CWQ and WEB. Usage: bash experiments/moe/run_model.sh <hf_model> [batch]
M=$1; B=${2:-32}
cd ~/mbzuai-projects/graphrag-followup
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MM="0:125GiB,1:70GiB,cpu:1100GiB"
for d in rmanluo/RoG-cwq rmanluo/RoG-webqsp; do
  .venv/bin/python experiments/moe/collect_traces.py --model $M --dataset $d --cond M0 --n 500 --batch $B \
      --max_new_tokens 32 --max_memory $MM --out_dir outputs/moe/traces
  .venv/bin/python experiments/moe/collect_traces.py --model $M --dataset $d --cond direct --n 500 --batch $B \
      --max_new_tokens 32 --max_memory $MM --no_trace --out_dir outputs/moe/traces
done
echo MODEL DONE $M
