#!/bin/bash
# Full multi-GPU recompute ablation sweep across models x datasets.
# Each combo uses all 8 GPUs (data-parallel shards). Results appended live.
set -u
cd "$(dirname "$0")/.."
source /home/am137688/anaconda3/etc/profile.d/conda.sh
conda activate graphrag
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

RESULTS="${1:-/tmp/abl_full_results.txt}"
: > "$RESULTS"
echo "Full ablation sweep started $(date)" >> "$RESULTS"

MODELS=("rmanluo/RoG" "meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-13b-hf")
DATASETS=("rmanluo/RoG-cwq" "rmanluo/RoG-webqsp")

for M in "${MODELS[@]}"; do
  for D in "${DATASETS[@]}"; do
    echo "=== $(date +%H:%M:%S)  $M x $D ===" | tee -a "$RESULTS"
    python run_full_ablation.py --model "$M" --dataset "$D" \
        --n_samples -1 --num_gpus 8 --path_enum 2>/dev/null \
        | grep -E "=====|method|GraphRAG|EPIC|CacheBlend|----" >> "$RESULTS"
    echo "" >> "$RESULTS"
  done
done
echo "Sweep finished $(date)" >> "$RESULTS"
