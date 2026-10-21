#!/bin/bash
# Runs after E0: Wsyn-rand (per-request random paraphrase variants) for every model.
cd ~/mbzuai-projects/graphrag-followup
while [ ! -f logs/e0_all.done ]; do sleep 120; done
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,4,5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for spec in "zai-org/GLM-4.5-Air 48" "mistralai/Mixtral-8x22B-Instruct-v0.1 48" "Qwen/Qwen3-235B-A22B 32" "deepseek-ai/DeepSeek-V2.5 32"; do
  set -- $spec
  echo "[$(date +%H:%M)] START synr $1"
  .venv/bin/python experiments/moe/collect_traces.py --model $1 --dataset rmanluo/RoG-cwq --cond M0 --n 500 \
      --batch $2 --max_new_tokens 32 --rel_map outputs/moe/syn_map_rand.json \
      --max_memory 0:128GiB,1:128GiB,2:128GiB,cpu:1100GiB --out_dir outputs/moe/traces
  echo "[$(date +%H:%M)] END synr $1 rc=$?"
done
echo EXTRA DONE > logs/e0_extra.done
