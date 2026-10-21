#!/bin/bash
# Download every model of the edge plan to the external SSD (the "model repository" tier).
# Usage: bash experiments/moe_edge/download_models.sh > logs_mac/download.log 2>&1
export HF_HOME=/Volumes/ZBW-SSD1/hf HF_HUB_ENABLE_HF_TRANSFER=0
HF=.venv-mac/bin/hf
for r in allenai/OLMoE-1B-7B-0924-Instruct \
         mlx-community/Qwen3-30B-A3B-4bit \
         deepseek-ai/DeepSeek-V2-Lite-Chat \
         mlx-community/GLM-4.5-Air-4bit \
         mlx-community/Mixtral-8x22B-Instruct-v0.1-4bit \
         mlx-community/Qwen3-235B-A22B-4bit \
         mlx-community/DeepSeek-V2.5-1210-4bit; do
  echo "[$(date '+%m-%d %H:%M:%S')] START $r"
  for attempt in 1 2 3; do
    $HF download "$r" --max-workers 8 > /dev/null && break
    echo "  attempt $attempt failed, retrying in 60s"; sleep 60
  done
  echo "[$(date '+%m-%d %H:%M:%S')] DONE $r ($(du -sh $HF_HOME/hub/models--${r//\//--} | cut -f1))"
done
echo ALL DOWNLOADS DONE
