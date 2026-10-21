#!/bin/bash
# W2 (decode-heavy) for the A tier with each model's chat template (raw few-shot prompts make the small
# models answer immediately). Starts after run_a_tier.sh. Usage: bash ... > logs_mac/a_tier_w2chat.log 2>&1
cd "$(dirname "$0")/../.."
until grep -q "A TIER DONE" logs_mac/a_tier.log; do sleep 60; done
PY=".venv-mac/bin/python experiments/moe_edge/collect_traces_mlx.py"
OUT=outputs/moe_edge/traces
for spec in allenai/OLMoE-1B-7B-0924-Instruct:OLMoE-1B-7B-0924-Instruct \
            mlx-community/Qwen3-30B-A3B-4bit:Qwen3-30B-A3B-4bit-cf \
            deepseek-ai/DeepSeek-V2-Lite-Chat:DeepSeek-V2-Lite-Chat; do
  m=${spec%%:*}; name=${spec##*:}
  tag=${name}_cwq_M0_cot_chat_cap8
  [ -f $OUT/$tag.json ] && { echo "SKIP $tag"; continue; }
  echo "[$(date +%H:%M)] START $tag"
  $PY --model $m --name $name --out_dir $OUT --dataset rmanluo/RoG-cwq --cond M0 --cot --chat --n 200 --max_new_tokens 2048 \
      2>&1 | grep -v -E "Warning|clean_up_tokenization|Generating|it/s\]|MallocStackLogging"
  echo "[$(date +%H:%M)] END $tag"
done
echo "W2 CHAT DONE"
