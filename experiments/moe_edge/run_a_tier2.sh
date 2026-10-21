#!/bin/bash
# A tier, second pass (2026-10-19): Qwen3-30B-A3B emits thinking / format debris on raw few-shot text
# (CWQ hit 0.20), so its W1 / Wtrain / Direct use --chat_fewshot (same few-shot text inside the chat
# template, thinking off; name suffix -cf). DeepSeek-V2-Lite reads raw few-shot fine (hit 0.42 on 50).
# Long-decode W2 for all three comes from run_a_tier_w2chat.sh. Appends "A TIER DONE" to logs_mac/a_tier.log.
cd "$(dirname "$0")/../.."
PY=".venv-mac/bin/python experiments/moe_edge/collect_traces_mlx.py"
OUT=outputs/moe_edge/traces
run() {   # run <model> <name> <tag> <args...>
  local m=$1 name=$2 tag=$3; shift 3
  if [ -f $OUT/${name}_$tag.json ]; then echo "[$(date +%H:%M)] SKIP ${name}_$tag"; return; fi
  echo "[$(date +%H:%M)] START ${name}_$tag $*"
  $PY --model $m --name $name --out_dir $OUT "$@" 2>&1 | grep -v -E "Warning|clean_up_tokenization|Generating|it/s\]|MallocStackLogging|Fetching"
  echo "[$(date +%H:%M)] END ${name}_$tag rc=${PIPESTATUS[0]}"
}
for spec in "mlx-community/Qwen3-30B-A3B-4bit Qwen3-30B-A3B-4bit-cf --chat_fewshot" \
            "deepseek-ai/DeepSeek-V2-Lite-Chat DeepSeek-V2-Lite-Chat"; do
  set -- $spec; m=$1; name=$2; cf=$3
  run $m $name cwq_M0_cap8    --dataset rmanluo/RoG-cwq    --cond M0 --n 500 --max_new_tokens 32 --check 20 $cf
  run $m $name webqsp_M0_cap8 --dataset rmanluo/RoG-webqsp --cond M0 --n 500 --max_new_tokens 32 $cf
  run $m $name train_M0_cap8  --paths_jsonl outputs/exp3/train_paths.jsonl --cond M0 --n 2000 --max_new_tokens 32 $cf
  run $m $name cwq_direct_cap8    --dataset rmanluo/RoG-cwq    --cond direct --n 500 --max_new_tokens 32 --no_trace $cf
  run $m $name webqsp_direct_cap8 --dataset rmanluo/RoG-webqsp --cond direct --n 500 --max_new_tokens 32 --no_trace $cf
done
echo "A TIER DONE" | tee -a logs_mac/a_tier.log
