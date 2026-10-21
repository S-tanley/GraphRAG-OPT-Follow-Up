#!/bin/bash
# E0 for the B tier (H200 plan models, 4bit, real SSD offload with offload_engine.py), one model at a time,
# after the A tier. Expert stores go to ~/mlx_offload (internal SSD = "host memory" tier); a store is deleted
# after its traces are done, except GLM's (kept for the phase-D prototype).
# Usage: bash experiments/moe_edge/run_b_tier.sh > logs_mac/b_tier.log 2>&1
cd "$(dirname "$0")/../.."
until grep -q "W2 CHAT DONE" logs_mac/a_tier_w2chat.log 2>/dev/null; do sleep 120; done
PY=".venv-mac/bin/python experiments/moe_edge/collect_traces_mlx.py"
OUT=outputs/moe_edge/traces
BUDGET_GB=41
run() {   # run <model> <name> <store> <slots> <tag> <args...>
  local m=$1 name=$2 st=$3 sl=$4 tag=$5; shift 5
  if [ -f $OUT/${name}_$tag.json ]; then echo "[$(date +%H:%M)] SKIP ${name}_$tag"; return; fi
  echo "[$(date +%H:%M)] START ${name}_$tag slots=$sl $*"
  $PY --model $m --name $name --out_dir $OUT --offload_store $st --slots $sl --policy sw-moeinf "$@" 2>&1 \
      | grep -v -E "Warning|clean_up_tokenization|Generating|it/s\]|MallocStackLogging|Fetching"
  echo "[$(date +%H:%M)] END ${m##*/}_$tag rc=${PIPESTATUS[0]}"
}
# Qwen3 emits thinking / format debris on raw few-shot text: wrap the same prompt in its chat template
for spec in "mlx-community/GLM-4.5-Air-4bit GLM-4.5-Air-4bit" \
            "mlx-community/Mixtral-8x22B-Instruct-v0.1-4bit Mixtral-8x22B-Instruct-v0.1-4bit" \
            "mlx-community/Qwen3-235B-A22B-4bit Qwen3-235B-A22B-4bit-cf --chat_fewshot" \
            "mlx-community/DeepSeek-V2.5-1210-4bit DeepSeek-V2.5-1210-4bit"; do
  set -- $spec; m=$1; name=$2; cf=$3
  until grep -q "DONE $m" logs_mac/download.log; do sleep 120; done
  st=~/mlx_offload/${m##*/}
  [ -f $st/meta.json ] || .venv-mac/bin/python experiments/moe_edge/export_experts.py --model $m --out $st 2>&1 | grep -v -E "Warning|Fetching"
  sl=$(.venv-mac/bin/python -c "import json; m=json.load(open('$st/meta.json')); print(max(1, min(m['E'], int($BUDGET_GB * 1e9 / m['record_bytes'] / len(m['layer_ids'])))))")
  echo "[$(date +%H:%M)] $m: $sl slots per layer"
  run $m $name $st $sl cwq_M0_cap8    --dataset rmanluo/RoG-cwq    --cond M0 --n 500 --max_new_tokens 32 --check 10 $cf
  run $m $name $st $sl webqsp_M0_cap8 --dataset rmanluo/RoG-webqsp --cond M0 --n 500 --max_new_tokens 32 $cf
  [ -f outputs/moe/syn_map_rand.json ] && run $m $name $st $sl cwq_M0_synr_cap8 --dataset rmanluo/RoG-cwq --cond M0 \
      --n 500 --max_new_tokens 32 --rel_map outputs/moe/syn_map_rand.json $cf
  run $m $name $st $sl train_M0_cap8  --paths_jsonl outputs/exp3/train_paths.jsonl --cond M0 --n 1000 --max_new_tokens 32 $cf
  run $m $name $st $sl cwq_M0_cot_chat_cap8 --dataset rmanluo/RoG-cwq --cond M0 --cot --chat --n 32 --max_new_tokens 1024
  if [ "$name" != "GLM-4.5-Air-4bit" ]; then rm -f $st/experts.bin; echo "removed $st/experts.bin"; fi
done
echo "B TIER DONE"
