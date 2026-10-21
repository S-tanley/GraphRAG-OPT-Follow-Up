#!/bin/bash
# E0 on the Mac for the A tier (models that fit in memory), one model at a time.
# Workloads whose summary json exists are skipped; Wsyn / Wsyn-rand / Wtrain run only once their input
# files are present (copied from the cluster). Usage: bash experiments/moe_edge/run_a_tier.sh > logs_mac/a_tier.log 2>&1
cd "$(dirname "$0")/../.."
PY=".venv-mac/bin/python experiments/moe_edge/collect_traces_mlx.py"
OUT=outputs/moe_edge/traces
run() {   # run <model> <tag> <args...>
  local m=$1 tag=$2; shift 2
  if [ -f $OUT/${m##*/}_$tag.json ]; then echo "[$(date +%H:%M)] SKIP ${m##*/}_$tag"; return; fi
  echo "[$(date +%H:%M)] START ${m##*/}_$tag $*"
  $PY --model $m --out_dir $OUT "$@" 2>&1 | grep -v -E "Warning|clean_up_tokenization|Generating|it/s\]"
  echo "[$(date +%H:%M)] END ${m##*/}_$tag rc=${PIPESTATUS[0]}"
}
for m in allenai/OLMoE-1B-7B-0924-Instruct mlx-community/Qwen3-30B-A3B-4bit deepseek-ai/DeepSeek-V2-Lite-Chat; do
  run $m cwq_M0_cap8       --dataset rmanluo/RoG-cwq    --cond M0 --n 500 --max_new_tokens 32 --check 20
  run $m webqsp_M0_cap8    --dataset rmanluo/RoG-webqsp --cond M0 --n 500 --max_new_tokens 32
  [ -f outputs/moe/syn_map.json ] && run $m cwq_M0_syn_cap8 --dataset rmanluo/RoG-cwq --cond M0 --n 500 \
      --max_new_tokens 32 --rel_map outputs/moe/syn_map.json
  [ -f outputs/moe/syn_map_rand.json ] && run $m cwq_M0_synr_cap8 --dataset rmanluo/RoG-cwq --cond M0 --n 500 \
      --max_new_tokens 32 --rel_map outputs/moe/syn_map_rand.json
  [ -f outputs/exp3/train_paths.jsonl ] && run $m train_M0_cap8 --paths_jsonl outputs/exp3/train_paths.jsonl \
      --cond M0 --n 2000 --max_new_tokens 32
  run $m cwq_M0_cot_cap8   --dataset rmanluo/RoG-cwq    --cond M0 --n 200 --max_new_tokens 2048 --cot
  run $m cwq_direct_cap8   --dataset rmanluo/RoG-cwq    --cond direct --n 500 --max_new_tokens 32 --no_trace
  run $m webqsp_direct_cap8 --dataset rmanluo/RoG-webqsp --cond direct --n 500 --max_new_tokens 32 --no_trace
done
echo "A TIER DONE"
