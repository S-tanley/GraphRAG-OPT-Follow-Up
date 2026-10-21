#!/bin/bash
# DeepSeek-V2.5 (last B-tier model) restarted with a smaller expert-cache budget: at 41 GB the pools plus
# its larger non-expert weights and staging buffer exceeded unified memory and the run thrashed in swap
# (15 requests in 3.5 h). Same workloads as run_b_tier.sh; finished requests are skipped.
cd "$(dirname "$0")/../.."
BUDGET_GB=${BUDGET_GB:-30}
m=mlx-community/DeepSeek-V2.5-1210-4bit; name=DeepSeek-V2.5-1210-4bit; st=~/mlx_offload/$name
OUT=outputs/moe_edge/traces
PY=".venv-mac/bin/python experiments/moe_edge/collect_traces_mlx.py"
sl=$(.venv-mac/bin/python -c "import json; m=json.load(open('$st/meta.json')); print(int($BUDGET_GB * 1e9 / m['record_bytes'] / len(m['layer_ids'])))")
echo "[$(date +%H:%M)] $name: $sl slots per layer (budget $BUDGET_GB GB)"
run() { local tag=$1; shift
  if [ -f $OUT/${name}_$tag.json ]; then echo "SKIP $tag"; return; fi
  echo "[$(date +%H:%M)] START $tag"
  $PY --model $m --name $name --out_dir $OUT --offload_store $st --slots $sl --policy sw-moeinf "$@" 2>&1 \
      | grep -v -E "Warning|clean_up_tokenization|Generating|it/s\]|MallocStackLogging|Fetching"
  echo "[$(date +%H:%M)] END $tag rc=${PIPESTATUS[0]}"
}
run cwq_M0_cap8    --dataset rmanluo/RoG-cwq    --cond M0 --n 500 --max_new_tokens 32 --check 10
run webqsp_M0_cap8 --dataset rmanluo/RoG-webqsp --cond M0 --n 500 --max_new_tokens 32
run train_M0_cap8  --paths_jsonl outputs/exp3/train_paths.jsonl --cond M0 --n 1000 --max_new_tokens 32
run cwq_M0_cot_chat_cap8 --dataset rmanluo/RoG-cwq --cond M0 --cot --chat --n 32 --max_new_tokens 1024
rm -f $st/experts.bin && echo "removed $st/experts.bin"
echo "B TIER DONE"
