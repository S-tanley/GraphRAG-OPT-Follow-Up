#!/bin/bash
# E0: routing traces for all workloads of one model (+ host RAM if the GPUs cannot hold it).
# Usage: GPUS=4,5 GIB=36 bash experiments/moe/run_e0.sh <hf_model> <batch> <n_cot> [skip_direct] [n_train]
# Workloads whose summary json already exists are skipped (traced runs also resume per batch).
M=$1; B=${2:-48}; NCOT=${3:-200}; SKIP_DIRECT=${4:-0}; NTRAIN=${5:-2000}
GPUS=${GPUS:-0,4,5}; GIB=${GIB:-128}
cd ~/mbzuai-projects/graphrag-followup
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MM=$(echo $GPUS | tr ',' '\n' | awk -v g=$GIB '{printf "%d:%dGiB,", NR-1, g}')cpu:1100GiB
OUT=outputs/moe/traces
MN=${M##*/}
PY=".venv/bin/python experiments/moe/collect_traces.py --model $M --max_memory $MM --out_dir $OUT"
wait_free() {   # do not start while another user's process sits on one of our GPUs
  while true; do
    busy=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | while IFS=, read u p; do
      idx=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | grep "$(echo $u | xargs)" | cut -d, -f1)
      own=$(ps -o user= -p $p 2>/dev/null)
      if [ "$own" != "bowenzheng" ] && echo ",$GPUS," | grep -q ",$idx,"; then echo $idx; fi; done)
    [ -z "$busy" ] && break
    echo "[$(date +%H:%M)] GPU $busy busy with another user, waiting"; sleep 300
  done
}
run() {   # run <tag> <args...>
  local tag=$1; shift
  if [ -f $OUT/$tag.json ]; then echo "[$(date +%H:%M)] SKIP $tag (done)"; return; fi
  wait_free; echo "[$(date +%H:%M)] START $tag $*"; $PY "$@"; local rc=$?; echo "[$(date +%H:%M)] END rc=$rc"
}
run ${MN}_cwq_M0_cap8    --dataset rmanluo/RoG-cwq    --cond M0 --n 500 --batch $B --max_new_tokens 32
run ${MN}_webqsp_M0_cap8 --dataset rmanluo/RoG-webqsp --cond M0 --n 500 --batch $B --max_new_tokens 32
run ${MN}_cwq_M0_syn_cap8 --dataset rmanluo/RoG-cwq --cond M0 --n 500 --batch $B --max_new_tokens 32 \
    --rel_map outputs/moe/syn_map.json
run ${MN}_cwq_M0_synr_cap8 --dataset rmanluo/RoG-cwq --cond M0 --n 500 --batch $B --max_new_tokens 32 \
    --rel_map outputs/moe/syn_map_rand.json
run ${MN}_train_M0_cap8 --paths_jsonl outputs/exp3/train_paths.jsonl --cond M0 --n $NTRAIN --batch $B --max_new_tokens 32
run ${MN}_cwq_M0_cot_cap8 --dataset rmanluo/RoG-cwq --cond M0 --cot --n $NCOT --batch $(( NCOT < B ? NCOT : B )) \
    --max_new_tokens 384
if [ "$SKIP_DIRECT" != "1" ]; then     # accuracy-only controls last: they produce no traces
  run ${MN}_cwq_direct_cap8    --dataset rmanluo/RoG-cwq    --cond direct --n 500 --batch $B --max_new_tokens 32 --no_trace
  run ${MN}_webqsp_direct_cap8 --dataset rmanluo/RoG-webqsp --cond direct --n 500 --batch $B --max_new_tokens 32 --no_trace
fi
echo "E0 DONE $M"
