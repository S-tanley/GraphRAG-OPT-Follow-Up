#!/bin/bash
# Run phase A for each A-tier model as soon as its last workload (chat W2) is collected.
cd "$(dirname "$0")/../.."
for spec in "OLMoE-1B-7B-0924-Instruct 64 olmoe" "Qwen3-30B-A3B-4bit-cf 128 qwen3_30b_4bit" \
            "DeepSeek-V2-Lite-Chat 64 dslite"; do
  set -- $spec
  until [ -f outputs/moe_edge/traces/${1}_cwq_M0_cot_chat_cap8.json ] && [ -f outputs/moe_edge/traces/${1}_train_M0_cap8.json ]; do sleep 60; done
  echo "[$(date +%H:%M)] phase A $1"
  bash experiments/moe_edge/run_phase_a.sh $1 $2 $3 > logs_mac/phase_a_$1.log 2>&1
  .venv-mac/bin/python experiments/moe_edge/summarize_phase_a.py $1 > outputs/moe_edge/phase_a/$1/summary.md
  echo "[$(date +%H:%M)] phase A $1 done"
done
echo "PHASE A ALL DONE"
