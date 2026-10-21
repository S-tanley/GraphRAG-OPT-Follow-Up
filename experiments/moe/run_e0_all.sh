#!/bin/bash
# Sequentially run E0 for all four models (GLM's direct runs already exist).
cd ~/mbzuai-projects/graphrag-followup
while tmux has-session -t gatetest 2>/dev/null; do sleep 30; done
bash experiments/moe/run_e0.sh zai-org/GLM-4.5-Air 48 200 1            > logs/e0_glm.log 2>&1
bash experiments/moe/run_e0.sh mistralai/Mixtral-8x22B-Instruct-v0.1 48 200 > logs/e0_mixtral.log 2>&1
bash experiments/moe/run_e0.sh Qwen/Qwen3-235B-A22B 32 64                 > logs/e0_qwen3.log 2>&1
bash experiments/moe/run_e0.sh deepseek-ai/DeepSeek-V2.5 32 64            > logs/e0_deepseek.log 2>&1
echo ALL E0 DONE > logs/e0_all.done
