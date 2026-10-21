#!/bin/bash
# GPUs 0/1 of mbzuai-masdar went into "GPU requires reset" on 2026-10-18 ~22:00 and CUDA cannot initialise on
# any GPU until an admin resets them. Poll every 5 min; once GPUs 0,4,5 work again, run the remaining E0
# queue (finished workloads are skipped by run_e0.sh).
cd ~/mbzuai-projects/graphrag-followup
ok() {
  CUDA_VISIBLE_DEVICES=0,4,5 timeout 120 .venv/bin/python -c "
import torch
assert torch.cuda.device_count() == 3
for i in range(3):
    torch.ones(1, device=f'cuda:{i}').sum().item()
    free, _ = torch.cuda.mem_get_info(i)
    assert free > 130 * 2**30, (i, free / 2**30)
" > /dev/null 2>&1
}
until ok; do echo "[$(date +%m-%d\ %H:%M)] GPUs 0,4,5 not usable yet"; sleep 300; done
echo "[$(date +%m-%d\ %H:%M)] GPUs usable, starting E0 queue"
export GPUS=0,4,5 GIB=128
bash experiments/moe/run_e0.sh mistralai/Mixtral-8x22B-Instruct-v0.1 48 200 0 2000 > logs/e0_mixtral3.log 2>&1
bash experiments/moe/run_e0.sh Qwen/Qwen3-235B-A22B 32 64 0 2000                 > logs/e0_qwen3.log 2>&1
bash experiments/moe/run_e0.sh deepseek-ai/DeepSeek-V2.5 32 64 0 2000            > logs/e0_deepseek.log 2>&1
bash experiments/moe/run_e0.sh zai-org/GLM-4.5-Air 48 200 1 2000                 > logs/e0_glm_synr.log 2>&1
echo ALL E0 DONE > logs/e0_all.done
