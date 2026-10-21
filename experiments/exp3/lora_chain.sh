#!/bin/bash
# Wait for exp3/exp3b workers, then train LoRA-M2 and LoRA-M0 (RoG) and evaluate each.
cd ~/mbzuai-projects/graphrag-followup
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while tmux ls 2>/dev/null | grep -q -E "^exp3_|^exp3b_" || pgrep -f "exp3b GPU" > /dev/null; do sleep 60; done
for c in M2-hyb M0; do
  out=outputs/exp3/lora_RoG_$c
  [ -f $out/adapter_model.safetensors ] || .venv/bin/python experiments/exp3/train_lora.py --cond $c --out $out > logs/lora_$c.log 2>&1
  for d in rmanluo/RoG-cwq rmanluo/RoG-webqsp; do
    .venv/bin/python experiments/exp2/run.py --model rmanluo/RoG --dataset $d --lora $out \
      --conds M2-hyb,M0,M2-par,T,M1-hyb --out_dir outputs/exp3 >> logs/lora_eval_$c.log 2>&1 &
  done
done
wait
echo CHAIN DONE >> logs/lora_eval_M0.log
