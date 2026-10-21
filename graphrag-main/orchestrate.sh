#!/bin/bash
# Master orchestrator for the GraphRAG MICRO rebuttal experiments.
# Runs in tmux (session gr, window orch). Survives session loss. Restart-safe:
# each phase skips if its output already exists. Logs to /tmp/orch and appends
# milestones to HANDOFF_LIVE.md.
set -u
source /home/am137688/anaconda3/etc/profile.d/conda.sh
conda activate graphrag
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/am137688/amir/graphRag/graphrag_kv
ORCH=/tmp/orch; mkdir -p $ORCH results_final
PD=precomputed_paths_v2
SMALL="rmanluo/RoG meta-llama/Llama-2-7b-hf meta-llama/Llama-2-13b-hf google/gemma-2-9b Qwen/Qwen3-8B"
BIG="google/gemma-2-27b Qwen/Qwen3-32B"
COMMON="--paths_dir $PD --max_paths 30 --max_new_tokens 64 --reuse_budget 4 --recompute_k 2 --methods direct graphrag_r graphrag_opt graphrag_opt_rc"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $ORCH/orchestrate.log
       echo "- [$(date '+%H:%M')] $*" >> HANDOFF_LIVE.md; }

log "ORCHESTRATOR START"

# ---- Phase A: wait for fine-tuned planner (launched in gr:finetune) ----
if [ ! -f $PD/paths_metaqa-3hop_ft.json ]; then
  log "Phase A: waiting for fine-tuned planner RoG_metaqa_planner/"
  until [ -f RoG_metaqa_planner/model.safetensors ]; do sleep 30; done
  log "planner ready"
  # ---- Phase B: wait for small CWQ+WEB table to free the GPUs, precompute MetaQA ----
  log "Phase B: waiting for small CWQ+WEB table (41 lines) before MetaQA precompute"
  until [ $(grep -c . results_final/table2_small.tsv 2>/dev/null) -ge 41 ]; do sleep 60; done
  log "small CWQ+WEB table done; precomputing MetaQA with fine-tuned planner"
  python precompute_metaqa_planner.py --planner RoG_metaqa_planner --num_gpus 8 \
    --out $PD/paths_metaqa-3hop_ft.json > $ORCH/mqa_precompute.log 2>&1
  cp $PD/paths_metaqa-3hop_ft.json $PD/paths_metaqa-3hop.json
  REC=$(python diag_recall.py --dataset metaqa-3hop --paths_file $PD/paths_metaqa-3hop.json --n 800 --max_paths 30 2>/dev/null | grep answer-recall)
  log "MetaQA precompute done. $REC"
else
  log "Phase A/B: MetaQA paths already present, skipping"
fi

# ---- Phase C: MetaQA cells for the 5 small models ----
if [ ! -f results_final/table2_small_mqa.tsv ] || [ $(grep -c . results_final/table2_small_mqa.tsv 2>/dev/null) -lt 21 ]; then
  log "Phase C: MetaQA table (5 small models x 4 methods)"
  python run_table2.py --models $SMALL --datasets metaqa-3hop $COMMON \
    --results results_final/table2_small_mqa.tsv --outdir /tmp/t2small_mqa > $ORCH/mqa_table.log 2>&1
  log "Phase C done"
else
  log "Phase C: MetaQA small table present, skipping"
fi

# ---- Phase D: big models (Gemma-27b, Qwen3-32B), single-GPU, all 3 datasets ----
if [ ! -f results_final/table2_big.tsv ] || [ $(grep -c . results_final/table2_big.tsv 2>/dev/null) -lt 25 ]; then
  log "Phase D: big models $BIG x 3 datasets x 4 methods (single-GPU each)"
  python run_table2.py --models $BIG --datasets rmanluo/RoG-cwq rmanluo/RoG-webqsp metaqa-3hop $COMMON \
    --results results_final/table2_big.tsv --outdir /tmp/t2big > $ORCH/big_table.log 2>&1
  log "Phase D done"
else
  log "Phase D: big table present, skipping"
fi

# ---- Phase E: R1 reuse-fidelity ablation (RoG/CWQ, N=1200) ----
if [ ! -f results_final/ablation_r1.tsv ]; then
  log "Phase E: R1 reuse-fidelity ablation (RoG/CWQ N=1200)"
  echo -e "config\treuse_budget\trecompute_k\tHits@1\tF1\tsubstr\tn" > results_final/ablation_r1.tsv
  ABL=$PD/paths_rmanluo-RoG-cwq.json
  run_abl(){ # method budget rc tag
    CUDA_VISIBLE_DEVICES=$5 python table2_worker.py --model rmanluo/RoG --dataset rmanluo/RoG-cwq \
      --method $1 --paths_file $ABL --n_samples 1200 --max_paths 30 --max_new_tokens 64 \
      --reuse_budget $2 --recompute_k $3 --out_json $ORCH/abl_$4.json > $ORCH/abl_$4.log 2>&1
  }
  # ceiling + reuse-amount sweep + recompute recovery, 8 configs across 8 GPUs
  run_abl graphrag_r   0  0 ceiling     0 &
  run_abl graphrag_opt 2  0 b2_rc0      1 &
  run_abl graphrag_opt 4  0 b4_rc0      2 &
  run_abl graphrag_opt 8  0 b8_rc0      3 &
  run_abl graphrag_opt 15 0 b15_rc0     4 &
  run_abl graphrag_opt 4  1 b4_rc1      5 &
  run_abl graphrag_opt 4  2 b4_rc2      6 &
  run_abl graphrag_opt 8  2 b8_rc2      7 &
  wait
  python score_ablation.py >> results_final/ablation_r1.tsv 2>$ORCH/score_abl.err
  log "Phase E done"
else
  log "Phase E: ablation present, skipping"
fi

# ---- Phase F: assemble + integrity ----
log "Phase F: assembling final table + integrity gate"
python assemble_final.py > $ORCH/assemble.log 2>&1
python check_table_integrity.py results_final/table2_final.tsv > $ORCH/integrity.log 2>&1 && \
  log "INTEGRITY PASS" || log "INTEGRITY FAIL (see $ORCH/integrity.log)"

log "ORCHESTRATOR COMPLETE — all phases done"
