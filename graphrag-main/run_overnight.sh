#!/bin/bash
# Unattended overnight pipeline (run under nohup). Survives terminal/session exit.
# Commits + pushes results to GitHub after every model so nothing is lost.
# Priority order: complete the headline/flagged models first, then the rest.
set -u
cd /home/am137688/amir/graphRag/graphrag_kv
source /home/am137688/anaconda3/etc/profile.d/conda.sh
conda activate graphrag
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

mkdir -p results
RES=results/table2_overnight.tsv
LOG=results/overnight.log
[ -f "$RES" ] || printf "model\tdataset\tmethod\tHits@1\tF1\tsubstr\tn\tsecs\n" > "$RES"
CW="rmanluo/RoG-cwq"; WE="rmanluo/RoG-webqsp"; MQ="metaqa-3hop"

cm()  { git add -A 2>/dev/null; git commit -q -m "overnight: $1 [$(date +%H:%M)]" 2>/dev/null; git push -q origin main 2>/dev/null; echo "  [committed $1 $(date +%H:%M)]" >> "$LOG"; }
run() { echo "=== $(date +%H:%M) run_table2 $* ===" >> "$LOG"; python run_table2.py "$@" --paths_dir precomputed_paths --max_new_tokens 64 --results "$RES" --outdir /tmp/ov >> "$LOG" 2>&1; }

echo "######## OVERNIGHT START $(date) ########" >> "$LOG"

# ---- Phase 1: headline + flagged RoPE models (<=13B), FULL aligned 4-col rows ----
# 4 methods (Direct/GraphRAG-R/GraphRAG-Opt/+RC) on CWQ+WEB full; MQA integrity full
# (direct/graphrag_r) + MQA reuse on a 2000 subset (reuse is per-sample/slow on 14k).
for M in rmanluo/RoG meta-llama/Llama-2-7b-hf google/gemma-2-9b Qwen/Qwen3-8B meta-llama/Llama-2-13b-hf; do
  run --models "$M" --datasets $CW $WE
  run --models "$M" --methods direct graphrag_r --datasets $MQ
  run --models "$M" --methods graphrag_opt graphrag_opt_rc --datasets $MQ --n_samples 2000
  cm "headline $M"
done

# ---- Phase 2: big RoPE (27/32/80B) + OPT (non-RoPE): integrity columns only ----
# Direct/GraphRAG-R (full prefill works for model-parallel + non-RoPE). CWQ+WEB full,
# MQA subset 3000 (per-sample on big models is slow).
for M in google/gemma-2-27b Qwen/Qwen3-32B Qwen/Qwen3-Next-80B-A3B-Thinking \
         facebook/opt-6.7b facebook/opt-13b facebook/opt-30b; do
  run --models "$M" --methods direct graphrag_r --datasets $CW $WE
  run --models "$M" --methods direct graphrag_r --datasets $MQ --n_samples 3000
  cm "big/opt $M"
done

# ---- Phase 3: CacheBlend/EPIC head-to-head ablation (reviewer R1) on RoG/CWQ ----
echo "=== $(date +%H:%M) ablation_kgqa ===" >> "$LOG"
python ablation_kgqa.py --model rmanluo/RoG --dataset rmanluo/RoG-cwq --path_enum \
  --n_samples 512 --policies full,none,epic1,epic2,cacheblend2,cacheblend4,cacheblend15 \
  --encode_prefix template > results/ablation_rog_cwq.txt 2>> "$LOG"
cm "ablation"

echo "######## OVERNIGHT DONE $(date) ########" >> "$LOG"
cm "ALL DONE"
