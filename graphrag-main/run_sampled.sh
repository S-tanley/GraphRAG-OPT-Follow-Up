#!/bin/bash
# Resource-bounded coverage of the REMAINING models via a fixed-seed N=500
# subsample (seed 42, same question IDs across all systems -> paired, literature
# standard; CacheBlend/EPIC both evaluate on subsets). The 4 primary small models
# (RoG/Llama-2-7b/Gemma-2-9b/Qwen3-8b) already have FULL CWQ/WEB banked in
# results/table2_overnight.tsv -- this script only ADDS the un-run cells.
# Commits + pushes after every model so nothing is lost.
set -u
cd /home/am137688/amir/graphRag/graphrag_kv
source /home/am137688/anaconda3/etc/profile.d/conda.sh
conda activate graphrag
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

mkdir -p results
RES=results/table2_sampled.tsv          # separate file; merged at assembly time
LOG=results/sampled.log
N=500
[ -f "$RES" ] || printf "model\tdataset\tmethod\tHits@1\tF1\tsubstr\tn\tsecs\n" > "$RES"
CW="rmanluo/RoG-cwq"; WE="rmanluo/RoG-webqsp"; MQ="metaqa-3hop"

cm()  { git add -A 2>/dev/null; git commit -q -m "sampled(N=$N): $1 [$(date +%H:%M)]" 2>/dev/null; git push -q origin main 2>/dev/null; echo "  [committed $1 $(date +%H:%M)]" >> "$LOG"; }
run() { echo "=== $(date +%H:%M) run_table2 $* ===" >> "$LOG"; python run_table2.py "$@" --paths_dir precomputed_paths --max_new_tokens 64 --n_samples $N --results "$RES" --outdir /tmp/sm >> "$LOG" 2>&1; }

echo "######## SAMPLED START $(date) ########" >> "$LOG"

# ---- Finish Qwen3-8b MQA reuse columns (only Direct ran before the stop) ----
run --models Qwen/Qwen3-8B --methods graphrag_r graphrag_opt graphrag_opt_rc --datasets $MQ
cm "qwen3-8b MQA reuse"

# ---- Mid RoPE models that fit on ONE 80GB GPU -> full 4 columns (reuse valid) ----
for M in meta-llama/Llama-2-13b-hf google/gemma-2-27b Qwen/Qwen3-32B; do
  run --models "$M" --datasets $CW $WE $MQ        # all 4 methods (default)
  cm "mid $M"
done

# ---- 80B (model-parallel) + OPT (non-RoPE): Direct/GraphRAG-R only (reuse path
#      is single-GPU RoPE-only; these establish the integrity/accuracy columns) --
for M in Qwen/Qwen3-Next-80B-A3B-Thinking facebook/opt-6.7b facebook/opt-13b facebook/opt-30b; do
  run --models "$M" --methods direct graphrag_r --datasets $CW $WE $MQ
  cm "big/opt $M"
done

# ---- R1 evidence: CacheBlend/EPIC head-to-head ablation on RoG/CWQ ----
echo "=== $(date +%H:%M) ablation_kgqa ===" >> "$LOG"
python ablation_kgqa.py --model rmanluo/RoG --dataset rmanluo/RoG-cwq --path_enum \
  --n_samples $N --policies full,none,epic1,epic2,cacheblend2,cacheblend4,cacheblend15 \
  --encode_prefix template > results/ablation_rog_cwq.txt 2>> "$LOG"
cm "ablation"

echo "######## SAMPLED DONE $(date) ########" >> "$LOG"
cm "ALL DONE"
