#!/bin/bash
# N=500 fixed-seed (seed 42) subset pipeline for the un-run / paired-stats cells.
# Uniform protocol: identical question IDs across models+methods (sampler shuffles
# seed 42 then selects range(500) BEFORE sharding -> paired). Per-shard prediction
# JSONs are kept in results/preds_subset so paired McNemar / non-inferiority can be
# computed downstream. max_new_tokens 64 matches the banked overnight full-set runs.
# Commits + pushes after every model so nothing is lost.
set -u
cd /home/am137688/amir/graphRag/graphrag_kv
source /home/am137688/anaconda3/etc/profile.d/conda.sh
conda activate graphrag
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1

mkdir -p results/preds_subset
RES=results/table2_subset.tsv
LOG=results/subset.log
OUT=results/preds_subset
[ -f "$RES" ] || printf "model\tdataset\tmethod\tHits@1\tF1\tsubstr\tn\tsecs\n" > "$RES"
CW="rmanluo/RoG-cwq"; WE="rmanluo/RoG-webqsp"; MQ="metaqa-3hop"
N=500

cm()  { git add -A 2>/dev/null; git commit -q -m "subset N=500: $1 [$(date +%H:%M)]" 2>/dev/null; git push -q origin main 2>/dev/null; echo "  [committed $1 $(date +%H:%M)]" >> "$LOG"; }
run() { echo "=== $(date +%H:%M) run_table2 $* ===" >> "$LOG"; python run_table2.py "$@" --paths_dir precomputed_paths --max_new_tokens 64 --n_samples $N --results "$RES" --outdir "$OUT" >> "$LOG" 2>&1; }

echo "######## SUBSET START $(date) ########" >> "$LOG"

# --- (1) Primary models: MQA reuse columns at N=500 (clean MQA pairing; CWQ/WEB
#         stay full-set from the banked overnight run + preds_full). 4 methods. ---
for M in rmanluo/RoG meta-llama/Llama-2-7b-hf google/gemma-2-9b Qwen/Qwen3-8B; do
  run --models "$M" --datasets $MQ --methods direct graphrag_r graphrag_opt graphrag_opt_rc
  cm "primary-MQA $M"
done

# --- (2) Mid/large RoPE models: all datasets, all 4 columns (reuse applies). ---
for M in meta-llama/Llama-2-13b-hf google/gemma-2-27b Qwen/Qwen3-32B; do
  run --models "$M" --datasets $CW $WE $MQ --methods direct graphrag_r graphrag_opt graphrag_opt_rc
  cm "rope $M"
done

# --- (3) Qwen3-Next-80B (MoE, model-parallel auto gps=4). Try 4 methods; if the
#         reuse path is unsupported for this arch the opt cells log FAILED and the
#         Direct/GraphRAG-R integrity columns still land. ---
run --models Qwen/Qwen3-Next-80B-A3B-Thinking --datasets $CW $WE $MQ \
    --methods direct graphrag_r graphrag_opt graphrag_opt_rc
cm "qwen3-80b"

# --- (4) OPT family (non-RoPE: absolute position emb -> KV reuse N/A). Integrity
#         columns Direct + GraphRAG-R only, all datasets. ---
for M in facebook/opt-6.7b facebook/opt-13b facebook/opt-30b; do
  run --models "$M" --datasets $CW $WE $MQ --methods direct graphrag_r
  cm "opt $M"
done

echo "######## SUBSET DONE $(date) ########" >> "$LOG"
cm "subset ALL DONE"
