#!/bin/bash
# Phase B: MetaQA-3hop at N=500 for ALL models, using the improved leak-free
# rule-based MetaQA retriever (precomputed_paths/paths_metaqa-3hop.json, rebuilt by
# precompute_metaqa_rule.py). Run AFTER that file is rebuilt. Writes to a separate
# TSV (table2_mqa.tsv) so the assembler can load it last (authoritative MQA cells),
# and keeps per-shard predictions for paired stats.
set -u
cd /home/am137688/amir/graphRag/graphrag_kv
source /home/am137688/anaconda3/etc/profile.d/conda.sh
conda activate graphrag
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1

mkdir -p results/preds_subset
RES=results/table2_mqa.tsv
LOG=results/mqa.log
OUT=results/preds_subset
MQ="metaqa-3hop"; N=500; MAXP=${MAXP:-20}
printf "model\tdataset\tmethod\tHits@1\tF1\tsubstr\tn\tsecs\n" > "$RES"

cm()  { git add -f results/table2_mqa.tsv results/preds_subset 2>/dev/null; git commit -q -m "mqa N=500 (rule retriever): $1 [$(date +%H:%M)]" 2>/dev/null; git push -q origin main 2>/dev/null; echo "  [committed $1 $(date +%H:%M)]" >> "$LOG"; }
run() { echo "=== $(date +%H:%M) run_table2 $* ===" >> "$LOG"; python run_table2.py "$@" --datasets $MQ --paths_dir precomputed_paths --max_paths $MAXP --max_new_tokens 64 --n_samples $N --results "$RES" --outdir "$OUT" >> "$LOG" 2>&1; }

echo "######## MQA START $(date) (max_paths=$MAXP) ########" >> "$LOG"

# RoPE models -> all 4 columns (reuse applies)
for M in rmanluo/RoG meta-llama/Llama-2-7b-hf meta-llama/Llama-2-13b-hf \
         google/gemma-2-9b google/gemma-2-27b Qwen/Qwen3-8B Qwen/Qwen3-32B \
         Qwen/Qwen3-Next-80B-A3B-Thinking; do
  run --models "$M" --methods direct graphrag_r graphrag_opt graphrag_opt_rc
  cm "rope-mqa $M"
done

# OPT family (non-RoPE) -> Direct + GraphRAG-R only
for M in facebook/opt-6.7b facebook/opt-13b facebook/opt-30b; do
  run --models "$M" --methods direct graphrag_r
  cm "opt-mqa $M"
done

echo "######## MQA DONE $(date) ########" >> "$LOG"
cm "mqa ALL DONE"
