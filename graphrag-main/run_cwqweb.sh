#!/bin/bash
# Phase A: CWQ + WEB at N=500 for the un-run models (primaries' CWQ/WEB are already
# banked full-set in table2_overnight.tsv + preds_full). MetaQA is handled separately
# (run_mqa.sh) after the improved leak-free MetaQA retriever is built, so no GPU is
# wasted on soon-to-be-replaced MQA cells. Keeps per-shard predictions for paired stats.
set -u
cd /home/am137688/amir/graphRag/graphrag_kv
source /home/am137688/anaconda3/etc/profile.d/conda.sh
conda activate graphrag
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1

mkdir -p results/preds_subset
RES=results/table2_subset.tsv
LOG=results/cwqweb.log
OUT=results/preds_subset
printf "model\tdataset\tmethod\tHits@1\tF1\tsubstr\tn\tsecs\n" > "$RES"   # fresh (old rows were old-path MQA)
CW="rmanluo/RoG-cwq"; WE="rmanluo/RoG-webqsp"; N=500

cm()  { git add -f results/table2_subset.tsv results/preds_subset 2>/dev/null; git commit -q -m "cwqweb N=500: $1 [$(date +%H:%M)]" 2>/dev/null; git push -q origin main 2>/dev/null; echo "  [committed $1 $(date +%H:%M)]" >> "$LOG"; }
run() { echo "=== $(date +%H:%M) run_table2 $* ===" >> "$LOG"; python run_table2.py "$@" --datasets $CW $WE --paths_dir precomputed_paths --max_new_tokens 64 --n_samples $N --results "$RES" --outdir "$OUT" >> "$LOG" 2>&1; }

echo "######## CWQWEB START $(date) ########" >> "$LOG"

# Mid/large RoPE models (reuse applies -> 4 methods)
for M in meta-llama/Llama-2-13b-hf google/gemma-2-27b Qwen/Qwen3-32B; do
  run --models "$M" --methods direct graphrag_r graphrag_opt graphrag_opt_rc
  cm "rope $M"
done

# Qwen3-Next-80B (MoE, model-parallel auto). Try 4 methods; opt cells log FAILED if
# the arch's reuse path is unsupported, leaving Direct/GraphRAG-R intact.
run --models Qwen/Qwen3-Next-80B-A3B-Thinking --methods direct graphrag_r graphrag_opt graphrag_opt_rc
cm "qwen3-80b"

# OPT family (non-RoPE -> reuse N/A): Direct + GraphRAG-R only
for M in facebook/opt-6.7b facebook/opt-13b facebook/opt-30b; do
  run --models "$M" --methods direct graphrag_r
  cm "opt $M"
done

echo "######## CWQWEB DONE $(date) ########" >> "$LOG"
cm "cwqweb ALL DONE"
