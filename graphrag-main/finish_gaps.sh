#!/bin/bash
# ============================================================================
# Gap-filler: runs AFTER the master pipeline (run_all_remaining.sh) finishes.
# Closes the two remaining gaps and produces the corrected FINAL table:
#   (1) OPT-13b/30b CWQ+WEB Direct/R -- earlier FAILED because the HF weights
#       were not yet fully downloaded; the blobs are now complete (48G/112G).
#   (2) Re-assemble with gemma_fix (bf16) listed FIRST so the authoritative
#       bf16 Gemma-2-27b CWQ wins over the fp16-overflow 0.0000 rows that the
#       float16 run wrote into table2_subset.tsv (assemble_table2 is first-wins
#       and does NOT skip literal 0.0000, only the string "FAILED").
# Designed to run detached in its own tmux session so it survives logout:
#   tmux new -d -s gaps 'bash finish_gaps.sh'
# Commits + pushes after each step.
# ============================================================================
set -u
cd /home/am137688/amir/graphRag/graphrag_kv
source /home/am137688/anaconda3/etc/profile.d/conda.sh
conda activate graphrag
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG=results/finish_gaps.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
cm(){ git add -A 2>/dev/null; git commit -q -m "finish_gaps: $1 [$(date +%H:%M)]" 2>/dev/null; git push -q origin main 2>/dev/null; say "committed: $1"; }

say "######## FINISH_GAPS START ########"

# ---- 1) wait for the master pipeline to complete (no GPU contention) --------
say "waiting for master run_all_remaining.sh to finish before touching GPUs..."
while pgrep -f run_all_remaining.sh >/dev/null; do sleep 60; done
say "master finished. Proceeding with gap fills."

# ---- 2) OPT-13b/30b CWQ+WEB+MQA Direct/R (non-RoPE -> Direct/GraphRAG-R only) -
# These FAILED in run_cwqweb (CWQ/WEB) AND in run_mqa (MQA) because the HF
# snapshot was missing pytorch_model.bin.index.json (sharded .bin, no index ->
# transformers couldn't find the checkpoint). Index fetched (network reachable),
# verified to reference all shards. Now refill all three datasets.
OPTFIX=results/table2_optfix.tsv
printf "model\tdataset\tmethod\tHits@1\tF1\tsubstr\tn\tsecs\n" > "$OPTFIX"
for M in facebook/opt-13b facebook/opt-30b; do
  say "OPT re-run (index.json fixed): $M  CWQ+WEB+MQA direct/graphrag_r N=500"
  python run_table2.py --models "$M" --methods direct graphrag_r \
    --datasets rmanluo/RoG-cwq rmanluo/RoG-webqsp metaqa-3hop --paths_dir precomputed_paths \
    --max_new_tokens 64 --n_samples 500 --results "$OPTFIX" --outdir /tmp/optfix >> "$LOG" 2>&1
  cm "opt fix $M"
done

# ---- 3) re-assemble FINAL table with corrected source order ----------------
# assemble_table2 --tsv merge is LAST-WINS (data[key]=... with no skip), so the
# authoritative sources go LAST: table2_gemma_fix.tsv (bf16 Gemma) LAST so its
# Gemma-2-27b/9b CWQ beat the fp16 0.0000 rows in subset; table2_mqa.tsv before
# it so the guided MQA sweep beats overnight's stale max_paths=8 MQA. optfix adds
# OPT-13b/30b (FAILED+skipped elsewhere, no collision).
TSVS="results/table2_overnight.tsv results/table2_subset.tsv results/table2_mqa.tsv results/table2_optfix.tsv results/table2_gemma_fix.tsv"
PREDS="results/preds_gemma results/preds_full results/preds_subset /tmp/optfix"
say "re-assembling FINAL table + literature alignment (corrected order)"
python check_literature_alignment.py > results/literature_alignment.txt 2>> "$LOG" || true
python assemble_table2.py --metric hits --latex --tsv $TSVS --preds $PREDS \
  > results/FINAL_TABLE_hits.txt 2>> "$LOG" || true
python assemble_table2.py --metric f1 --tsv $TSVS --preds $PREDS \
  > results/FINAL_TABLE_f1.txt 2>> "$LOG" || true

{ echo "######## FINAL SUMMARY (gaps filled, corrected order) $(date) ########"; echo;
  echo "=== Literature alignment ==="; cat results/literature_alignment.txt 2>/dev/null; echo;
  echo "=== Ablation (R1, leak-free) ==="; tail -15 results/ablation_rog_cwq.txt 2>/dev/null; echo;
  echo "=== Final Table 2 (Hits@1 + paired reuse stats) ==="; cat results/FINAL_TABLE_hits.txt 2>/dev/null;
} > results/FINAL_SUMMARY.txt
cm "finish_gaps: FINAL assembly (OPT-13b/30b filled + bf16 gemma-27b authoritative)"

say "######## FINISH_GAPS DONE -> results/FINAL_SUMMARY.txt ########"
