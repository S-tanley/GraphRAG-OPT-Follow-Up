#!/bin/bash
# ============================================================================
# MASTER unattended pipeline for the remaining GraphRAG-Opt rebuttal runs.
# Runs every remaining GPU job to completion, sequentially (no contention),
# committing+pushing after each phase. Designed to run detached in tmux and
# survive logout. A FINAL summary is written to results/FINAL_SUMMARY.txt.
#
#   tmux new -d -s grrun 'bash run_all_remaining.sh'
#   tmux attach -t grrun     # to watch
# ============================================================================
set -u
cd /home/am137688/amir/graphRag/graphrag_kv
source /home/am137688/anaconda3/etc/profile.d/conda.sh
conda activate graphrag
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG=results/master.log
mkdir -p results/preds_subset results/preds_gemma
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
cm(){ git add -f results/*.tsv results/preds_subset results/preds_gemma results/*.txt results/*.log 2>/dev/null
      git add -A 2>/dev/null
      git commit -q -m "master: $1 [$(date +%H:%M)]" 2>/dev/null
      git push -q origin main 2>/dev/null; say "committed: $1"; }

say "######## MASTER START ########"

# ---- PHASE 0: wait for the already-running CWQ/WEB sweep to finish ----
say "PHASE 0: waiting for run_cwqweb.sh (Qwen3-80B + OPT remaining)"
while pgrep -f run_cwqweb.sh >/dev/null; do
  grep -q "CWQWEB DONE" results/cwqweb.log 2>/dev/null && break
  sleep 30
done
say "PHASE 0 done. CWQ/WEB rows: $(tail -n+2 results/table2_subset.tsv 2>/dev/null | wc -l)"
cm "phase0 cwqweb complete"

# ---- PHASE 1: Gemma bf16 fixes (float16 overflowed -> empty/zero on Gemma-2) ----
# Worker now auto-selects bfloat16 for Gemma. Re-run the affected CWQ/WEB cells.
GFIX=results/table2_gemma_fix.tsv
printf "model\tdataset\tmethod\tHits@1\tF1\tsubstr\tn\tsecs\n" > "$GFIX"
say "PHASE 1: Gemma-2-9b CWQ+WEB FULL (bf16, authoritative) + Gemma-2-27b CWQ N=500 (bf16)"
# Gemma-2-9b: re-run primaries' CWQ/WEB at FULL test set in bf16 (overrides overnight fp16)
python run_table2.py --models google/gemma-2-9b --datasets rmanluo/RoG-cwq rmanluo/RoG-webqsp \
  --methods direct graphrag_r graphrag_opt graphrag_opt_rc --paths_dir precomputed_paths \
  --max_new_tokens 64 --results "$GFIX" --outdir results/preds_gemma >> "$LOG" 2>&1
cm "phase1 gemma-9b cwq/web bf16"
# Gemma-2-27b: re-run CWQ at N=500 in bf16 (overrides the float16 zeros)
python run_table2.py --models google/gemma-2-27b --datasets rmanluo/RoG-cwq \
  --methods direct graphrag_r graphrag_opt graphrag_opt_rc --paths_dir precomputed_paths \
  --max_new_tokens 64 --n_samples 500 --results "$GFIX" --outdir results/preds_gemma >> "$LOG" 2>&1
cm "phase1 gemma-27b cwq bf16"
say "PHASE 1 done."

# ---- PHASE 2: MetaQA sweep for ALL models with the guided leak-free retriever ----
say "PHASE 2: MQA sweep (all 11 models, guided retriever, max_paths=20)"
MAXP=20 bash run_mqa.sh >> "$LOG" 2>&1
say "PHASE 2 done. MQA rows: $(tail -n+2 results/table2_mqa.tsv 2>/dev/null | wc -l)"
cm "phase2 mqa sweep"

# ---- PHASE 3: leak-free CacheBlend/EPIC ablation (R1 evidence) ----
say "PHASE 3: leak-free ablation (RoG/CWQ, N=500, RoG-plan paths)"
python run_full_ablation.py --model rmanluo/RoG --dataset rmanluo/RoG-cwq \
  --paths_dir precomputed_paths --n_samples 500 \
  --policies full,none,epic1,epic2,cacheblend2,cacheblend4,cacheblend15 \
  --encode_prefix template > results/ablation_rog_cwq.txt 2>> "$LOG"
say "PHASE 3 done."
cm "phase3 ablation"

# ---- PHASE 4: assemble final table + literature alignment + summary ----
say "PHASE 4: assemble final Table 2 + literature check"
python check_literature_alignment.py > results/literature_alignment.txt 2>> "$LOG" || true
python assemble_table2.py --metric hits --latex \
  --tsv results/table2_overnight.tsv results/table2_subset.tsv results/table2_gemma_fix.tsv results/table2_mqa.tsv \
  --preds results/preds_gemma results/preds_full results/preds_subset \
  > results/FINAL_TABLE_hits.txt 2>> "$LOG" || true
python assemble_table2.py --metric f1 \
  --tsv results/table2_overnight.tsv results/table2_subset.tsv results/table2_gemma_fix.tsv results/table2_mqa.tsv \
  --preds results/preds_gemma results/preds_full results/preds_subset \
  > results/FINAL_TABLE_f1.txt 2>> "$LOG" || true

{ echo "######## FINAL SUMMARY $(date) ########"; echo;
  echo "=== Literature alignment ==="; cat results/literature_alignment.txt 2>/dev/null; echo;
  echo "=== Ablation (R1, leak-free) ==="; tail -12 results/ablation_rog_cwq.txt 2>/dev/null; echo;
  echo "=== Final Table 2 (Hits@1 + paired reuse stats) ==="; cat results/FINAL_TABLE_hits.txt 2>/dev/null;
} > results/FINAL_SUMMARY.txt
cm "phase4 final assembly + summary"

say "######## MASTER DONE — see results/FINAL_SUMMARY.txt ########"
