#!/bin/bash
# OPT family KV-reuse columns. Smoke test confirmed OPT (non-RoPE, learned abs.
# positions) tolerates the reused KV: OPT-6.7b CWQ Opt 0.260 ~= R 0.267 (Prompt
# Cache, MLSys'24: discontinuous position IDs are fine if intra-segment relative
# order is preserved). The engine already skips RoPE-realign for non-RoPE, so no
# code change -- just run GraphRAG-Opt / +RC for the 3 OPT models, N=500.
set -u
cd /home/am137688/amir/graphRag/graphrag_kv
source /home/am137688/anaconda3/etc/profile.d/conda.sh
conda activate graphrag
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RES=results/table2_optreuse.tsv
LOG=results/optreuse.log
[ -f "$RES" ] || printf "model\tdataset\tmethod\tHits@1\tF1\tsubstr\tn\tsecs\n" > "$RES"
cm(){ git add -A 2>/dev/null; git commit -q -m "opt-reuse: $1 [$(date +%H:%M)]" 2>/dev/null; git push -q origin main 2>/dev/null; echo "[committed $1 $(date +%H:%M)]" >> "$LOG"; }
run(){ echo "=== $(date +%H:%M) $* ===" >> "$LOG"; python run_table2.py "$@" --paths_dir precomputed_paths --max_new_tokens 64 --n_samples 500 --results "$RES" --outdir /tmp/optreuse >> "$LOG" 2>&1; }

echo "######## OPT-REUSE START $(date) ########" >> "$LOG"
for M in facebook/opt-6.7b facebook/opt-13b facebook/opt-30b; do
  run --models "$M" --methods graphrag_opt graphrag_opt_rc \
      --datasets rmanluo/RoG-cwq rmanluo/RoG-webqsp metaqa-3hop
  cm "opt reuse $M"
done

# fold into the FINAL table: optreuse LAST so its Opt/+RC cells fill the "--"
TSVS="results/table2_overnight.tsv results/table2_subset.tsv results/table2_mqa.tsv results/table2_optfix.tsv results/table2_gemma_fix.tsv results/table2_optreuse.tsv"
PREDS="results/preds_gemma results/preds_full results/preds_subset /tmp/optfix /tmp/optreuse"
python assemble_table2.py --metric hits --latex --tsv $TSVS --preds $PREDS > results/FINAL_TABLE_hits.txt 2>> "$LOG"
python assemble_table2.py --metric f1 --tsv $TSVS --preds $PREDS > results/FINAL_TABLE_f1.txt 2>> "$LOG"
python check_literature_alignment.py > results/literature_alignment.txt 2>> "$LOG" || true
cm "opt reuse folded into FINAL table"
echo "######## OPT-REUSE DONE $(date) ########" >> "$LOG"
