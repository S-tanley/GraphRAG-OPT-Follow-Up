#!/bin/bash
# Large-model GraphRAG-Opt via MODEL-PARALLEL device-aware KV reuse.
# Qwen2.5-72B-Instruct (standard RoPE full-attention, normal KV cache) loaded
# bf16 across 4 GPUs (device_map=auto); reuse works because hf_reuse is now
# device-aware (_layer_kv_device). Smoke test: CWQ Opt 0.5417 == R 0.5417 (N=24).
# 2 data-parallel shards (8 GPUs / 4 per replica). CWQ first so the headline
# large-model reuse row lands soonest. Commits after each dataset.
set -u
cd /home/am137688/amir/graphRag/graphrag_kv
source /home/am137688/anaconda3/etc/profile.d/conda.sh
conda activate graphrag
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

M="Qwen/Qwen2.5-72B-Instruct"
RES=results/table2_qwen72.tsv
LOG=results/qwen72.log
[ -f "$RES" ] || printf "model\tdataset\tmethod\tHits@1\tF1\tsubstr\tn\tsecs\n" > "$RES"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
cm(){ git add -A 2>/dev/null; git commit -q -m "qwen72: $1 [$(date +%H:%M)]" 2>/dev/null; git push -q origin main 2>/dev/null; say "committed: $1"; }
run(){ python run_table2.py --models "$M" --methods direct graphrag_r graphrag_opt graphrag_opt_rc \
       --datasets "$1" --paths_dir precomputed_paths --max_new_tokens 64 --n_samples 500 \
       --results "$RES" --outdir /tmp/qwen72 >> "$LOG" 2>&1; }

say "######## QWEN72 START (model-parallel device-aware reuse) ########"
for DS in rmanluo/RoG-cwq rmanluo/RoG-webqsp metaqa-3hop; do
  say "running $M on $DS (4 cols, N=500, MP=4)"
  run "$DS"
  cm "qwen72 $DS"
done

# fold into FINAL table (qwen72 LAST)
TSVS="results/table2_overnight.tsv results/table2_subset.tsv results/table2_mqa.tsv results/table2_optfix.tsv results/table2_optreuse.tsv results/table2_gemma_fix.tsv results/table2_qwen72.tsv"
PREDS="results/preds_gemma results/preds_full results/preds_subset /tmp/optfix /tmp/optreuse /tmp/qwen72"
python assemble_table2.py --metric hits --latex --tsv $TSVS --preds $PREDS > results/FINAL_TABLE_hits.txt 2>> "$LOG"
python assemble_table2.py --metric f1 --tsv $TSVS --preds $PREDS > results/FINAL_TABLE_f1.txt 2>> "$LOG"
python check_paper_coverage.py >> "$LOG" 2>&1 || true
cm "qwen72 folded into FINAL table + coverage"
say "######## QWEN72 DONE ########"
