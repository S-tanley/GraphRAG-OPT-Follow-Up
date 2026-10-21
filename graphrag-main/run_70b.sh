#!/bin/bash
# 70B-class RoPE models in NF4 4-bit on a SINGLE GPU so the KV-reuse path stays
# valid (bf16 70B needs model-parallel -> breaks reuse, like Qwen3-80B). Weights
# 4-bit, KV cache fp16. Qwen2.5-72B is already downloaded (ungated); Llama uses an
# ungated mirror (meta-llama/Llama-3.1-70B-Instruct is gated/unauthorized for this
# token). Each model runs as soon as ITS weights are present (Qwen first).
set -u
cd /home/am137688/amir/graphRag/graphrag_kv
source /home/am137688/anaconda3/etc/profile.d/conda.sh
conda activate graphrag
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RES=results/table2_70b.tsv
LOG=results/run70b.log
M_QWEN="Qwen/Qwen2.5-72B-Instruct"
M_LLAMA="NousResearch/Meta-Llama-3.1-70B-Instruct"
[ -f "$RES" ] || printf "model\tdataset\tmethod\tHits@1\tF1\tsubstr\tn\tsecs\n" > "$RES"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
cm(){ git add -A 2>/dev/null; git commit -q -m "70b: $1 [$(date +%H:%M)]" 2>/dev/null; git push -q origin main 2>/dev/null; say "committed: $1"; }
run(){ python run_table2.py "$@" --paths_dir precomputed_paths --max_new_tokens 64 --n_samples 500 --load_4bit --results "$RES" --outdir /tmp/run70b >> "$LOG" 2>&1; }
# wait until a model's snapshot has >=10 safetensors shards and no .incomplete
wait_model(){ local M="$1"; local d
  d=$(echo "$M" | sed 's#/#--#'); d="$HOME/.cache/huggingface/hub/models--$d"
  say "waiting for weights: $M"
  while :; do
    local n inc
    n=$(find "$d" -name '*.safetensors' 2>/dev/null | wc -l)
    inc=$(find "$d" -name '*.incomplete' 2>/dev/null | wc -l)
    [ "$n" -ge 10 ] && [ "$inc" -eq 0 ] && break
    sleep 60
  done
  say "weights present: $M ($n shards)"
}

say "######## RUN_70B START ########"
SMOKE_DONE=0; METHODS="direct graphrag_r graphrag_opt graphrag_opt_rc"
for M in "$M_QWEN" "$M_LLAMA"; do
  wait_model "$M"
  if [ "$SMOKE_DONE" = 0 ]; then
    say "smoke test 4-bit reuse: $M graphrag_opt N=40"
    CUDA_VISIBLE_DEVICES=0 python table2_worker.py --model "$M" --dataset rmanluo/RoG-cwq \
      --method graphrag_opt --n_samples 40 --max_paths 8 --max_new_tokens 64 --reuse_budget 4 \
      --recompute_k 0 --load_4bit --shard_id 0 --num_shards 1 \
      --paths_file precomputed_paths/paths_rmanluo-RoG-cwq.json --out_json /tmp/smoke70b.json >> "$LOG" 2>&1
    if [ -s /tmp/smoke70b.json ]; then say "4-bit reuse smoke OK -> all 4 methods"
    else METHODS="direct graphrag_r"; say "4-bit reuse smoke FAILED -> Direct/R only"; fi
    SMOKE_DONE=1
  fi
  say "70B run: $M methods=[$METHODS]"
  run --models "$M" --methods $METHODS --datasets rmanluo/RoG-cwq rmanluo/RoG-webqsp metaqa-3hop
  cm "70b $M"
done

# fold into FINAL table (70b LAST)
TSVS="results/table2_overnight.tsv results/table2_subset.tsv results/table2_mqa.tsv results/table2_optfix.tsv results/table2_optreuse.tsv results/table2_gemma_fix.tsv results/table2_70b.tsv"
PREDS="results/preds_gemma results/preds_full results/preds_subset /tmp/optfix /tmp/optreuse /tmp/run70b"
python assemble_table2.py --metric hits --latex --tsv $TSVS --preds $PREDS > results/FINAL_TABLE_hits.txt 2>> "$LOG"
python assemble_table2.py --metric f1 --tsv $TSVS --preds $PREDS > results/FINAL_TABLE_f1.txt 2>> "$LOG"
python check_literature_alignment.py > results/literature_alignment.txt 2>> "$LOG" || true
cm "70b folded into FINAL table"
say "######## RUN_70B DONE ########"
