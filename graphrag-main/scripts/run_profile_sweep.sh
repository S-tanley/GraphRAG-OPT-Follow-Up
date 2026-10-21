#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# run_profile_sweep.sh  —  Full RAG vs GraphRAG profiling sweep (ISCA paper)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Every experiment appends its results to a single CSV immediately after it
# finishes.  Failed experiments get a row with status=FAILED so nothing is lost.
#
# Usage:
#   ./run_profile_sweep.sh              # run full sweep
#   ./run_profile_sweep.sh --resume     # skip already-completed jobs
#   ./run_profile_sweep.sh --fresh      # wipe checkpoint, start over
#   ./run_profile_sweep.sh --dry-run    # print commands without executing
#
# ═══════════════════════════════════════════════════════════════════════════════

set -uo pipefail

# ─── Config ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON="${PYTHON:-/home/am137688/anaconda3/envs/graphrag/bin/python}"

OUT_DIR="$SCRIPT_DIR/profile_results/sweep_$(date +%Y%m%d)"
CHECKPOINT="$OUT_DIR/checkpoint.txt"
MERGED_CSV="$OUT_DIR/sweep_results.csv"

mkdir -p "$OUT_DIR"
touch "$CHECKPOINT"

# ─── Sweep parameters ────────────────────────────────────────────────────────
MODELS=(
  "google/gemma-2-9b"
  "Qwen/Qwen3-8B"
  "facebook/opt-13b"
)

DATASETS=(
  "rmanluo/RoG-cwq"
  "metaqa-3hop"
)

HOPS="1 2 3 4"

BATCH_SIZE=16
SAMPLES=128
MAX_TOKENS=50
NUM_BEAMS=1
SEED=42
DEVICE=0

# Pin to a single GPU to avoid RoPE inv_freq landing on wrong device.
export CUDA_VISIBLE_DEVICES="$DEVICE"

# ─── Parse args ───────────────────────────────────────────────────────────────
DRY_RUN=false
RESUME=true
for arg in "$@"; do
  case "$arg" in
    --dry-run)  DRY_RUN=true ;;
    --resume)   RESUME=true ;;
    --fresh)    RESUME=false; : > "$CHECKPOINT" ;;
    *)          echo "Unknown arg: $arg"; exit 1 ;;
  esac
done

# ─── Helpers ──────────────────────────────────────────────────────────────────
slug() { echo "$1" | sed 's|/|_|g; s| |_|g'; }

is_done() {
  grep -qF "$1|$2" "$CHECKPOINT" 2>/dev/null
}

# ─── CSV header (written once) ────────────────────────────────────────────────
# Matches the columns from _save_flat_csv in profile_rag_vs_graphrag.py,
# with status, timestamp, duration_s, and error prepended.
CSV_HEADER="status,timestamp,duration_s,error,experiment,config,model,dataset,use_kv,hop,top_k,batch_size,n_samples,model_B,model_GB,hits_at_1,f1,retrieval_ms_mean,ttft_ms_mean,decode_ms_mean,e2e_ms_mean,e2e_ms_p95,gen_tok_per_s,queries_per_s,prefill_tok_s,decode_tok_s,prefill_arith_intensity,decode_arith_intensity,prefill_mfu,decode_mfu,decode_bw_GBs,decode_bw_pct_peak,peak_hbm_GB,kv_cache_GB,energy_per_query_J,mean_power_W,mean_sm_pct,retrieval_pct,prefill_pct,decode_pct,avg_input_tok,avg_ctx_tok,avg_output_tok,avg_paths_per_query,avg_graph_nodes,avg_graph_edges,avg_graph_diameter,avg_graph_longest_path,avg_graph_avg_degree,avg_graph_max_degree,avg_graph_density,avg_graph_components,avg_graph_n_relations,avg_per_query_path_len"

# Total columns in final CSV = 4 extra + 50 data = 54
N_TOTAL_COLS=54

# Write header if the merged CSV doesn't exist or is empty
if [[ ! -s "$MERGED_CSV" ]]; then
  echo "$CSV_HEADER" > "$MERGED_CSV"
fi

# ─── Count jobs ───────────────────────────────────────────────────────────────
TOTAL_JOBS=$(( ${#MODELS[@]} * ${#DATASETS[@]} ))
COMPLETED=0
FAILED=0
SKIPPED=0

if $RESUME; then
  for m in "${MODELS[@]}"; do
    for d in "${DATASETS[@]}"; do
      is_done "$m" "$d" && SKIPPED=$((SKIPPED + 1))
    done
  done
fi
REMAINING=$((TOTAL_JOBS - SKIPPED))

# ─── Header ───────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  RAG vs GraphRAG — ISCA Profile Sweep"
echo "  Models:   ${MODELS[*]}"
echo "  Datasets: ${DATASETS[*]}"
echo "  Hops:     $HOPS"
echo "  Sizing:   batch=$BATCH_SIZE  samples=$SAMPLES  device=$DEVICE"
echo "  Output:   $OUT_DIR"
echo "  CSV:      $MERGED_CSV"
echo "  Jobs:     $TOTAL_JOBS total, $SKIPPED already done, $REMAINING to run"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

if $DRY_RUN; then
  echo "[DRY RUN] Commands that would execute:"
  echo ""
fi

# ─── Main sweep loop ─────────────────────────────────────────────────────────
JOB_IDX=0
SWEEP_START=$(date +%s)

for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    JOB_IDX=$((JOB_IDX + 1))
    m_slug=$(slug "$model")
    d_slug=$(slug "$dataset")
    tag="${m_slug}__${d_slug}"
    job_csv="$OUT_DIR/${tag}_profile.csv"
    job_log="$OUT_DIR/${tag}.log"

    # ── Skip if already completed ──────────────────────────────────────
    if $RESUME && is_done "$model" "$dataset"; then
      echo "  [$JOB_IDX/$TOTAL_JOBS] SKIP  $model × $dataset (already done)"
      continue
    fi

    # ── Build command ──────────────────────────────────────────────────
    CMD="$PYTHON profile_rag_vs_graphrag.py \
      --experiments graph_char \
      --llm_model '$model' \
      --dataset '$dataset' \
      --hops $HOPS \
      --batch_size $BATCH_SIZE \
      --samples $SAMPLES \
      --max_tokens $MAX_TOKENS \
      --num_beams $NUM_BEAMS \
      --seed $SEED \
      --device $DEVICE \
      --out_dir '$OUT_DIR'"

    if $DRY_RUN; then
      echo "  [$JOB_IDX/$TOTAL_JOBS] $model × $dataset"
      echo "    $CMD"
      echo ""
      continue
    fi

    # ── Run ────────────────────────────────────────────────────────────
    echo ""
    echo "┌─────────────────────────────────────────────────────────────────"
    echo "│ Job $JOB_IDX/$TOTAL_JOBS: $model × $dataset"
    echo "│ Log: $job_log"
    echo "└─────────────────────────────────────────────────────────────────"

    JOB_START=$(date +%s)
    ts=$(date -Iseconds)

    eval "$CMD" > "$job_log" 2>&1
    exit_code=$?

    JOB_END=$(date +%s)
    JOB_ELAPSED=$((JOB_END - JOB_START))

    if [[ $exit_code -eq 0 && -f "$job_csv" ]]; then
      # ── Success: append each data row from the per-job CSV ────────
      COMPLETED=$((COMPLETED + 1))
      echo "$model|$dataset" >> "$CHECKPOINT"

      # Read per-job CSV, skip header, prepend status columns
      while IFS= read -r line; do
        echo "OK,${ts},${JOB_ELAPSED},,${line}" >> "$MERGED_CSV"
      done < <(tail -n +2 "$job_csv")

      n_rows=$(( $(wc -l < "$job_csv") - 1 ))
      echo "  ✓ Done in ${JOB_ELAPSED}s — $n_rows rows appended to sweep CSV"
    else
      # ── Failed: write one row with status=FAILED ──────────────────
      FAILED=$((FAILED + 1))

      # Grab last non-empty line from log as error summary (truncate to 200 chars)
      if [[ -f "$job_log" ]]; then
        err_msg=$(grep -v '^$' "$job_log" | tail -1 | head -c 200 | sed 's/,/;/g')
      else
        err_msg="exit code $exit_code"
      fi

      # Row: status,ts,duration,error,experiment,config,model,dataset + 42 empty cols
      # 8 filled + 46 commas = 54 columns total
      empties=$(printf ',%.0s' $(seq 1 46))
      echo "FAILED,${ts},${JOB_ELAPSED},${err_msg},graph_char,,${model},${dataset}${empties}" >> "$MERGED_CSV"

      echo "  ✗ FAILED (exit=$exit_code) after ${JOB_ELAPSED}s — see $job_log"
    fi

    # Cleanup GPU memory between jobs
    echo "  Clearing GPU cache ..."
    $PYTHON -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

  done
done

echo ""

# ─── Final report ─────────────────────────────────────────────────────────────
SWEEP_END=$(date +%s)
SWEEP_ELAPSED=$((SWEEP_END - SWEEP_START))
SWEEP_MIN=$((SWEEP_ELAPSED / 60))
SWEEP_SEC=$((SWEEP_ELAPSED % 60))

TOTAL_ROWS=$(( $(wc -l < "$MERGED_CSV") - 1 ))

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  SWEEP COMPLETE"
echo "  Total time:  ${SWEEP_MIN}m ${SWEEP_SEC}s"
echo "  Completed:   $COMPLETED"
echo "  Failed:      $FAILED"
echo "  Skipped:     $SKIPPED (already done)"
echo "  CSV rows:    $TOTAL_ROWS"
echo "  Results:     $MERGED_CSV"
echo "═══════════════════════════════════════════════════════════════════════"

if [[ $FAILED -gt 0 ]]; then
  echo ""
  echo "  Some jobs failed. Check individual logs in $OUT_DIR/"
  echo "  Re-run with: ./run_profile_sweep.sh --resume"
fi
