#!/usr/bin/env bash
# Run GraphRAG experiments: 4 settings × N models, store results per model.
# Usage: ./run_experiments.sh
# Requires: conda with env "graphrag", run from graphrag_kv or parent.

set -euo pipefail

# ─── Paths ─────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
OUT_DIR="$SCRIPT_DIR/experiment_results"
FAILED_RUNS_LOG="$OUT_DIR/failed_runs.txt"
mkdir -p "$OUT_DIR"
# Clear previous failure log so we only report this run's failures
: > "$FAILED_RUNS_LOG"
TMP_LOG=$(mktemp)
trap "rm -f $TMP_LOG" EXIT

# ─── Conda ─────────────────────────────────────────────────────────────────
if [[ -z "${CONDA_DEFAULT_ENV:-}" ]] || [[ "$CONDA_DEFAULT_ENV" != "graphrag" ]]; then
  if command -v conda &>/dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate graphrag
  else
    echo "Error: conda not found. Activate env 'graphrag' and re-run." >&2
    exit 1
  fi
fi
echo "Using conda env: ${CONDA_DEFAULT_ENV:-none}"

# ─── Base args (same for all runs) ──────────────────────────────────────────
DATASET="metaqa-3hop"
FAST_MODE="neurips"
# 80B model is too large for data_parallel; use model_parallel (layers spread across GPUs)
MODEL_PARALLEL_MODEL="Qwen/Qwen3-Next-80B-A3B-Thinking"
# Optional: for gated models (e.g. Llama), set HF_TOKEN and we pass --use_auth_token
[[ -n "${HF_TOKEN:-}" ]] && AUTH_ARGS="--use_auth_token $HF_TOKEN" || AUTH_ARGS=""

# ─── Models ─────────────────────────────────────────────────────────────────
MODELS=(
  "meta-llama/Llama-2-7b-hf"
  "meta-llama/Llama-2-13b-hf"
  "Qwen/Qwen3-8B"
  "facebook/opt-6.7b"
  "facebook/opt-13b"
  "facebook/opt-30b"
  "google/gemma-2-2b"
  "google/gemma-2-9b"
  "google/gemma-2-27b"
  "Qwen/Qwen3-Next-80B-A3B-Thinking"
  "Qwen/Qwen3-32B"
)

# ─── Settings: name and extra args ──────────────────────────────────────────
# Setting 1: original (no retrieve)
# Setting 2: + retrieve
# Setting 3: + retrieve + path_enum
# Setting 4: + retrieve + path_enum + use_pfkv + batch_opt
SETTING_NAMES=(
  "1_original"
  "2_retrieve"
  "3_retrieve_path_enum"
  "4_retrieve_path_enum_pfkv_batch_opt"
)
SETTING_ARGS=(
  ""
  "--retrieve"
  "--retrieve --path_enum"
  "--retrieve --path_enum --use_pfkv --batch_opt"
)

# ─── Run one experiment and append parsed results to a file ─────────────────
run_one() {
  local model="$1"
  local setting_name="$2"
  local extra_args="$3"
  local out_file="$4"
  if [[ "$model" == "$MODEL_PARALLEL_MODEL" ]]; then
    parallel_args="--parallel_mode model_parallel --num_gpus 4"
  else
    parallel_args="--parallel_mode data_parallel --num_gpus 4"
  fi
  local full_cmd="python main.py --dataset $DATASET --fast_mode $FAST_MODE $parallel_args $AUTH_ARGS --llm_model $model $extra_args"

  echo "" >> "$out_file"
  echo "###########################################################################" >> "$out_file"
  echo "# Setting: $setting_name" >> "$out_file"
  echo "# Command: $full_cmd" >> "$out_file"
  echo "###########################################################################" >> "$out_file"

  eval "$full_cmd" 2>&1 | tee "$TMP_LOG" | tail -n 60
  py_exit=${PIPESTATUS[0]}
  if [[ "$py_exit" -eq 0 ]]; then
    # Extract from "RESULTS —" through "Done — results..." and optional PFKV/Subgraph line
    awk '
      /^RESULTS —/ { capture = 1 }
      capture { print }
      /^Done — results are reproducible/ {
        print
        capture = 0
        if ((getline line) > 0 && (line ~ /Subgraph KV Cache|Legacy PathFirstKVCache/)) print line
      }
    ' "$TMP_LOG" >> "$out_file" 2>/dev/null || true
  else
    echo "[RUN FAILED — exit code: $py_exit — see log above]" >> "$out_file"
    echo "model=$model | setting=$setting_name | exit=$py_exit" >> "$FAILED_RUNS_LOG"
  fi
  echo "" >> "$out_file"
  return 0
}

# ─── Model name → safe filename ─────────────────────────────────────────────
model_to_slug() {
  echo "$1" | sed 's|/|_|g'
}

# ─── Main loop (disable set -e so one failed run does not stop the script) ─────
set +e
for model in "${MODELS[@]}"; do
  slug=$(model_to_slug "$model")
  out_file="$OUT_DIR/results_${slug}.txt"
  echo "========================================"
  echo "Model: $model"
  echo "Output: $out_file"
  echo "========================================"

  if [[ "$model" == "$MODEL_PARALLEL_MODEL" ]]; then
    parallel_label="model_parallel"
  else
    parallel_label="data_parallel"
  fi
  {
    echo "###########################################################################"
    echo "# GraphRAG experiment results — model: $model"
    echo "# Dataset: $DATASET  |  Fast mode: $FAST_MODE  |  Parallel: $parallel_label, 4 GPUs"
    echo "# Generated: $(date -Iseconds 2>/dev/null || date)"
    echo "###########################################################################"
  } > "$out_file"

  for i in "${!SETTING_NAMES[@]}"; do
    echo "--- Running ${SETTING_NAMES[$i]} ---"
    run_one "$model" "${SETTING_NAMES[$i]}" "${SETTING_ARGS[$i]}" "$out_file" || true
  done

  echo "Done with model: $model → $out_file"
done
set -e

echo ""
echo "All experiments finished. Results in: $OUT_DIR"
if [[ -s "$FAILED_RUNS_LOG" ]]; then
  echo ""
  echo "Failed runs (see result files for details):"
  cat "$FAILED_RUNS_LOG"
  echo ""
  echo "Re-run specific commands from the '# Command:' lines in the result files."
fi
