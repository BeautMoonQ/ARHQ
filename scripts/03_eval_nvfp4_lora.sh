#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-${HOME}/work/models/Qwen3-4B-Thinking-2507}
DECOMP_DIR=${DECOMP_DIR:-results/layer_results}
OUTPUT_BASE=${OUTPUT_BASE:-eval_result}
METHOD=${1:-arhq}
SETTING=${2:-smoothing}
RANK=${3:-128}
DATASET=${4:-MATH-500}
GPU=${5:-cuda:0}
MODULE_SET=${6:-all}
MAX_SAMPLES=${MAX_SAMPLES:--1}

cd "$(dirname "$0")/.."
conda run -n llmc --no-capture-output python -m arhq.eval_quantized \
  --model_path "$MODEL_PATH" \
  --decomp_dir "$DECOMP_DIR" \
  --method "$METHOD" \
  --setting "$SETTING" \
  --rank "$RANK" \
  --device "$GPU" \
  --module_set "$MODULE_SET" \
  --datasets "$DATASET" \
  --max_samples "$MAX_SAMPLES" \
  --batch_size 4 \
  --max_new_tokens 52768 \
  --temperature 0.6 \
  --top_p 0.95 \
  --top_k 20 \
  --think_mode think \
  --seed 0 \
  --output_base "$OUTPUT_BASE"
