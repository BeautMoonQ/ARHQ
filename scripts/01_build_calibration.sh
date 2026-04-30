#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-${HOME}/work/models/Qwen3-4B-Thinking-2507}
CALIB_DATA_DIR=${CALIB_DATA_DIR:-data/calib_data}
CALIB_TENSOR_DIR=${CALIB_TENSOR_DIR:-data/calib_tensor}
GPU=${1:-cuda:0}
LAYERS=${2:-0-35}
MODULE_SET=${3:-all}
SAMPLE_RANGE=${4:-0-127}
MAX_LAYER_TOKENS=${5:-30000}

cd "$(dirname "$0")/.."
conda run -n llmc --no-capture-output python -m arhq.calibration \
  --model_path "$MODEL_PATH" \
  --result_dir "$CALIB_DATA_DIR" \
  --output_dir "$CALIB_TENSOR_DIR" \
  --repeat_index 0 \
  --sample_range "$SAMPLE_RANGE" \
  --layers "$LAYERS" \
  --module_set "$MODULE_SET" \
  --device "$GPU" \
  --max_seq_len 32768 \
  --max_layer_tokens "$MAX_LAYER_TOKENS"
