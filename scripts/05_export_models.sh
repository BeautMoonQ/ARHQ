#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-${HOME}/work/models/Qwen3-4B-Thinking-2507}
DECOMP_DIR=${DECOMP_DIR:-results/layer_results}
OUTPUT_DIR=${OUTPUT_DIR:-models}
RANK=${1:-128}
MODULE_SET=${2:-all}
LAYERS=${3:-0-35}
EXPORTS=${EXPORTS:-svdquant:smoothing:svdquant_smoothing_rank128_fp16:fp16,svdquant:smoothing:svdquant_smoothing_rank128_packed4bit:packed4bit,arhq:raw:arhq_raw_rank128_fp16:fp16,arhq:raw:arhq_raw_rank128_packed4bit:packed4bit,arhq:smoothing:arhq_smoothing_rank128_fp16:fp16,arhq:smoothing:arhq_smoothing_rank128_packed4bit:packed4bit}

cd "$(dirname "$0")/.."
conda run -n llmc --no-capture-output python -m arhq.export_model \
  --model_path "$MODEL_PATH" \
  --decomp_dir "$DECOMP_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --rank "$RANK" \
  --module_set "$MODULE_SET" \
  --layers "$LAYERS" \
  --exports "$EXPORTS"
