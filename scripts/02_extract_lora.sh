#!/usr/bin/env bash
set -euo pipefail

CALIB_TENSOR_DIR=${CALIB_TENSOR_DIR:-data/calib_tensor}
OUTPUT_DIR=${OUTPUT_DIR:-results/layer_results}
SUMMARY_DIR=${SUMMARY_DIR:-results/summary}
GPU=${1:-cuda:0}
LAYERS=${2:-0-35}
RANK=${3:-128}
MODULE_SET=${4:-all}
CONFIGS=${5:-arhq:raw,arhq:smoothing,svdquant:smoothing}
TAG=${TAG:-arhq}

cd "$(dirname "$0")/.."
conda run -n llmc --no-capture-output python -m arhq.decompose \
  --calib_dir "$CALIB_TENSOR_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --summary_dir "$SUMMARY_DIR" \
  --device "$GPU" \
  --layers "$LAYERS" \
  --rank "$RANK" \
  --module_set "$MODULE_SET" \
  --configs "$CONFIGS" \
  --tag "$TAG"
