#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-${HOME}/work/models/Qwen3-4B-Thinking-2507}
DECOMP_DIR=${DECOMP_DIR:-results/layer_results}
METHOD=${1:-arhq}
SETTING=${2:-smoothing}
RANK=${3:-128}
GPU=${4:-cuda:0}
MODULE_SET=${5:-all}
QUESTION=${QUESTION:-"What is 2+2? Please put the final answer in \\boxed{}."}

cd "$(dirname "$0")/.."
conda run -n llmc --no-capture-output python -m arhq.eval_quantized \
  --model_path "$MODEL_PATH" \
  --decomp_dir "$DECOMP_DIR" \
  --method "$METHOD" \
  --setting "$SETTING" \
  --rank "$RANK" \
  --device "$GPU" \
  --module_set "$MODULE_SET" \
  --question "$QUESTION" \
  --max_new_tokens 4096 \
  --temperature 0.6 \
  --top_p 0.95 \
  --top_k 20 \
  --think_mode think \
  --seed 0
