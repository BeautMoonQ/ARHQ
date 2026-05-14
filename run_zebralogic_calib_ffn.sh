#!/bin/bash
GPU=${1:-cuda:0}
SAMPLE_RANGE=${2:-0-127}
LAYERS=${3:-0-35}
MAX_LAYER_TOKENS=${4:-30000}
RESULT_DIR=${5:-${HOME}/work/code/TNQ/claude/eval_result}
CALIB_DIR=${6:-${HOME}/work/data/calib_zebralogic}

cd "$(dirname "$0")"
conda run -n llmc --no-capture-output python reference_code/hadazca_calib.py \
    --model_path "${HOME}/work/models/Qwen3-4B-Thinking-2507" \
    --result_dir "$RESULT_DIR" \
    --precision vllm \
    --dataset ZebraLogic \
    --repeat_index 0 \
    --sample_range "$SAMPLE_RANGE" \
    --layers "$LAYERS" \
    --device "$GPU" \
    --max_seq_len 32768 \
    --max_layer_tokens "$MAX_LAYER_TOKENS" \
    --module_set ffn \
    --save_tag ffn \
    --calib_dir "$CALIB_DIR"
