#!/bin/bash
GPU=${1:-cuda:0}
BATCH_SIZE=${2:-4}
REPEAT_NUM=${3:-1}

cd "$(dirname "$0")"
conda run -n llmc --no-capture-output python -m arhq.eval_quantized \
    --method r_only --setting raw --rank 128 --device "$GPU" \
    --datasets ZebraLogic --batch_size "$BATCH_SIZE" --repeat_num "$REPEAT_NUM" \
    --max_new_tokens 52768 --temperature 0.6 --top_p 0.95 --top_k 20 \
    --think_mode think
