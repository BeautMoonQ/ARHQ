#!/bin/bash
RANK=${1:-128}
DATASET=${2:-MATH-500}
GPU=${3:-cuda:0}

cd "$(dirname "$0")"
conda run -n llmc --no-capture-output python -m arhq.eval_quantized \
    --method arhq --setting raw --rank $RANK --device $GPU \
    --datasets $DATASET --batch_size 4 --max_new_tokens 52768 \
    --temperature 0.6 --top_p 0.95 --top_k 20 --think_mode think
