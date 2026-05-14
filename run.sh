#!/bin/bash
METHOD=${1:-arhq}       # arhq | svdquant | baseline | original | r_only(legacy)
RANK=${2:-128}
DATASET=${3:-MATH-500}
GPU=${4:-cuda:3}
SETTING=${5:-smoothing} # raw | smoothing
MODULE_SET=${6:-attn}   # attn | ffn | all
DECOMP_DIR=${DECOMP_DIR:-results/layer_results}

cd "$(dirname "$0")"
conda run -n llmc --no-capture-output python -m arhq.eval_quantized \
    --method $METHOD --setting $SETTING --rank $RANK --device $GPU \
    --decomp_dir "$DECOMP_DIR" \
    --module_set $MODULE_SET \
    --datasets $DATASET --batch_size 4 --max_new_tokens 52768 \
    --temperature 0.6 --top_p 0.95 --top_k 20 --think_mode think
