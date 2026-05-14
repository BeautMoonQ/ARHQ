#!/bin/bash
METHOD=${1:-arhq}        # arhq | svdquant | baseline | original | r_only(legacy)
RANK=${2:-128}
GPU=${3:-cuda:2}
SETTING=${4:-smoothing}
MODULE_SET=${5:-attn}
DECOMP_DIR=${DECOMP_DIR:-results/layer_results}

cd "$(dirname "$0")"
conda run -n llmc --no-capture-output python -m arhq.eval_quantized \
    --method $METHOD --setting $SETTING --rank $RANK --device $GPU \
    --module_set "$MODULE_SET" --decomp_dir "$DECOMP_DIR" \
    --question 'Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\theta),$ where $r > 0$ and $0 \le \theta < 2 \pi.$\n\nPlease reason step by step, and put your final answer within \boxed{}.' \
    --max_new_tokens 52768 --temperature 0.6 --top_p 0.95 --top_k 20 \
    --think_mode think --seed 0
