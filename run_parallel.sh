#!/bin/bash
# 4-GPU parallel dataset evaluation
# Usage: ./run_parallel.sh [method] [rank] [dataset] [batch_size] [setting]
# Example: ./run_parallel.sh r_only 128 MATH-500 4 raw

METHOD=${1:-r_only}
RANK=${2:-128}
DATASET=${3:-MATH-500}
BATCH_SIZE=${4:-4}
SETTING=${5:-raw}
WORLD_SIZE=4

cd "$(dirname "$0")"

echo "========================================"
echo "4-GPU Parallel Eval: $METHOD setting=$SETTING rank=$RANK $DATASET bs=$BATCH_SIZE"
echo "========================================"

for GPU_ID in 0 1 2 3; do
    LOG="eval_${METHOD}_${SETTING}_rank${RANK}_${DATASET}_gpu${GPU_ID}.log"
    nohup conda run -n llmc --no-capture-output python -m arhq.eval_quantized \
        --method $METHOD --setting $SETTING --rank $RANK --device cuda:$GPU_ID \
        --datasets $DATASET --batch_size $BATCH_SIZE \
        --max_new_tokens 52768 --temperature 0.6 --top_p 0.95 --top_k 20 \
        --think_mode think \
        --rank_id $GPU_ID --world_size $WORLD_SIZE \
        > "$LOG" 2>&1 &
    echo "GPU $GPU_ID: PID=$!, log=$LOG"
done

echo ""
echo "Monitor progress:"
echo "  watch -n 10 'ls eval_result/arhq_${METHOD}_${SETTING}_rank${RANK}/${DATASET}/ 2>/dev/null | wc -l'"
echo ""
echo "Check logs:"
echo "  tail -f eval_${METHOD}_${SETTING}_rank${RANK}_${DATASET}_gpu*.log"
