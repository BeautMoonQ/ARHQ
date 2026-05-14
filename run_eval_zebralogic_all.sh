#!/bin/bash
GPU=${1:-cuda:0}
BATCH_SIZE=${2:-4}
REPEAT_NUM=${3:-1}

cd "$(dirname "$0")"

echo "[1/3] ZebraLogic: r_only smoothing"
"$(dirname "$0")/run_eval_zebralogic_ronly_smoothing.sh" "$GPU" "$BATCH_SIZE" "$REPEAT_NUM"

echo "[2/3] ZebraLogic: svdquant smoothing"
"$(dirname "$0")/run_eval_zebralogic_svdq_smoothing.sh" "$GPU" "$BATCH_SIZE" "$REPEAT_NUM"

echo "[3/3] ZebraLogic: r_only raw"
"$(dirname "$0")/run_eval_zebralogic_ronly_raw.sh" "$GPU" "$BATCH_SIZE" "$REPEAT_NUM"
