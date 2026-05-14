#!/bin/bash
GPU=${1:-cuda:0}
LAYERS=${2:-0-35}
RANK=${3:-128}
SAMPLES_DIR=${4:-${HOME}/work/data/calib_zebralogic/vllm_ZebraLogic/samples_0000}
TAG=${5:-zebralogic}

cd "$(dirname "$0")"
conda run -n llmc --no-capture-output python -m archive.arhq_legacy.sweep_three_ronly \
    --samples_dir "$SAMPLES_DIR" \
    --device "$GPU" \
    --layers "$LAYERS" \
    --rank "$RANK" \
    --tag "$TAG"
