#!/bin/bash
GPU=${1:-cuda:1}
RANK=${2:-128}
LAYERS=${3:-0-35}
SAMPLES_DIR=${4:-${HOME}/work/data/calib_zebralogic/vllm_ZebraLogic_ffn/samples_0000}
TAG=${5:-zebralogic}
EVAL_TOKENS=${6:-2048}
PROJS=${7:-gate_proj,up_proj,down_proj}
SKIP_EXISTING=${8:-}

EXTRA_ARGS=()
if [ -n "$SKIP_EXISTING" ]; then
    EXTRA_ARGS+=(--skip_existing)
fi

cd "$(dirname "$0")"
conda run -n llmc --no-capture-output python -m archive.arhq_legacy.sweep_three_ronly_ffn \
    --samples_dir "$SAMPLES_DIR" \
    --device "$GPU" \
    --layers "$LAYERS" \
    --rank "$RANK" \
    --tag "$TAG" \
    --eval_tokens "$EVAL_TOKENS" \
    --projs "$PROJS" \
    "${EXTRA_ARGS[@]}"
