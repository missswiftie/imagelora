#!/bin/bash
# Ablation full ΔW 多卡 DDP（显存大，建议 4~5 卡）
set -e
source "$(dirname "$0")/../../env.sh"

cd "${IMAGE2LORA_ROOT}"

NUM_GPUS="${NUM_GPUS:-5}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

run_python -m accelerate.commands.launch \
    --num_processes "${NUM_GPUS}" \
    --mixed_precision fp16 \
    ablation/share_aux/scripts/train.py \
    --config configs/train.yaml \
    --output_dir ablation/share_aux/outputs/full_delta_ddp \
    "$@"
