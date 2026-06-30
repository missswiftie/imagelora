#!/bin/bash
# 5×A40 多卡 DDP 训练 v2 (增强 Hypernetwork + 风格防塌缩 loss)
set -e
source "$(dirname "$0")/../env.sh"

cd "${IMAGE2LORA_ROOT}/image2lora_scale"

NUM_GPUS="${NUM_GPUS:-5}"

run_python -m accelerate.commands.launch \
    --num_processes "${NUM_GPUS}" \
    --mixed_precision bf16 \
    scripts/train.py \
    --config configs/train_sdxl.yaml \
    --max_train_steps 16000 \
    --output_dir outputs/image2lora_sdxl_v2 \
    "$@"
