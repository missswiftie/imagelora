#!/bin/bash
# Image2LoRA SDXL scale-up 单卡训练 (建议用 train_ddp.sh 多卡 + precompute_cache.sh)
set -e
source "$(dirname "$0")/../env.sh"

cd "${IMAGE2LORA_ROOT}/image2lora_scale"

run_python scripts/train.py \
    --config configs/train_sdxl.yaml \
    "$@"

