#!/bin/bash
# Ablation: full ΔW LoRA 单卡训练（显存不足请用 train_ddp.sh）
set -e
source "$(dirname "$0")/../../env.sh"

cd "${IMAGE2LORA_ROOT}"

run_python ablation/share_aux/scripts/train.py \
    --config configs/train.yaml \
    "$@"
