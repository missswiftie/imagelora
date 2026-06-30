#!/bin/bash
# 用 5 张差异大的 style ref 批量验证风格分化
set -e
source "$(dirname "$0")/../env.sh"

cd "${IMAGE2LORA_ROOT}"

CKPT="${1:-image2lora_scale/outputs/image2lora_sdxl_v2/checkpoint-16000}"
OUT="${2:-${CKPT}/validation}"

run_python image2lora_scale/scripts/validate_styles.py \
    --checkpoint_dir "${CKPT}" \
    --output_dir "${OUT}" \
    "${@:3}"
