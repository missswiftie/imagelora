#!/bin/bash
# Ablation 推理
set -e
source "$(dirname "$0")/../../env.sh"

cd "${IMAGE2LORA_ROOT}"

CKPT="${1:-ablation/share_aux/outputs/full_delta/checkpoint-1000}"
REF="${2:-sampled_100styles_150pairs_package/style/s0000____0912_01_query_2_img_000079_1683294877098_05408690224086452.jpeg.jpg}"
PROMPT="${3:-a beautiful landscape painting in the style of the reference}"
OUT="${4:-ablation/share_aux/outputs/result.png}"

run_python ablation/share_aux/scripts/infer.py \
    --checkpoint_dir "${CKPT}" \
    --ref_image "${REF}" \
    --prompt "${PROMPT}" \
    --output "${OUT}" \
    "${@:5}"
