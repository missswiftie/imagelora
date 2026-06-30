#!/bin/bash
# Image2LoRA SDXL scale-up 推理
set -e
source "$(dirname "$0")/../env.sh"

cd "${IMAGE2LORA_ROOT}"

CKPT="${1:-image2lora_scale/outputs/image2lora_sdxl_ddp/checkpoint-8000}"
REF="${2:-dataset/style/s0000____0912_01_query_2_img_000079_1683294877098_05408690224086452.jpeg.jpg}"
PROMPT="${3:-a beautiful landscape painting, highly detailed, masterpiece}"
OUT="${4:-image2lora_scale/outputs/infer_result.png}"

run_python image2lora_scale/scripts/infer.py \
    --checkpoint_dir "${CKPT}" \
    --ref_image "${REF}" \
    --prompt "${PROMPT}" \
    --output "${OUT}" \
    --resolution 1024 \
    --num_inference_steps 30 \
    --guidance_scale 7.0 \
    --seed 42 \
    "${@:5}"
