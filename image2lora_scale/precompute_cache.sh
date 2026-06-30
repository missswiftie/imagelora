#!/bin/bash
# 预计算 Style DINOv2 特征 + SDXL text embedding 缓存
set -e
source "$(dirname "$0")/../env.sh"

cd "${IMAGE2LORA_ROOT}"

run_python image2lora_scale/scripts/precompute_cache.py \
    --manifest dataset/omnistyle_150k.json \
    --data_root dataset \
    --output_dir dataset/cache \
    "$@"

echo "Done. Cache saved to dataset/cache/"
