#!/bin/bash
# Image2LoRA 环境变量与 Python 包装器（兼容旧版 glibc 系统）
export IMAGE2LORA_ENV="image2lora"
export IMAGE2LORA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export IMAGE2LORA_PYTHON="/scratch/jiaqi/anaconda3/envs/${IMAGE2LORA_ENV}/bin/python"
export IMAGE2LORA_PYTHON_WRAPPER="${IMAGE2LORA_ROOT}/bin/python"
# torch.distributed.run / accelerate 多卡 spawn 子进程时使用
export PYTHON_EXEC="${IMAGE2LORA_PYTHON_WRAPPER}"

# HuggingFace 镜像（预下载模型时生效，可按需取消注释）
# export HF_ENDPOINT="https://hf-mirror.com"

run_python() {
    "${IMAGE2LORA_PYTHON_WRAPPER}" "$@"
}

export -f run_python
