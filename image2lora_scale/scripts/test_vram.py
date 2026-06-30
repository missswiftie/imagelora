#!/usr/bin/env python3
"""测试 SDXL + LightLoRA + Hypernetwork 在 A40 上的显存占用。"""

import argparse
import gc
import os
import sys

import torch
import torch.nn.functional as F

SCALE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(SCALE_ROOT)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCALE_ROOT)

from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from image2lora.models.encoder import DINOv2Encoder
from image2lora_scale.models.hypernet import ImageHyperDream
from image2lora_scale.models.lora import create_network
from image2lora_scale.models.sdxl_utils import encode_sdxl_prompt, get_add_time_ids


def mem_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024 ** 3


def run_test(model_path, dino_path, resolution, batch_size, rank, down_dim, up_dim,
             decoder_blocks, sample_iters, device):
    dtype = torch.float16
    lk = {"local_files_only": True}

    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()

    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", **lk)
    tokenizer_2 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2", **lk)
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", **lk).to(device, dtype)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        model_path, subfolder="text_encoder_2", **lk
    ).to(device, dtype)
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", **lk).to(device, dtype)
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", **lk).to(device, dtype)
    unet.enable_gradient_checkpointing()
    print(f"[1] SDXL loaded: {mem_gb():.2f} GB")

    network = create_network(
        1.0, rank, float(rank), [text_encoder, text_encoder_2], unet,
        down_dim=down_dim, up_dim=up_dim, is_train=True, train_unet=True, train_text_encoder=False,
    )
    network.apply_to([text_encoder, text_encoder_2], unet, apply_text_encoder=False, apply_unet=True)
    print(f"[2] LightLoRA ({len(network.unet_loras)} layers): {mem_gb():.2f} GB")

    wd = (down_dim + up_dim) * rank
    dino = DINOv2Encoder(model_path=dino_path, local_files_only=True).to(device)
    hyper = ImageHyperDream(
        image_feat_dim=dino.feat_dim,
        weight_dim=wd,
        weight_num=len(network.unet_loras),
        decoder_blocks=decoder_blocks,
        sample_iters=sample_iters,
        style_embed_dim=256,
    )
    hyper.set_lilora(network.unet_loras)
    hyper.enable_gradient_checkpointing()
    hyper.to(device)
    print(f"[3] + DINOv2 + Hypernet: {mem_gb():.2f} GB")

    ref = torch.randn(batch_size, 3, resolution, resolution, device=device)
    tgt = torch.randn(batch_size, 3, resolution, resolution, device=device, dtype=dtype)

    with torch.no_grad():
        feat = dino.encode(ref)
    _, weight_list, _ = hyper(feat.float())
    for w, layer in zip(weight_list, network.unet_loras):
        w = w.view(-1) if w.dim() > 1 and w.size(0) == 1 else w
        if w.dim() == 2:
            w = w.view(w.size(0), -1)
        layer.update_weight(w if w.dim() > 1 else w.view(-1))

    with torch.no_grad():
        latents = vae.encode(tgt).latent_dist.sample() * vae.config.scaling_factor

    prompt_embeds, pooled = encode_sdxl_prompt(
        ["test prompt"] * batch_size,
        tokenizer, tokenizer_2, text_encoder, text_encoder_2, device,
    )
    noise = torch.randn_like(latents)
    ts = torch.randint(0, 1000, (batch_size,), device=device)
    noisy = noise_scheduler_add(noise, latents, ts)
    add_time_ids = get_add_time_ids(resolution, batch_size, dtype, device)

    pred = unet(
        noisy, ts,
        encoder_hidden_states=prompt_embeds,
        added_cond_kwargs={"text_embeds": pooled, "time_ids": add_time_ids},
    ).sample
    loss = F.mse_loss(pred.float(), noise.float())
    loss.backward()
    print(f"[4] Full train step ({resolution}px, batch={batch_size}): {mem_gb():.2f} GB peak")
    return mem_gb()


def noise_scheduler_add(noise, latents, timesteps):
    return latents + 0.01 * noise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available, skip VRAM test.")
        return

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, {props.total_memory / 1024**3:.1f} GB total")

    model_path = os.path.join(REPO_ROOT, "pretrained_models/stable-diffusion-xl-base-1.0")
    dino_path = os.path.join(REPO_ROOT, "pretrained_models/dinov2-base")

    peak = run_test(
        model_path, dino_path,
        resolution=args.resolution,
        batch_size=args.batch_size,
        rank=2, down_dim=128, up_dim=64,
        decoder_blocks=20, sample_iters=5,
        device=device,
    )

    total_gb = props.total_memory / 1024 ** 3
    if peak < total_gb * 0.85:
        print(f"\nOK: peak {peak:.1f} GB fits in {total_gb:.0f} GB A40 — single-GPU training feasible.")
    else:
        print(f"\nTIGHT: peak {peak:.1f} GB near {total_gb:.0f} GB limit — use batch=1 or multi-GPU DDP.")


if __name__ == "__main__":
    main()
