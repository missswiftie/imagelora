#!/usr/bin/env python3
"""Ablation infer: full rank-1 ΔW from hypernetwork, no aux."""

import argparse
import os
import sys

import torch
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from PIL import Image
from safetensors.torch import load_file
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms

ABLATION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(ABLATION_ROOT))
sys.path.insert(0, REPO_ROOT)

from image2lora.models.encoder import DINOv2Encoder
from ablation.share_aux.models.hypernet_full_delta import FullDeltaHyperDream
from ablation.share_aux.models.lora_full_delta import create_network


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model", type=str, default="pretrained_models/stable-diffusion-v1-5")
    p.add_argument("--checkpoint_dir", type=str, required=True)
    p.add_argument("--ref_image", type=str, required=True)
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--output", type=str, default="ablation/share_aux/outputs/result.png")
    p.add_argument("--negative_prompt", type=str, default="low quality, blurry, distorted")
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rank", type=int, default=1)
    p.add_argument("--decoder_blocks", type=int, default=8)
    p.add_argument("--sample_iters", type=int, default=4)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--dinov2_model_path", type=str, default="pretrained_models/dinov2-base")
    return p.parse_args()


def resolve(path):
    if os.path.isabs(path):
        return path
    for root in (REPO_ROOT, ABLATION_ROOT):
        c = os.path.join(root, path)
        if os.path.exists(c):
            return c
    return os.path.join(REPO_ROOT, path)


def update_lora_weights(network, weight_list):
    for weight, lora_layer in zip(weight_list, network.unet_loras):
        if weight.dim() == 3:
            weight = weight.view(weight.size(0), -1)
        elif weight.dim() == 2 and weight.size(0) == 1:
            weight = weight.view(-1)
        lora_layer.update_weight(weight)


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    model_path = resolve(args.pretrained_model)
    load_kw = {"local_files_only": True} if os.path.isdir(model_path) else {}

    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", **load_kw)
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=dtype, **load_kw)
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype, **load_kw)
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=dtype, **load_kw)
    scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler", **load_kw)

    network = create_network(
        1.0, args.rank, 1.0, text_encoder, unet,
        is_train=True, train_unet=True, train_text_encoder=False,
    )
    network.apply_to(text_encoder, unet, apply_text_encoder=False, apply_unet=True)

    hypernetwork = FullDeltaHyperDream(
        weight_dims=network.weight_dims,
        decoder_blocks=args.decoder_blocks,
        sample_iters=args.sample_iters,
    )
    hypernetwork.load_state_dict(load_file(os.path.join(args.checkpoint_dir, "hypernetwork.safetensors")))
    hypernetwork.eval()

    dino_path = resolve(args.dinov2_model_path)
    image_encoder = DINOv2Encoder(model_path=dino_path, local_files_only=os.path.isdir(dino_path))
    image_encoder.eval().to(device)

    network.to(device, dtype=dtype)
    hypernetwork.to(device, dtype=dtype)
    text_encoder.to(device)
    vae.to(device)
    unet.to(device)

    ref_transform = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    ref_tensor = ref_transform(Image.open(resolve(args.ref_image)).convert("RGB")).unsqueeze(0).to(device)
    _, weight_list = hypernetwork(image_encoder.encode(ref_tensor).to(dtype=dtype))
    update_lora_weights(network, weight_list)

    pipe = StableDiffusionPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        unet=unet, scheduler=scheduler, safety_checker=None, feature_extractor=None,
    ).to(device)

    out_path = resolve(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.resolution,
        width=args.resolution,
        generator=torch.Generator(device=device).manual_seed(args.seed),
    ).images[0]
    img.save(out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
