#!/usr/bin/env python3
"""Image2LoRA SDXL 推理：参考图 + 文本 → 风格化生成。"""

import argparse
import os
import sys

import torch
from diffusers import AutoencoderKL, EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from PIL import Image
from safetensors.torch import load_file
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

SCALE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(SCALE_ROOT)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCALE_ROOT)

from image2lora.models.encoder import DINOv2Encoder
from image2lora_scale.models.hypernet import ImageHyperDream
from image2lora_scale.models.lora import create_network


def parse_args():
    p = argparse.ArgumentParser(description="Image2LoRA SDXL inference")
    p.add_argument("--pretrained_model", type=str, default="pretrained_models/stable-diffusion-xl-base-1.0")
    p.add_argument("--checkpoint_dir", type=str, required=True)
    p.add_argument("--ref_image", type=str, required=True)
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--negative_prompt", type=str, default="low quality, blurry, distorted, ugly")
    p.add_argument("--output", type=str, default="output_sdxl.png")
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=7.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rank", type=int, default=2)
    p.add_argument("--network_alpha", type=float, default=2.0)
    p.add_argument("--down_dim", type=int, default=128)
    p.add_argument("--up_dim", type=int, default=64)
    p.add_argument("--decoder_blocks", type=int, default=20)
    p.add_argument("--sample_iters", type=int, default=5)
    p.add_argument("--style_embed_dim", type=int, default=256)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--lora_scale", type=float, default=1.0)
    p.add_argument("--dinov2_model", type=str, default="dinov2_vitb14")
    p.add_argument("--dinov2_model_path", type=str, default="pretrained_models/dinov2-base")
    return p.parse_args()


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    for root in (REPO_ROOT, SCALE_ROOT):
        candidate = os.path.join(root, path)
        if os.path.exists(candidate):
            return candidate
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

    model_path = resolve_path(args.pretrained_model)
    load_kw = {"local_files_only": True} if os.path.isdir(model_path) else {}

    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", **load_kw)
    tokenizer_2 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2", **load_kw)
    text_encoder = CLIPTextModel.from_pretrained(
        model_path, subfolder="text_encoder", torch_dtype=dtype, **load_kw
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        model_path, subfolder="text_encoder_2", torch_dtype=dtype, **load_kw
    )
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype, **load_kw)
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=dtype, **load_kw)
    scheduler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler", **load_kw)

    for m in (text_encoder, text_encoder_2, vae, unet):
        m.requires_grad_(False)

    text_encoders = [text_encoder, text_encoder_2]
    network = create_network(
        args.lora_scale, args.rank, args.network_alpha,
        text_encoders, unet,
        down_dim=args.down_dim, up_dim=args.up_dim,
        is_train=True, train_unet=True, train_text_encoder=False,
    )
    lora_path = os.path.join(args.checkpoint_dir, "lora_aux.safetensors")
    network.load_state_dict(load_file(lora_path), strict=False)
    network.apply_to(text_encoders, unet, apply_text_encoder=False, apply_unet=True)
    print(f"Loaded LightLoRA aux from {lora_path}")

    lora_weight_dim = (args.down_dim + args.up_dim) * args.rank
    hypernetwork = ImageHyperDream(
        image_feat_dim=768,
        weight_dim=lora_weight_dim,
        weight_num=len(network.unet_loras),
        decoder_blocks=args.decoder_blocks,
        sample_iters=args.sample_iters,
        style_embed_dim=args.style_embed_dim,
    )
    hyper_path = os.path.join(args.checkpoint_dir, "hypernetwork.safetensors")
    hypernetwork.load_state_dict(load_file(hyper_path))
    hypernetwork.eval()
    print(f"Loaded hypernetwork from {hyper_path}")

    dinov2_path = resolve_path(args.dinov2_model_path)
    image_encoder = DINOv2Encoder(
        model_name=args.dinov2_model,
        model_path=dinov2_path,
        local_files_only=os.path.isdir(dinov2_path),
    )
    image_encoder.eval()
    image_encoder.to(device)

    network.to(device, dtype=dtype)
    hypernetwork.to(device, dtype=dtype)
    text_encoder.to(device)
    text_encoder_2.to(device)
    vae.to(device)
    unet.to(device)

    from torchvision import transforms

    ref_transform = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    ref_tensor = ref_transform(Image.open(args.ref_image).convert("RGB")).unsqueeze(0).to(device)
    ref_features = image_encoder.encode(ref_tensor).to(dtype=dtype)
    _, weight_list, _ = hypernetwork(ref_features)
    update_lora_weights(network, weight_list)
    print(f"Generated LightLoRA weights from reference ({len(weight_list)} layers)")

    pipe = StableDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        unet=unet,
        scheduler=scheduler,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.resolution,
        width=args.resolution,
        generator=generator,
    ).images[0]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    result.save(args.output)
    print(f"Saved result to {args.output}")


if __name__ == "__main__":
    main()
