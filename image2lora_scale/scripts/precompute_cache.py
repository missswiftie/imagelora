#!/usr/bin/env python3
"""
预计算训练缓存:
  1. Style DINOv2 特征 (950 张 style 图)
  2. SDXL 双 text encoder embedding (~1446 条 caption)

用法:
  source env.sh
  run_python image2lora_scale/scripts/precompute_cache.py \\
      --manifest dataset/omnistyle_150k.json \\
      --data_root dataset \\
      --output_dir dataset/cache
"""

import argparse
import json
import os
import sys

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

SCALE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(SCALE_ROOT)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCALE_ROOT)

from image2lora.models.encoder import DINOv2Encoder
from image2lora_scale.models.sdxl_utils import encode_sdxl_prompt


def resolve_path(path: str, base: str) -> str:
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base, path))


def collect_unique_styles_and_captions(manifest_path: str, data_root: str):
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    styles = {}
    captions = set()
    for group in manifest.get("samples", []):
        for pair in group.get("pairs", []):
            style_rel = pair.get("style_image") or group.get("style_image")
            caption = pair.get("language_instruction", "")
            if style_rel:
                styles[style_rel] = os.path.join(data_root, style_rel)
            if caption is not None:
                captions.add(caption)
    captions.add("")  # CFG empty prompt
    return styles, sorted(captions)


def build_transform(resolution: int):
    return transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


@torch.no_grad()
def precompute_style_features(
    styles: dict,
    dinov2_model: str,
    dinov2_path: str,
    resolution: int,
    device: torch.device,
) -> dict:
    encoder = DINOv2Encoder(
        model_name=dinov2_model,
        model_path=dinov2_path,
        local_files_only=bool(dinov2_path and os.path.isdir(dinov2_path)),
    ).to(device)
    encoder.eval()
    transform = build_transform(resolution)

    features = {}
    for style_key, abs_path in tqdm(sorted(styles.items()), desc="Style DINOv2"):
        if not os.path.isfile(abs_path):
            print(f"  skip missing: {abs_path}")
            continue
        img = transform(Image.open(abs_path).convert("RGB")).unsqueeze(0).to(device)
        feat = encoder.encode(img).squeeze(0).cpu().to(torch.float16)
        features[style_key] = feat
    return features


@torch.no_grad()
def precompute_text_embeddings(
    captions: list,
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    load_kw = {"local_files_only": True}
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", **load_kw)
    tokenizer_2 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2", **load_kw)
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", **load_kw).to(device, dtype)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        model_path, subfolder="text_encoder_2", **load_kw
    ).to(device, dtype)

    embeddings = {}
    batch_size = 32
    for i in tqdm(range(0, len(captions), batch_size), desc="Text SDXL"):
        batch_caps = captions[i : i + batch_size]
        prompt_embeds, pooled = encode_sdxl_prompt(
            batch_caps, tokenizer, tokenizer_2, text_encoder, text_encoder_2, device,
        )
        for j, cap in enumerate(batch_caps):
            embeddings[cap] = {
                "prompt_embeds": prompt_embeds[j].cpu().to(torch.float16),
                "pooled_prompt_embeds": pooled[j].cpu().to(torch.float16),
            }
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Precompute style/text training caches")
    parser.add_argument("--manifest", type=str, default="dataset/omnistyle_150k.json")
    parser.add_argument("--data_root", type=str, default="dataset")
    parser.add_argument("--output_dir", type=str, default="dataset/cache")
    parser.add_argument("--sdxl_model", type=str, default="pretrained_models/stable-diffusion-xl-base-1.0")
    parser.add_argument("--dinov2_model", type=str, default="dinov2_vitb14")
    parser.add_argument("--dinov2_path", type=str, default="pretrained_models/dinov2-base")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--skip_style", action="store_true")
    parser.add_argument("--skip_text", action="store_true")
    args = parser.parse_args()

    manifest = resolve_path(args.manifest, REPO_ROOT)
    data_root = resolve_path(args.data_root, REPO_ROOT)
    output_dir = resolve_path(args.output_dir, REPO_ROOT)
    sdxl_model = resolve_path(args.sdxl_model, REPO_ROOT)
    dinov2_path = resolve_path(args.dinov2_path, REPO_ROOT)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    styles, captions = collect_unique_styles_and_captions(manifest, data_root)
    print(f"Unique styles: {len(styles)}, unique captions: {len(captions)}")

    meta = {
        "manifest": manifest,
        "resolution": args.resolution,
        "dinov2_model": args.dinov2_model,
        "dinov2_path": dinov2_path,
        "sdxl_model": sdxl_model,
    }

    if not args.skip_style:
        style_features = precompute_style_features(
            styles, args.dinov2_model, dinov2_path, args.resolution, device,
        )
        style_path = os.path.join(output_dir, "style_features.pt")
        torch.save({"meta": meta, "features": style_features}, style_path)
        print(f"Saved {len(style_features)} style features -> {style_path}")

    if not args.skip_text:
        text_embeddings = precompute_text_embeddings(captions, sdxl_model, device, dtype)
        text_path = os.path.join(output_dir, "text_embeddings.pt")
        torch.save({"meta": meta, "embeddings": text_embeddings}, text_path)
        print(f"Saved {len(text_embeddings)} text embeddings -> {text_path}")


if __name__ == "__main__":
    main()
