#!/usr/bin/env python3
"""Image2LoRA SDXL 训练脚本。"""

import argparse
import logging
import math
import os
import sys

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

SCALE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(SCALE_ROOT)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCALE_ROOT)

from image2lora.models.encoder import DINOv2Encoder
from image2lora_scale.data.cache import TrainingCache
from image2lora_scale.data.dataset import OmniStyleDataset, collate_fn
from image2lora_scale.models.hypernet import ImageHyperDream
from image2lora_scale.models.lora import create_network
from image2lora_scale.models.losses import (
    style_separation_loss,
    style_supervised_contrastive_loss,
    weight_diversity_loss,
)
from image2lora_scale.models.sdxl_utils import encode_sdxl_prompt, get_add_time_ids

check_min_version("0.27.0")
logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Image2LoRA SDXL training")
    parser.add_argument("--config", type=str, default="configs/train_sdxl.yaml")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None)
    parser.add_argument("--train_data_json", type=str, default=None)
    parser.add_argument("--train_data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--require_files_exist", type=str, default=None,
                        help="true/false, 数据未解压完可设 false")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    return parser.parse_args()


def resolve_path(path: str, base: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base, path))


def load_config(args):
    cfg = OmegaConf.load(resolve_path(args.config, SCALE_ROOT))
    overrides = [
        "pretrained_model_name_or_path", "train_data_json", "train_data_dir",
        "output_dir", "train_batch_size", "learning_rate", "max_train_steps",
        "resume_from_checkpoint",
    ]
    for key in overrides:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    if args.require_files_exist is not None:
        cfg.require_files_exist = args.require_files_exist.lower() in ("1", "true", "yes")

    model_path = resolve_path(cfg.pretrained_model_name_or_path, REPO_ROOT)
    if os.path.isdir(model_path):
        cfg.pretrained_model_name_or_path = model_path
        cfg.local_files_only = True
    else:
        cfg.local_files_only = False

    cfg.train_data_json = resolve_path(cfg.train_data_json, REPO_ROOT)
    cfg.train_data_dir = resolve_path(cfg.train_data_dir, REPO_ROOT)
    if not os.path.isabs(cfg.output_dir):
        cfg.output_dir = resolve_path(cfg.output_dir, SCALE_ROOT)

    dinov2_path = cfg.get("dinov2_model_path")
    if dinov2_path:
        dinov2_path = resolve_path(dinov2_path, REPO_ROOT)
        cfg.dinov2_model_path = dinov2_path

    for key in ("style_feature_cache", "text_embedding_cache"):
        if cfg.get(key):
            cfg[key] = resolve_path(cfg[key], REPO_ROOT)
    return cfg


def save_checkpoint(output_dir, network, hypernetwork, global_step, weight_dtype):
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    network.save_weights(os.path.join(ckpt_dir, "lora_aux.safetensors"), weight_dtype)
    from safetensors.torch import save_file
    hyper_sd = {k: v.detach().cpu().to(weight_dtype) for k, v in hypernetwork.state_dict().items()}
    save_file(hyper_sd, os.path.join(ckpt_dir, "hypernetwork.safetensors"))
    logger.info(f"Saved checkpoint to {ckpt_dir}")


def resolve_resume_checkpoint(resume_path: str, output_dir: str) -> str:
    if resume_path in (None, "", "latest"):
        ckpts = sorted(
            [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[-1]),
        )
        if not ckpts:
            raise ValueError(f"No checkpoint found in {output_dir}")
        return os.path.join(output_dir, ckpts[-1])
    if os.path.isdir(resume_path):
        return resume_path
    if os.path.isdir(os.path.join(output_dir, resume_path)):
        return os.path.join(output_dir, resume_path)
    raise ValueError(f"Checkpoint not found: {resume_path}")


def load_checkpoint(ckpt_dir, network, hypernetwork):
    from safetensors.torch import load_file
    network.load_state_dict(load_file(os.path.join(ckpt_dir, "lora_aux.safetensors")), strict=False)
    hypernetwork.load_state_dict(load_file(os.path.join(ckpt_dir, "hypernetwork.safetensors")))
    global_step = int(os.path.basename(ckpt_dir).split("-")[-1])
    logger.info(f"Resumed weights from {ckpt_dir} (step={global_step})")
    return global_step


def update_lora_weights(network, weight_list):
    actual = network.module if hasattr(network, "module") else network
    for weight, lora_layer in zip(weight_list, actual.unet_loras):
        if weight.dim() == 3:
            weight = weight.view(weight.size(0), -1)
        elif weight.dim() == 2 and weight.size(0) == 1:
            weight = weight.view(-1)
        lora_layer.update_weight(weight)


def main():
    args = parse_args()
    cfg = load_config(args)

    if not os.path.isfile(cfg.train_data_json):
        raise FileNotFoundError(
            f"Manifest not found: {cfg.train_data_json}\n"
            "Run: run_python scripts/build_manifest.py --data_root ../dataset "
            "--output ../dataset/omnistyle_manifest.json"
        )

    logging_dir = os.path.join(cfg.output_dir, "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=cfg.report_to,
        project_config=ProjectConfiguration(project_dir=cfg.output_dir, logging_dir=logging_dir),
    )
    set_seed(cfg.seed)

    if accelerator.is_main_process:
        os.makedirs(cfg.output_dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(cfg.output_dir, "config.yaml"))

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    model_path = cfg.pretrained_model_name_or_path
    load_kwargs = {"local_files_only": cfg.get("local_files_only", False)}
    if accelerator.is_main_process:
        logger.info(f"Loading SDXL from {model_path}")

    use_style_cache = bool(cfg.get("style_feature_cache"))
    use_text_cache = bool(cfg.get("text_embedding_cache"))

    tokenizer = None
    tokenizer_2 = None
    if not use_text_cache:
        tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", **load_kwargs)
        tokenizer_2 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2", **load_kwargs)

    training_cache = TrainingCache(
        style_cache_path=cfg.get("style_feature_cache") if use_style_cache else None,
        text_cache_path=cfg.get("text_embedding_cache") if use_text_cache else None,
    )
    if use_style_cache and not training_cache.has_style_cache:
        raise FileNotFoundError(f"Style cache not found: {cfg.style_feature_cache}")
    if use_text_cache and not training_cache.has_text_cache:
        raise FileNotFoundError(f"Text cache not found: {cfg.text_embedding_cache}")

    text_encoder = None
    text_encoder_2 = None
    if not use_text_cache:
        text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", **load_kwargs)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            model_path, subfolder="text_encoder_2", **load_kwargs
        )
        text_encoder.requires_grad_(False)
        text_encoder_2.requires_grad_(False)

    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", **load_kwargs)
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", **load_kwargs)
    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler", **load_kwargs)
    unet.requires_grad_(False)

    text_encoders_for_lora = [text_encoder, text_encoder_2] if text_encoder is not None else []
    network = create_network(
        1.0, cfg.rank, cfg.network_alpha,
        text_encoders_for_lora, unet,
        down_dim=cfg.down_dim, up_dim=cfg.up_dim,
        is_train=True, train_unet=True, train_text_encoder=False,
    )
    network.apply_to(text_encoders_for_lora, unet, apply_text_encoder=False, apply_unet=True)

    image_encoder = None
    dinov2_path = cfg.get("dinov2_model_path")
    if not use_style_cache:
        image_encoder = DINOv2Encoder(
            model_name=cfg.dinov2_model,
            model_path=dinov2_path,
            local_files_only=bool(dinov2_path and os.path.isdir(dinov2_path)),
        )
        image_encoder.eval()
        feat_dim = image_encoder.feat_dim
    else:
        sample_feat = next(iter(training_cache.style_features.values()))
        feat_dim = sample_feat.shape[-1]
    lora_weight_dim = (cfg.down_dim + cfg.up_dim) * cfg.rank
    hypernetwork = ImageHyperDream(
        image_feat_dim=feat_dim,
        weight_dim=lora_weight_dim,
        weight_num=len(network.unet_loras),
        decoder_blocks=cfg.decoder_blocks,
        sample_iters=cfg.sample_iters,
        style_embed_dim=cfg.get("style_embed_dim", 256),
    )
    hypernetwork.set_lilora(network.unet_loras)
    hypernetwork.set_device(accelerator.device)

    if cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        hypernetwork.enable_gradient_checkpointing()

    trainable_params = network.prepare_optimizer_params(
        text_encoder_lr=None, unet_lr=cfg.learning_rate / 2, default_lr=cfg.learning_rate
    )
    hyper_params = [{"params": list(hypernetwork.parameters()), "lr": cfg.learning_rate}]
    optimizer = torch.optim.AdamW(
        trainable_params + hyper_params,
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon,
    )

    train_dataset = OmniStyleDataset(
        manifest_path=cfg.train_data_json,
        data_root=cfg.train_data_dir,
        resolution=cfg.resolution,
        text_drop_ratio=cfg.text_drop_ratio,
        require_files_exist=cfg.get("require_files_exist", True),
        max_samples=cfg.get("max_samples", -1),
        skip_ref_image=use_style_cache,
    )
    if len(train_dataset) == 0:
        raise RuntimeError(
            "Dataset is empty. Wait for styleized/ extraction, run build_manifest.py, "
            "and ensure require_files_exist=true."
        )

    num_workers = cfg.get("dataloader_num_workers", 4)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    max_train_steps = cfg.max_train_steps if cfg.max_train_steps > 0 else (
        cfg.num_train_epochs * num_update_steps_per_epoch
    )

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
        num_training_steps=max_train_steps,
    )

    network, hypernetwork, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        network, hypernetwork, optimizer, train_dataloader, lr_scheduler
    )

    vae.to(accelerator.device, dtype=weight_dtype)
    if text_encoder is not None:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    if image_encoder is not None:
        image_encoder.to(accelerator.device)

    if accelerator.is_main_process:
        logger.info(f"  Style cache: {use_style_cache}, Text cache: {use_text_cache}")
        logger.info(f"  Num processes: {accelerator.num_processes}")
        logger.info(
            f"  Hypernet: decoder_blocks={cfg.decoder_blocks}, "
            f"style_loss_w=({cfg.get('style_contrastive_weight', 0)}, "
            f"{cfg.get('weight_diversity_weight', 0)}, {cfg.get('style_separation_weight', 0)})"
        )

    style_contrastive_weight = float(cfg.get("style_contrastive_weight", 0.0))
    weight_diversity_weight = float(cfg.get("weight_diversity_weight", 0.0))
    style_separation_weight = float(cfg.get("style_separation_weight", 0.0))
    style_temperature = float(cfg.get("style_contrastive_temperature", 0.07))
    style_margin = float(cfg.get("style_separation_margin", 0.2))
    use_style_aux_loss = (
        style_contrastive_weight > 0 or weight_diversity_weight > 0 or style_separation_weight > 0
    )

    global_step = 0
    if cfg.get("resume_from_checkpoint"):
        ckpt_dir = resolve_resume_checkpoint(cfg.resume_from_checkpoint, cfg.output_dir)
        global_step = load_checkpoint(
            ckpt_dir,
            accelerator.unwrap_model(network),
            accelerator.unwrap_model(hypernetwork),
        )
        for _ in range(global_step):
            lr_scheduler.step()

    if accelerator.is_main_process:
        accelerator.init_trackers("image2lora_sdxl", config=OmegaConf.to_container(cfg))

    progress_bar = tqdm(
        range(global_step, max_train_steps),
        initial=global_step,
        total=max_train_steps,
        disable=not accelerator.is_local_main_process,
    )

    logger.info("***** Image2LoRA SDXL training *****")
    logger.info(f"  Num pairs = {len(train_dataset)}")
    logger.info(f"  Num LightLoRA layers = {len(accelerator.unwrap_model(network).unet_loras)}")
    logger.info(f"  Resolution = {cfg.resolution}")
    logger.info(f"  Max steps = {max_train_steps}")

    for epoch in range(cfg.num_train_epochs):
        network.train()
        hypernetwork.train()
        train_loss = 0.0
        train_diff_loss = 0.0
        train_style_loss = 0.0

        for batch in train_dataloader:
            with accelerator.accumulate(network, hypernetwork):
                tgt_images = batch["tgt_image"].to(accelerator.device, dtype=weight_dtype)

                if use_style_cache:
                    ref_features = training_cache.get_style_features(
                        batch["ref_image_key"], accelerator.device,
                    )
                else:
                    ref_images = batch["ref_image"].to(accelerator.device)
                    with torch.no_grad():
                        ref_features = image_encoder.encode(ref_images)

                hyper_dtype = torch.bfloat16 if weight_dtype == torch.bfloat16 else torch.float32
                weights, weight_list, style_emb = hypernetwork(ref_features.to(dtype=hyper_dtype))
                update_lora_weights(network, weight_list)

                with torch.no_grad():
                    latents = vae.encode(tgt_images).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                with torch.no_grad():
                    if use_text_cache:
                        prompt_embeds, pooled_prompt_embeds = training_cache.get_text_embeddings(
                            batch["caption"],
                            batch["caption_dropped"],
                            accelerator.device,
                            weight_dtype,
                        )
                    else:
                        captions = [
                            "" if drop else cap
                            for cap, drop in zip(batch["caption"], batch["caption_dropped"])
                        ]
                        prompt_embeds, pooled_prompt_embeds = encode_sdxl_prompt(
                            captions,
                            tokenizer, tokenizer_2,
                            text_encoder, text_encoder_2,
                            accelerator.device,
                        )

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                add_time_ids = get_add_time_ids(
                    cfg.resolution, bsz, weight_dtype, latents.device,
                )

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={
                        "text_embeds": pooled_prompt_embeds,
                        "time_ids": add_time_ids,
                    },
                ).sample

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                diff_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss = diff_loss
                style_loss = diff_loss.new_zeros(())
                style_con_loss = diff_loss.new_zeros(())
                style_div_loss = diff_loss.new_zeros(())
                style_sep_loss = diff_loss.new_zeros(())

                if use_style_aux_loss:
                    ref_keys = batch["ref_image_key"]
                    flat_weights = weights.float().reshape(weights.size(0), -1)
                    style_emb_f = style_emb.float()
                    if style_contrastive_weight > 0:
                        style_con_loss = style_supervised_contrastive_loss(
                            style_emb_f, ref_keys, temperature=style_temperature,
                        )
                        if torch.isfinite(style_con_loss):
                            loss = loss + style_contrastive_weight * style_con_loss
                    if weight_diversity_weight > 0:
                        style_div_loss = weight_diversity_loss(flat_weights, ref_keys)
                        if torch.isfinite(style_div_loss):
                            loss = loss + weight_diversity_weight * style_div_loss
                    if style_separation_weight > 0:
                        style_sep_loss = style_separation_loss(
                            style_emb_f, ref_keys, margin=style_margin,
                        )
                        if torch.isfinite(style_sep_loss):
                            loss = loss + style_separation_weight * style_sep_loss
                    style_loss = (
                        style_contrastive_weight * (style_con_loss if torch.isfinite(style_con_loss) else diff_loss.new_zeros(()))
                        + weight_diversity_weight * (style_div_loss if torch.isfinite(style_div_loss) else diff_loss.new_zeros(()))
                        + style_separation_weight * (style_sep_loss if torch.isfinite(style_sep_loss) else diff_loss.new_zeros(()))
                    )

                if not torch.isfinite(loss):
                    logger.warning(f"Non-finite loss at step {global_step + 1}, using diff_loss only")
                    loss = diff_loss
                    style_loss = diff_loss.new_zeros(())

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        list(network.parameters()) + list(hypernetwork.parameters()),
                        cfg.max_grad_norm,
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if torch.isfinite(loss):
                    train_loss += loss.detach().item()
                    train_diff_loss += diff_loss.detach().item()
                    train_style_loss += style_loss.detach().item()

                if global_step % cfg.logging_steps == 0:
                    n = cfg.logging_steps
                    avg_loss = train_loss / n if train_loss else float("nan")
                    avg_diff = train_diff_loss / n if train_diff_loss else float("nan")
                    avg_style = train_style_loss / n if train_style_loss else 0.0
                    train_loss = 0.0
                    train_diff_loss = 0.0
                    train_style_loss = 0.0
                    lr_val = lr_scheduler.get_last_lr()[0]
                    logs = {
                        "loss": avg_loss,
                        "diff_loss": avg_diff,
                        "style_loss": avg_style,
                        "lr": lr_val,
                        "step": global_step,
                    }
                    def _fmt(k, v):
                        if not isinstance(v, float):
                            return v
                        if k == "lr":
                            return f"{v:.2e}"
                        return f"{v:.4f}"
                    progress_bar.set_postfix(**{k: _fmt(k, v) for k, v in logs.items()})
                    accelerator.log(logs, step=global_step)

                if global_step % cfg.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_checkpoint(
                            cfg.output_dir,
                            accelerator.unwrap_model(network),
                            accelerator.unwrap_model(hypernetwork),
                            global_step, weight_dtype,
                        )

                if global_step >= max_train_steps:
                    break

        if global_step >= max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_checkpoint(
            cfg.output_dir,
            accelerator.unwrap_model(network),
            accelerator.unwrap_model(hypernetwork),
            global_step, weight_dtype,
        )
        logger.info("Training complete!")
    accelerator.end_training()


if __name__ == "__main__":
    main()
