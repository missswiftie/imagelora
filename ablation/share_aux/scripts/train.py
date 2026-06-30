#!/usr/bin/env python3
"""Ablation train: hypernet predicts full rank-1 ΔW per layer, no aux."""

import argparse
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
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

ABLATION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(ABLATION_ROOT))
sys.path.insert(0, REPO_ROOT)

from image2lora.data.dataset import ImagePairDataset, collate_fn
from image2lora.models.encoder import DINOv2Encoder
from ablation.share_aux.models.hypernet_full_delta import FullDeltaHyperDream
from ablation.share_aux.models.lora_full_delta import create_network

logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/train.yaml")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--train_batch_size", type=int, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--max_train_steps", type=int, default=None)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    return p.parse_args()


def load_config(args):
    config_path = args.config
    if not os.path.isabs(config_path):
        candidate = os.path.join(ABLATION_ROOT, config_path)
        if not os.path.isfile(candidate):
            candidate = os.path.join(REPO_ROOT, config_path)
        config_path = candidate
    cfg = OmegaConf.load(config_path)
    for key in ("output_dir", "train_batch_size", "gradient_accumulation_steps",
                "learning_rate", "max_train_steps", "resume_from_checkpoint"):
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val

    model_path = cfg.pretrained_model_name_or_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(REPO_ROOT, model_path)
    cfg.pretrained_model_name_or_path = model_path
    cfg.local_files_only = os.path.isdir(model_path)

    if not os.path.isabs(cfg.output_dir):
        cfg.output_dir = os.path.join(REPO_ROOT, cfg.output_dir)

    for key in ("train_data_meta", "train_data_dir"):
        if not os.path.isabs(cfg[key]):
            cfg[key] = os.path.join(REPO_ROOT, cfg[key])

    dinov2 = cfg.get("dinov2_model_path")
    if dinov2 and not os.path.isabs(dinov2):
        cfg.dinov2_model_path = os.path.join(REPO_ROOT, dinov2)
    return cfg


def save_checkpoint(output_dir, network, hypernetwork, global_step, weight_dtype):
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    network.save_weights(os.path.join(ckpt_dir, "lora_full_delta.safetensors"), weight_dtype)
    hyper_sd = {k: v.detach().cpu().to(weight_dtype) for k, v in hypernetwork.state_dict().items()}
    save_file(hyper_sd, os.path.join(ckpt_dir, "hypernetwork.safetensors"))
    logger.info(f"Saved checkpoint to {ckpt_dir}")


def load_checkpoint(ckpt_dir, hypernetwork):
    hypernetwork.load_state_dict(load_file(os.path.join(ckpt_dir, "hypernetwork.safetensors")))
    return int(os.path.basename(ckpt_dir).split("-")[-1])


def resolve_resume_checkpoint(resume_path, output_dir):
    if resume_path in (None, "", "latest"):
        ckpts = sorted(
            [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[-1]),
        )
        if not ckpts:
            raise ValueError(f"No checkpoint in {output_dir}")
        return os.path.join(output_dir, ckpts[-1])
    if os.path.isdir(resume_path):
        return resume_path
    return os.path.join(output_dir, resume_path)


def update_lora_weights(network, weight_list):
    actual = network.module if hasattr(network, "module") else network
    for weight, lora_layer in zip(weight_list, actual.unet_loras):
        if weight.dim() == 3:
            weight = weight.view(weight.size(0), -1)
        elif weight.dim() == 2 and weight.size(0) == 1:
            weight = weight.view(-1)
        lora_layer.update_weight(weight)


def enable_memory_savings(unet, vae, cfg):
    if cfg.get("enable_attention_slicing", True):
        slice_mode = cfg.get("attention_slice_size", "max")
        if hasattr(unet, "set_attention_slice"):
            unet.set_attention_slice(slice_mode)
        elif hasattr(unet, "enable_attention_slicing"):
            unet.enable_attention_slicing(slice_mode if slice_mode != "max" else 1)
    if cfg.get("enable_xformers", True):
        try:
            unet.enable_xformers_memory_efficient_attention()
        except Exception as exc:
            logger.warning(f"xFormers unavailable, skip: {exc}")
    if cfg.get("enable_vae_slicing", True) and hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
    if cfg.get("enable_vae_tiling", False) and hasattr(vae, "enable_tiling"):
        vae.enable_tiling()


def maybe_offload(*modules):
    for module in modules:
        module.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def maybe_onload(module, device, dtype=None):
    if dtype is None:
        module.to(device)
    else:
        module.to(device, dtype=dtype)


def encode_batch(cfg, device, weight_dtype, image_encoder, vae, text_encoder, tokenizer, ref_images, tgt_images, captions):
    cpu_offload = cfg.get("cpu_offload_encoders", True)

    with torch.no_grad():
        maybe_onload(image_encoder, device, weight_dtype if cfg.get("dinov2_fp16", True) else torch.float32)
        ref_features = image_encoder.encode(ref_images)
        if cpu_offload:
            maybe_offload(image_encoder)

    del ref_images

    with torch.no_grad():
        maybe_onload(vae, device, weight_dtype)
        latents = vae.encode(tgt_images).latent_dist.sample() * vae.config.scaling_factor
        if cpu_offload:
            maybe_offload(vae)

        maybe_onload(text_encoder, device, weight_dtype)
        text_inputs = tokenizer(
            captions, padding="max_length",
            max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt",
        )
        encoder_hidden_states = text_encoder(text_inputs.input_ids.to(device))[0]
        if cpu_offload:
            maybe_offload(text_encoder)

    del tgt_images
    return ref_features, latents, encoder_hidden_states


def main():
    args = parse_args()
    cfg = load_config(args)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=cfg.report_to,
        project_config=ProjectConfiguration(
            project_dir=cfg.output_dir,
            logging_dir=os.path.join(cfg.output_dir, "logs"),
        ),
    )
    set_seed(cfg.seed)

    if accelerator.is_main_process:
        os.makedirs(cfg.output_dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(cfg.output_dir, "config.yaml"))

    weight_dtype = torch.float16 if cfg.mixed_precision == "fp16" else torch.float32
    model_path = cfg.pretrained_model_name_or_path
    load_kw = {"local_files_only": cfg.local_files_only}

    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", **load_kw)
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", **load_kw)
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", **load_kw)
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", **load_kw)
    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler", **load_kw)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    network = create_network(
        1.0, cfg.rank, cfg.network_alpha,
        text_encoder, unet,
        is_train=True, train_unet=True, train_text_encoder=False,
    )
    network.apply_to(text_encoder, unet, apply_text_encoder=False, apply_unet=True)

    hypernetwork = FullDeltaHyperDream(
        weight_dims=network.weight_dims,
        decoder_blocks=cfg.decoder_blocks,
        sample_iters=cfg.sample_iters,
    )
    hypernetwork.set_lilora(network.unet_loras)

    dinov2_path = cfg.get("dinov2_model_path")
    image_encoder = DINOv2Encoder(
        model_name=cfg.dinov2_model,
        model_path=dinov2_path,
        local_files_only=bool(dinov2_path and os.path.isdir(dinov2_path)),
    )
    image_encoder.eval()

    if cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        hypernetwork.enable_gradient_checkpointing()

    enable_memory_savings(unet, vae, cfg)

    optimizer = torch.optim.AdamW(
        [{"params": list(hypernetwork.parameters()), "lr": cfg.learning_rate}],
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon,
    )

    train_dataset = ImagePairDataset(
        meta_path=cfg.train_data_meta,
        data_root=cfg.train_data_dir,
        resolution=cfg.resolution,
        text_drop_ratio=cfg.text_drop_ratio,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.get("dataloader_num_workers", 4),
        pin_memory=True,
        persistent_workers=cfg.get("dataloader_num_workers", 4) > 0,
    )

    num_update = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    max_train_steps = cfg.max_train_steps if cfg.max_train_steps > 0 else cfg.num_train_epochs * num_update

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
        num_training_steps=max_train_steps,
    )

    # network 无可训参数，仅 hypernetwork 需 DDP
    hypernetwork, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        hypernetwork, optimizer, train_dataloader, lr_scheduler,
    )
    network.to(accelerator.device)
    unet.to(accelerator.device, dtype=weight_dtype)

    encoder_dtype = weight_dtype if cfg.get("dinov2_fp16", True) else torch.float32
    if cfg.get("cpu_offload_encoders", True):
        vae.to("cpu", dtype=weight_dtype)
        text_encoder.to("cpu", dtype=weight_dtype)
        image_encoder.to("cpu", dtype=encoder_dtype)
    else:
        vae.to(accelerator.device, dtype=weight_dtype)
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        image_encoder.to(accelerator.device, dtype=encoder_dtype)

    global_step = 0
    if cfg.get("resume_from_checkpoint"):
        ckpt = resolve_resume_checkpoint(cfg.resume_from_checkpoint, cfg.output_dir)
        global_step = load_checkpoint(ckpt, accelerator.unwrap_model(hypernetwork))
        for _ in range(global_step):
            lr_scheduler.step()

    progress_bar = tqdm(
        range(global_step, max_train_steps),
        initial=global_step,
        total=max_train_steps,
        disable=not accelerator.is_local_main_process,
    )

    if accelerator.is_main_process:
        logger.info(f"  Num processes: {accelerator.num_processes}")
        logger.info(f"  Per-GPU batch: {cfg.train_batch_size}, grad_accum: {cfg.gradient_accumulation_steps}")
        logger.info(f"  Effective batch: {cfg.train_batch_size * cfg.gradient_accumulation_steps * accelerator.num_processes}")

    for epoch in range(cfg.num_train_epochs):
        network.train()
        hypernetwork.train()
        train_loss = 0.0

        for batch in train_dataloader:
            with accelerator.accumulate(network, hypernetwork):
                ref_images = batch["ref_image"].to(accelerator.device, non_blocking=True)
                tgt_images = batch["tgt_image"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)

                ref_features, latents, encoder_hidden_states = encode_batch(
                    cfg, accelerator.device, weight_dtype,
                    image_encoder, vae, text_encoder, tokenizer,
                    ref_images, tgt_images, batch["caption"],
                )
                with accelerator.autocast():
                    _, weight_list = hypernetwork(ref_features)
                del ref_features
                update_lora_weights(network, weight_list)
                del weight_list

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                target = noise if noise_scheduler.config.prediction_type == "epsilon" else noise_scheduler.get_velocity(
                    latents, noise, timesteps,
                )
                del latents

                with accelerator.autocast():
                    model_pred = unet(
                        noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states,
                    ).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                del noisy_latents, encoder_hidden_states, timesteps, noise, target, model_pred
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        accelerator.unwrap_model(hypernetwork).parameters(),
                        cfg.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                train_loss += loss.detach().item()
                if global_step % cfg.logging_steps == 0:
                    logs = {"loss": train_loss / cfg.logging_steps, "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                    train_loss = 0.0
                    progress_bar.set_postfix(loss=f"{logs['loss']:.4f}", lr=f"{logs['lr']:.2e}")
                    accelerator.log(logs, step=global_step)
                if global_step % cfg.checkpointing_steps == 0 and accelerator.is_main_process:
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
    accelerator.end_training()


if __name__ == "__main__":
    main()
