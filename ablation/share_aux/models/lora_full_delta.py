"""Ablation: Hypernetwork predicts full rank-1 ΔW (A|B per layer), no aux."""

from __future__ import annotations

import os
from typing import List, Optional, Type, Union

import safetensors.torch
import torch
import torch.nn as nn
from transformers import CLIPTextModel

UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel"]
UNET_TARGET_LINEAR_NAMES = ["to_q", "to_k", "to_v", "to_out.0"]
TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention"]
LORA_PREFIX_UNET = "lora_unet"
LORA_PREFIX_TEXT_ENCODER = "lora_te"


class FullDeltaLoRAModule(nn.Module):
    """
    Rank-1 LoRA via explicit ΔW = B @ A (outer product).
    Hypernetwork predicts concat(A, B) with dim = in_features + out_features.
    No down_aux / up_aux / shared aux.
    """

    def __init__(
        self,
        lora_name: str,
        org_module: nn.Module,
        multiplier: float = 1.0,
        rank: int = 1,
        alpha: float = 1.0,
        is_train: bool = False,
    ):
        super().__init__()
        if rank != 1:
            raise ValueError("Full ΔW ablation currently supports rank=1 only")
        self.lora_name = lora_name
        self.multiplier = multiplier
        self.rank = rank
        self.network_alpha = alpha
        self.is_train = is_train

        if org_module.__class__.__name__ == "Conv2d":
            in_features = org_module.in_channels
            out_features = org_module.out_channels
        else:
            in_features = org_module.in_features
            out_features = org_module.out_features

        self.in_features = in_features
        self.out_features = out_features
        self.org_module = org_module
        self.weight_dim = in_features + out_features

        if is_train:
            self.weight_embedding = nn.Parameter(torch.zeros(self.weight_dim))

    @property
    def lora_param_count(self) -> int:
        """Effective rank-1 DoF acting on this layer (= in + out)."""
        return self.in_features + self.out_features

    def update_weight(self, weight_embedding: torch.Tensor):
        if weight_embedding.dim() == 1:
            if weight_embedding.shape[0] != self.weight_dim:
                raise ValueError(
                    f"{self.lora_name}: expected {self.weight_dim}, got {weight_embedding.shape}"
                )
        elif weight_embedding.dim() == 2:
            if weight_embedding.shape[1] != self.weight_dim:
                raise ValueError(
                    f"{self.lora_name}: expected dim {self.weight_dim}, got {weight_embedding.shape}"
                )
        else:
            raise ValueError(f"Invalid weight shape {weight_embedding.shape}")
        self._parameters.pop("weight_embedding", None)
        self.weight_embedding = weight_embedding

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def _delta_linear(self, hidden_states: torch.Tensor, weight_embedding: torch.Tensor) -> torch.Tensor:
        """Rank-1 ΔW·x without materializing (out, in) or (B, out, in) matrices."""
        if weight_embedding.dim() == 1:
            a = weight_embedding[: self.in_features]
            b = weight_embedding[self.in_features :]
            proj = (hidden_states * a).sum(dim=-1, keepdim=True)
            return proj * b
        a = weight_embedding[:, : self.in_features]
        b = weight_embedding[:, self.in_features :]
        proj = torch.einsum("b...i,bi->b...", hidden_states, a)
        return torch.einsum("b...,bo->b...o", proj, b)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.weight_embedding.dtype
        org_out = self.org_forward(hidden_states)
        delta_out = self._delta_linear(hidden_states.to(dtype), self.weight_embedding)

        if self.network_alpha is not None:
            delta_out = delta_out * (self.network_alpha / self.rank)
        return org_out + (delta_out * self.multiplier).to(orig_dtype)


class FullDeltaLoRANetwork(nn.Module):
    def __init__(
        self,
        text_encoder: Union[List[nn.Module], nn.Module],
        unet,
        multiplier: float = 1.0,
        rank: int = 1,
        alpha: float = 1.0,
        module_class: Type[FullDeltaLoRAModule] = FullDeltaLoRAModule,
        is_train: bool = False,
        train_unet: bool = True,
        train_text_encoder: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.multiplier = multiplier
        self.rank = rank
        self.alpha = alpha
        self.is_train = is_train

        def create_modules(is_unet: bool, root_module: nn.Module, target_replace_modules: List[str]):
            prefix = LORA_PREFIX_UNET if is_unet else LORA_PREFIX_TEXT_ENCODER
            loras = []
            for name, module in root_module.named_modules():
                if module.__class__.__name__ not in target_replace_modules:
                    continue
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ not in ["Linear", "LoRACompatibleLinear"]:
                        continue
                    if is_unet and child_name.split(".")[-1] not in UNET_TARGET_LINEAR_NAMES:
                        continue
                    lora_name = f"{prefix}_{name}_{child_name}".replace(".", "_")
                    loras.append(
                        module_class(
                            lora_name=lora_name,
                            org_module=child_module,
                            multiplier=self.multiplier,
                            rank=self.rank,
                            alpha=self.alpha,
                            is_train=self.is_train,
                        )
                    )
            return loras

        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]
        self.text_encoder_loras: List[FullDeltaLoRAModule] = []
        if train_text_encoder:
            for te in text_encoders:
                if te is not None:
                    self.text_encoder_loras.extend(
                        create_modules(False, te, TEXT_ENCODER_TARGET_REPLACE_MODULE)
                    )

        self.unet_loras = create_modules(True, unet, UNET_TARGET_REPLACE_MODULE) if train_unet else []
        self.weight_dims = [l.weight_dim for l in self.unet_loras]
        total_dof = sum(self.weight_dims)
        print(
            f"[ablation/full_delta] rank={rank}, alpha={alpha}, "
            f"layers={len(self.unet_loras)}, "
            f"per-layer ΔW DoF = in+out (rank-1), total hypernet DoF={total_dof}"
        )
        print(f"  (baseline LightLoRA hypernet: 96 x {len(self.unet_loras)} = {96 * len(self.unet_loras)})")

        for lora in self.text_encoder_loras + self.unet_loras:
            self.add_module(lora.lora_name, lora)

    def apply_to(self, text_encoder, unet, apply_text_encoder=True, apply_unet=True):
        if not apply_text_encoder:
            for lora in self.text_encoder_loras:
                if hasattr(self, lora.lora_name):
                    delattr(self, lora.lora_name)
            self.text_encoder_loras = []
        if not apply_unet:
            self.unet_loras = []
            self.weight_dims = []

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr):
        """No trainable LoRA-side params; only hypernetwork trains."""
        return []

    def save_weights(self, file, dtype=None, metadata=None):
        meta = metadata or {}
        meta["format"] = "full_delta_no_aux"
        meta["note"] = "LoRA weights are generated entirely by hypernetwork"
        if os.path.splitext(file)[1] == ".safetensors":
            safetensors.torch.save_file({}, file, metadata=meta)
        else:
            torch.save({}, file)
        print(f"Full ΔW ablation: no static LoRA weights (hypernet-only) -> {file}")


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    text_encoder: Union[CLIPTextModel, List[CLIPTextModel]],
    unet,
    **kwargs,
):
    if network_dim is None:
        network_dim = 1
    if network_alpha is None:
        network_alpha = 1.0
    return FullDeltaLoRANetwork(
        text_encoder=text_encoder,
        unet=unet,
        multiplier=multiplier,
        rank=network_dim,
        alpha=network_alpha,
        is_train=kwargs.get("is_train", False),
        train_unet=kwargs.get("train_unet", True),
        train_text_encoder=kwargs.get("train_text_encoder", False),
    )
