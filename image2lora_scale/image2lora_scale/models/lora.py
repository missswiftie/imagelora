"""LightLoRA adapter for Stable Diffusion XL UNet."""

import os
from typing import List, Optional, Type, Union

import safetensors.torch
import torch
import torch.nn.functional as F


class LoRAModule(torch.nn.Module):
    """LightLoRA module: low-rank weights predicted by hypernetwork."""

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        rank=1,
        alpha=1,
        down_dim: int = 64,
        up_dim: int = 32,
        is_train: bool = False,
    ):
        super().__init__()
        self.lora_name = lora_name

        if org_module.__class__.__name__ == "Conv2d":
            in_features = org_module.in_channels
            out_features = org_module.out_channels
        else:
            in_features = org_module.in_features
            out_features = org_module.out_features

        self.rank = rank
        down_aux = torch.empty(down_dim, in_features)
        up_aux = torch.empty(out_features, up_dim)
        torch.nn.init.orthogonal_(down_aux, gain=1)
        torch.nn.init.orthogonal_(up_aux, gain=1)

        self.down_aux = torch.nn.Parameter(down_aux)
        self.up_aux = torch.nn.Parameter(up_aux)
        self.in_features = in_features
        self.out_features = out_features
        self.down_dim = down_dim
        self.up_dim = up_dim
        self.network_alpha = alpha
        self.is_train = is_train
        self.org_module = org_module
        self.multiplier = multiplier

        if is_train:
            down_weight = torch.empty(rank, down_dim)
            up_weight = torch.empty(up_dim, rank)
            torch.nn.init.xavier_normal_(down_weight)
            torch.nn.init.zeros_(up_weight)
            weight = torch.concat([torch.flatten(down_weight), torch.flatten(up_weight)])
            self.weight_embedding = torch.nn.Parameter(weight)

    def update_weight(self, weight_embedding):
        expected_dim = (self.up_dim + self.down_dim) * self.rank
        if len(weight_embedding.shape) > 2 or (
            len(weight_embedding.shape) == 2 and weight_embedding.shape[1] != expected_dim
        ) or (len(weight_embedding.shape) == 1 and weight_embedding.shape != (expected_dim,)):
            raise ValueError(
                f"weight_embedding shape {weight_embedding.shape} invalid, expected dim {expected_dim}"
            )
        self._parameters.pop("weight_embedding", None)
        self.weight_embedding = weight_embedding

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.weight_embedding.dtype
        org_out = self.org_forward(hidden_states)

        down_aux = self.down_aux.to(self.weight_embedding.device)
        up_aux = self.up_aux.to(self.weight_embedding.device)
        down_weight, up_weight = self.weight_embedding.split(
            [self.down_dim * self.rank, self.up_dim * self.rank], dim=-1
        )

        if self.weight_embedding.dim() == 1:
            down_weight = down_weight.reshape(self.rank, -1)
            up_weight = up_weight.reshape(-1, self.rank)
            down = down_weight @ down_aux
            up = up_aux @ up_weight
            delta_out = F.linear(hidden_states.to(dtype), down)
            delta_out = F.linear(delta_out, up)
        else:
            down_weight = down_weight.reshape(self.weight_embedding.size(0), self.rank, -1)
            up_weight = up_weight.reshape(self.weight_embedding.size(0), -1, self.rank)
            down = down_weight @ down_aux
            up = up_aux @ up_weight
            delta_out = torch.einsum("b r i, b ... i -> b ... r", down, hidden_states.to(dtype))
            delta_out = torch.einsum("b o r, b ... r -> b ... o", up, delta_out)

        if self.network_alpha is not None:
            delta_out *= self.network_alpha / self.rank
        return org_out + (delta_out * self.multiplier).to(orig_dtype)


UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel"]
UNET_TARGET_LINEAR_NAMES = ["to_q", "to_k", "to_v", "to_out.0"]
TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention"]
LORA_PREFIX_UNET = "lora_unet"
LORA_PREFIX_TEXT_ENCODER = "lora_te"


class LoRANetwork(torch.nn.Module):
    def __init__(
        self,
        text_encoder: Union[List[torch.nn.Module], torch.nn.Module],
        unet,
        multiplier: float = 1.0,
        rank: int = 1,
        alpha: float = 1,
        down_dim: int = 64,
        up_dim: int = 32,
        module_class: Type[object] = LoRAModule,
        is_train: bool = False,
        train_unet: bool = True,
        train_text_encoder: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.multiplier = multiplier
        self.rank = rank
        self.alpha = alpha
        self.down_dim = down_dim
        self.up_dim = up_dim
        self.is_train = is_train

        print(f"Creating SDXL LightLoRA. rank={rank}, alpha={alpha}, aux={down_dim}/{up_dim}")

        def create_modules(is_unet: bool, root_module: torch.nn.Module, target_replace_modules: List[str]):
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
                            down_dim=self.down_dim,
                            up_dim=self.up_dim,
                            is_train=self.is_train,
                        )
                    )
            return loras

        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]
        self.text_encoder_loras = []
        if train_text_encoder:
            for te in text_encoders:
                if te is not None:
                    self.text_encoder_loras.extend(
                        create_modules(False, te, TEXT_ENCODER_TARGET_REPLACE_MODULE)
                    )

        self.unet_loras = create_modules(True, unet, UNET_TARGET_REPLACE_MODULE) if train_unet else []
        print(f"SDXL LightLoRA modules: TE={len(self.text_encoder_loras)}, UNet={len(self.unet_loras)}")

        for lora in self.text_encoder_loras + self.unet_loras:
            self.add_module(lora.lora_name, lora)

    def apply_to(self, text_encoder, unet, apply_text_encoder=True, apply_unet=True):
        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]
        if not apply_text_encoder:
            for lora in self.text_encoder_loras:
                if hasattr(self, lora.lora_name):
                    delattr(self, lora.lora_name)
            self.text_encoder_loras = []
        if not apply_unet:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr):
        self.requires_grad_(True)
        all_params = []

        def enumerate_params(loras):
            params = []
            for lora in loras:
                params.extend(lora.parameters())
            return params

        if self.text_encoder_loras:
            param_data = {"params": enumerate_params(self.text_encoder_loras)}
            if text_encoder_lr is not None:
                param_data["lr"] = text_encoder_lr
            all_params.append(param_data)

        if self.unet_loras:
            param_data = {"params": enumerate_params(self.unet_loras)}
            if unet_lr is not None:
                param_data["lr"] = unet_lr
            all_params.append(param_data)

        return all_params

    def save_weights(self, file, dtype=None, metadata=None):
        state_dict = {}
        for key, value in self.state_dict().items():
            if "weight_embedding" in key or "org_module" in key:
                continue
            v = value.detach().clone().to("cpu")
            if dtype is not None:
                v = v.to(dtype)
            state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            safetensors.torch.save_file(state_dict, file, metadata or {})
        else:
            torch.save(state_dict, file)
        print(f"LightLoRA aux weights saved to {file}")


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    text_encoder: Union[torch.nn.Module, List[torch.nn.Module]],
    unet,
    **kwargs,
):
    if network_dim is None:
        network_dim = 1
    if network_alpha is None:
        network_alpha = 1.0
    return LoRANetwork(
        text_encoder=text_encoder,
        unet=unet,
        multiplier=multiplier,
        rank=network_dim,
        alpha=network_alpha,
        down_dim=kwargs.get("down_dim", 64),
        up_dim=kwargs.get("up_dim", 32),
        is_train=kwargs.get("is_train", False),
        train_unet=kwargs.get("train_unet", True),
        train_text_encoder=kwargs.get("train_text_encoder", False),
    )
