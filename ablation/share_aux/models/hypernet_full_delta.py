"""Hypernetwork with variable per-layer output dim = in+out for full rank-1 ΔW."""

from typing import List, Optional, Tuple

import torch
import torch.utils.checkpoint as checkpoint

from image2lora.models.hypernet import ImageHyperDream


class FullDeltaHyperDream(ImageHyperDream):
    """
    Each UNet layer i gets weight_dims[i] = in_i + out_i (rank-1 A|B).
    Decoder internal dim = max(weight_dims); slice when returning weight_list.
    """

    def __init__(
        self,
        weight_dims: List[int],
        image_feat_dim: int = 768,
        decoder_blocks: int = 8,
        sample_iters: int = 4,
    ):
        if not weight_dims:
            raise ValueError("weight_dims must be non-empty")
        self.weight_dims = list(weight_dims)
        max_dim = max(self.weight_dims)
        super().__init__(
            image_feat_dim=image_feat_dim,
            weight_dim=max_dim,
            weight_num=len(weight_dims),
            decoder_blocks=decoder_blocks,
            sample_iters=sample_iters,
        )
        print(
            f"FullDeltaHyperDream: {len(weight_dims)} layers, "
            f"decoder_dim={max_dim}, total_output_dof={sum(weight_dims)}"
        )

    def gen_weight(self, image_features, iters, weight, ensure_grad=0):
        weights = self.img_weight_generator(image_features, iters, weight, ensure_grad)
        raw_list = [w.squeeze(1) for w in weights.split(1, dim=1)]
        weight_list = [raw[..., : d] for raw, d in zip(raw_list, self.weight_dims)]
        return weights, weight_list

    def forward(
        self, image_features: torch.Tensor, iters: Optional[int] = None, weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if self.training and self.gradient_checkpointing:
            ensure_grad = torch.zeros(1, device=image_features.device).requires_grad_(True)
            return checkpoint.checkpoint(
                self.gen_weight, image_features, iters, weight, ensure_grad,
                use_reentrant=False,
            )
        return self.gen_weight(image_features, iters, weight)
