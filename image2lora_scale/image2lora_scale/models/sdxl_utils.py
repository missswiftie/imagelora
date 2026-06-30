"""SDXL text encoding and conditioning helpers."""

from typing import List, Tuple

import torch
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer


def encode_sdxl_prompt(
    captions: List[str],
    tokenizer: CLIPTokenizer,
    tokenizer_2: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    text_encoder_2: CLIPTextModelWithProjection,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (prompt_embeds, pooled_prompt_embeds) for SDXL UNet conditioning."""
    text_inputs = tokenizer(
        captions,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_inputs_2 = tokenizer_2(
        captions,
        padding="max_length",
        max_length=tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = text_inputs.input_ids.to(device)
    input_ids_2 = text_inputs_2.input_ids.to(device)

    prompt_embeds_1 = text_encoder(input_ids, output_hidden_states=True).hidden_states[-2]
    prompt_embeds_2 = text_encoder_2(input_ids_2, output_hidden_states=True).hidden_states[-2]
    prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
    pooled_prompt_embeds = text_encoder_2(input_ids_2).text_embeds
    return prompt_embeds, pooled_prompt_embeds


def get_add_time_ids(
    resolution: int,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    crops_coords_top_left: Tuple[int, int] = (0, 0),
) -> torch.Tensor:
    """SDXL micro-conditioning time ids: (original_h, original_w, crop_y, crop_x, target_h, target_w)."""
    add_time_ids = list((resolution, resolution) + crops_coords_top_left + (resolution, resolution))
    add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)
    return add_time_ids.repeat(batch_size, 1)
