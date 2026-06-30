"""Auxiliary losses to reduce style collapse during Image2LoRA training."""

from typing import List, Sequence

import torch
import torch.nn.functional as F


def _safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp(min=eps)


def _ref_pair_mask(ref_keys: Sequence[str], device: torch.device) -> torch.Tensor:
    """Return (B, B) bool mask: True where i and j share the same ref key."""
    keys = list(ref_keys)
    b = len(keys)
    mask = torch.zeros(b, b, dtype=torch.bool, device=device)
    for i in range(b):
        for j in range(b):
            if i != j and keys[i] == keys[j]:
                mask[i, j] = True
    return mask


def style_supervised_contrastive_loss(
    style_emb: torch.Tensor,
    ref_keys: List[str],
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Pull together samples with the same ref_image_key within a batch.
    style_emb: (B, D), L2-normalized.
    """
    if style_emb.size(0) < 2:
        return style_emb.new_zeros(())

    style_emb = _safe_normalize(style_emb.float())
    sim = style_emb @ style_emb.T / max(temperature, 1e-6)
    pos_mask = _ref_pair_mask(ref_keys, style_emb.device)
    if not pos_mask.any():
        return style_emb.new_zeros(())

    b = style_emb.size(0)
    self_mask = torch.eye(b, dtype=torch.bool, device=style_emb.device)
    logits = sim.masked_fill(self_mask, -1e4)

    log_denom = torch.logsumexp(logits, dim=1, keepdim=True)
    log_prob = logits - log_denom

    pos_count = pos_mask.sum(dim=1).clamp(min=1)
    loss = -(log_prob * pos_mask.float()).sum(dim=1) / pos_count
    active = (pos_mask.sum(dim=1) > 0).float()
    if active.sum() == 0:
        return style_emb.new_zeros(())
    out = (loss * active).sum() / active.sum()
    return out if torch.isfinite(out) else style_emb.new_zeros(())


def weight_diversity_loss(
    flat_weights: torch.Tensor,
    ref_keys: List[str],
    min_norm: float = 1e-3,
) -> torch.Tensor:
    """
    Penalize high cosine similarity between LoRA weights from different refs.
    Skip near-zero weights (hypernet init) to avoid NaN from normalize.
    """
    if flat_weights.size(0) < 2:
        return flat_weights.new_zeros(())

    fw = flat_weights.float()
    norms = fw.norm(dim=-1)
    valid = norms >= min_norm
    if valid.sum().item() < 2:
        return flat_weights.new_zeros(())

    fw = fw[valid]
    keys = [k for k, ok in zip(ref_keys, valid.tolist()) if ok]
    normed = _safe_normalize(fw)

    b = fw.size(0)
    sim = normed @ normed.T

    diff_mask = torch.ones(b, b, dtype=torch.bool, device=fw.device)
    diff_mask.fill_diagonal_(False)
    for i in range(b):
        for j in range(b):
            if keys[i] == keys[j]:
                diff_mask[i, j] = False

    if not diff_mask.any():
        return flat_weights.new_zeros(())

    penalty = F.relu(sim[diff_mask])
    out = penalty.mean()
    return out if torch.isfinite(out) else flat_weights.new_zeros(())


def style_separation_loss(
    style_emb: torch.Tensor,
    ref_keys: List[str],
    margin: float = 0.2,
) -> torch.Tensor:
    """
    Hinge loss: different refs should be at least `margin` apart in cosine distance.
    """
    if style_emb.size(0) < 2:
        return style_emb.new_zeros(())

    style_emb = _safe_normalize(style_emb.float())
    sim = style_emb @ style_emb.T
    b = style_emb.size(0)
    losses = []
    for i in range(b):
        for j in range(i + 1, b):
            if ref_keys[i] != ref_keys[j]:
                losses.append(F.relu(sim[i, j] - (1.0 - margin)))
    if not losses:
        return style_emb.new_zeros(())
    out = torch.stack(losses).mean()
    return out if torch.isfinite(out) else style_emb.new_zeros()
