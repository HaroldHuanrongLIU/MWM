"""Loss functions for the SurgWMBench PyTorch MWM baseline."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average values over a boolean mask, returning zero for an empty mask."""

    mask_f = mask.to(dtype=values.dtype)
    while mask_f.ndim < values.ndim:
        mask_f = mask_f.unsqueeze(-1)
    denom = mask_f.sum().clamp_min(1.0)
    return (values * mask_f).sum() / denom


def coordinate_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str = "smooth_l1",
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked coordinate regression loss."""

    assert pred.shape == target.shape, f"Coordinate shapes differ: {pred.shape} vs {target.shape}."
    if loss_type == "mse":
        values = (pred - target).pow(2).sum(dim=-1)
    elif loss_type == "smooth_l1":
        values = F.smooth_l1_loss(pred, target, reduction="none").sum(dim=-1)
    else:
        raise ValueError(f"Unsupported coordinate loss_type={loss_type!r}.")
    if weights is not None:
        assert weights.shape == mask.shape, f"Weight shape {weights.shape} must match mask shape {mask.shape}."
        values = values * weights.to(dtype=values.dtype)
    return masked_mean(values, mask)


def latent_dynamics_loss(pred_next: torch.Tensor, true_next: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE between predicted next latent and stop-gradient true next latent."""

    assert pred_next.shape == true_next.shape, f"Latent shapes differ: {pred_next.shape} vs {true_next.shape}."
    values = (pred_next - true_next.detach()).pow(2).mean(dim=-1)
    return masked_mean(values, mask)


def action_reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Optional action prediction/reconstruction loss."""

    assert pred.shape == target.shape, f"Action shapes differ: {pred.shape} vs {target.shape}."
    values = (pred - target).pow(2).sum(dim=-1)
    return masked_mean(values, mask)


def temporal_smoothness_loss(coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Penalize second-order coordinate differences."""

    if coords.shape[1] < 3:
        return coords.sum() * 0.0
    accel = coords[:, 2:] - 2.0 * coords[:, 1:-1] + coords[:, :-2]
    triple_mask = mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]
    values = accel.pow(2).sum(dim=-1)
    return masked_mean(values, triple_mask)
