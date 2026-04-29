"""Trajectory metrics for SurgWMBench evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch


def _to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _ensure_batched(values: np.ndarray) -> np.ndarray:
    if values.ndim == 2:
        return values[None, ...]
    if values.ndim != 3:
        raise ValueError(f"Expected trajectory shape [T, 2] or [B, T, 2], got {values.shape}.")
    return values


def _mask_or_ones(mask: torch.Tensor | np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    mask_np = _to_numpy(mask).astype(bool)
    if mask_np.ndim == 1:
        mask_np = mask_np[None, :]
    if mask_np.shape != shape:
        raise ValueError(f"Mask shape {mask_np.shape} must match trajectory prefix {shape}.")
    return mask_np


def _valid_points(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs = []
    for p, t, m in zip(pred, target, mask):
        valid = m.astype(bool)
        pairs.append((p[valid], t[valid]))
    return pairs


def discrete_frechet(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray, mask: torch.Tensor | np.ndarray | None = None) -> float:
    """Discrete Fréchet distance between two 2D trajectories."""

    pred = _to_numpy(pred)
    target = _to_numpy(target)
    if mask is not None or pred.ndim == 3:
        pred_b = _ensure_batched(pred)
        target_b = _ensure_batched(target)
        mask_b = _mask_or_ones(mask, pred_b.shape[:2])
        values = [discrete_frechet(p, t) for p, t in _valid_points(pred_b, target_b, mask_b) if len(p)]
        return float(np.nanmean(values)) if values else float("nan")
    if len(pred) == 0 or len(target) == 0:
        return float("nan")
    ca = np.full((len(pred), len(target)), -1.0, dtype=np.float64)

    def c(i: int, j: int) -> float:
        if ca[i, j] > -1:
            return float(ca[i, j])
        dist = float(np.linalg.norm(pred[i] - target[j]))
        if i == 0 and j == 0:
            ca[i, j] = dist
        elif i > 0 and j == 0:
            ca[i, j] = max(c(i - 1, 0), dist)
        elif i == 0 and j > 0:
            ca[i, j] = max(c(0, j - 1), dist)
        else:
            ca[i, j] = max(min(c(i - 1, j), c(i - 1, j - 1), c(i, j - 1)), dist)
        return float(ca[i, j])

    return c(len(pred) - 1, len(target) - 1)


def hausdorff_distance(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray, mask: torch.Tensor | np.ndarray | None = None) -> float:
    """Symmetric Hausdorff distance between two point sets."""

    pred = _to_numpy(pred)
    target = _to_numpy(target)
    if mask is not None or pred.ndim == 3:
        pred_b = _ensure_batched(pred)
        target_b = _ensure_batched(target)
        mask_b = _mask_or_ones(mask, pred_b.shape[:2])
        values = [hausdorff_distance(p, t) for p, t in _valid_points(pred_b, target_b, mask_b) if len(p)]
        return float(np.nanmean(values)) if values else float("nan")
    if len(pred) == 0 or len(target) == 0:
        return float("nan")
    distances = np.linalg.norm(pred[:, None, :] - target[None, :, :], axis=-1)
    return float(max(distances.min(axis=1).max(), distances.min(axis=0).max()))


def symmetric_hausdorff(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray, mask: torch.Tensor | np.ndarray | None = None) -> float:
    return hausdorff_distance(pred, target, mask)


def ade(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray, mask: torch.Tensor | np.ndarray | None = None) -> float:
    """Average displacement error over valid points."""

    pred_np, target_np = _ensure_batched(_to_numpy(pred)), _ensure_batched(_to_numpy(target))
    mask_np = _mask_or_ones(mask, pred_np.shape[:2])
    dists = np.linalg.norm(pred_np - target_np, axis=-1)
    valid = mask_np.astype(bool)
    if not valid.any():
        return float("nan")
    return float(dists[valid].mean())


def fde(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray, mask: torch.Tensor | np.ndarray | None = None) -> float:
    """Final displacement error at the final valid point of each trajectory."""

    pred_np, target_np = _ensure_batched(_to_numpy(pred)), _ensure_batched(_to_numpy(target))
    mask_np = _mask_or_ones(mask, pred_np.shape[:2])
    finals = []
    for p, t in _valid_points(pred_np, target_np, mask_np):
        if len(p):
            finals.append(np.linalg.norm(p[-1] - t[-1]))
    return float(np.mean(finals)) if finals else float("nan")


def smoothness_metric(coords: torch.Tensor | np.ndarray, mask: torch.Tensor | np.ndarray | None = None) -> float:
    """Mean squared second-order difference over valid coordinate triples."""

    coords_np = _ensure_batched(_to_numpy(coords))
    mask_np = _mask_or_ones(mask, coords_np.shape[:2])
    values = []
    for c, m in zip(coords_np, mask_np):
        if len(c) < 3:
            continue
        triple = m[2:] & m[1:-1] & m[:-2]
        if triple.any():
            accel = c[2:] - 2.0 * c[1:-1] + c[:-2]
            values.extend(np.sum(accel[triple] ** 2, axis=-1).tolist())
    return float(np.mean(values)) if values else float("nan")


def trajectory_smoothness(coords: torch.Tensor | np.ndarray, mask: torch.Tensor | np.ndarray | None = None) -> float:
    return smoothness_metric(coords, mask)


def endpoint_error(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray, mask: torch.Tensor | np.ndarray | None = None) -> float:
    return fde(pred, target, mask)


def trajectory_length_error(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    mask: torch.Tensor | np.ndarray | None = None,
) -> float:
    pred_np, target_np = _ensure_batched(_to_numpy(pred)), _ensure_batched(_to_numpy(target))
    mask_np = _mask_or_ones(mask, pred_np.shape[:2])
    errors = []
    for p, t, m in zip(pred_np, target_np, mask_np):
        valid = np.flatnonzero(m)
        if len(valid) < 2:
            continue
        p_valid = p[valid]
        t_valid = t[valid]
        pred_len = np.linalg.norm(p_valid[1:] - p_valid[:-1], axis=-1).sum()
        target_len = np.linalg.norm(t_valid[1:] - t_valid[:-1], axis=-1).sum()
        errors.append(abs(pred_len - target_len))
    return float(np.mean(errors)) if errors else float("nan")


def error_by_horizon(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    horizons: list[int],
    mask: torch.Tensor | np.ndarray | None = None,
) -> dict[int, float]:
    pred_np, target_np = _ensure_batched(_to_numpy(pred)), _ensure_batched(_to_numpy(target))
    mask_np = _mask_or_ones(mask, pred_np.shape[:2])
    result: dict[int, float] = {}
    for horizon in horizons:
        idx = horizon - 1
        errors = []
        for p, t, m in zip(pred_np, target_np, mask_np):
            valid_indices = np.flatnonzero(m)
            if len(valid_indices) > idx:
                point_idx = valid_indices[idx]
                errors.append(np.linalg.norm(p[point_idx] - t[point_idx]))
        result[horizon] = float(np.mean(errors)) if errors else float("nan")
    return result


def trajectory_metrics(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    mask: torch.Tensor | np.ndarray,
    horizons: list[int] | None = None,
    prefix: str = "",
    pixel_scale: torch.Tensor | np.ndarray | None = None,
) -> dict[str, float]:
    """Compute normalized and optional pixel trajectory metrics."""

    pred_np = _ensure_batched(_to_numpy(pred).astype(np.float64))
    target_np = _ensure_batched(_to_numpy(target).astype(np.float64))
    mask_np = _mask_or_ones(mask, pred_np.shape[:2])
    if pred_np.shape != target_np.shape:
        raise ValueError(f"Prediction and target shapes differ: {pred_np.shape} vs {target_np.shape}.")
    if mask_np.shape != pred_np.shape[:2]:
        raise ValueError(f"Mask shape {mask_np.shape} must match trajectory prefix {pred_np.shape[:2]}.")

    result: dict[str, float] = {
        f"{prefix}ade": ade(pred_np, target_np, mask_np),
        f"{prefix}fde": fde(pred_np, target_np, mask_np),
        f"{prefix}endpoint_error": endpoint_error(pred_np, target_np, mask_np),
        f"{prefix}trajectory_length_error": trajectory_length_error(pred_np, target_np, mask_np),
        f"{prefix}smoothness": smoothness_metric(pred_np, mask_np),
    }
    frechets = []
    hausdorffs = []
    for p, t in _valid_points(pred_np, target_np, mask_np):
        if len(p):
            frechets.append(discrete_frechet(p, t))
            hausdorffs.append(hausdorff_distance(p, t))
    result[f"{prefix}frechet"] = float(np.nanmean(frechets)) if frechets else float("nan")
    result[f"{prefix}hausdorff"] = float(np.nanmean(hausdorffs)) if hausdorffs else float("nan")

    for horizon, value in error_by_horizon(pred_np, target_np, horizons or [], mask_np).items():
        result[f"{prefix}horizon_{horizon}_error"] = value

    if pixel_scale is not None:
        scale = _to_numpy(pixel_scale).astype(np.float64)
        if scale.ndim == 1:
            scale = np.broadcast_to(scale[None, None, :], pred_np.shape)
        elif scale.ndim == 2:
            scale = scale[:, None, :]
        pixel = trajectory_metrics(
            pred_np * scale,
            target_np * scale,
            mask_np,
            horizons=horizons,
            prefix=f"{prefix}pixel_",
            pixel_scale=None,
        )
        result.update(pixel)
    return result


def average_metric_dicts(items: list[dict[str, float]]) -> dict[str, float]:
    """Average a list of metric dictionaries, ignoring NaN values."""

    if not items:
        return {}
    keys = sorted({key for item in items for key in item})
    averaged = {}
    for key in keys:
        values = np.asarray([item[key] for item in items if key in item], dtype=np.float64)
        values = values[~np.isnan(values)]
        averaged[key] = float(values.mean()) if len(values) else float("nan")
    return averaged


def save_metrics(metrics: dict[str, float], output: str | Path, csv_output: str | Path | None = None) -> None:
    """Save metrics to JSON and optionally a one-row CSV."""

    output_path = Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    if csv_output is not None:
        csv_path = Path(csv_output).expanduser()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(metrics))
            writer.writeheader()
            writer.writerow(metrics)
