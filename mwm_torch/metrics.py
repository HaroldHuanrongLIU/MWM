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


def _valid_points(
    pred: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs = []
    for p, t, m in zip(pred, target, mask):
        valid = m.astype(bool)
        pairs.append((p[valid], t[valid]))
    return pairs


def discrete_frechet(pred: np.ndarray, target: np.ndarray) -> float:
    """Discrete Fréchet distance between two 2D trajectories."""

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


def hausdorff_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """Symmetric Hausdorff distance between two point sets."""

    if len(pred) == 0 or len(target) == 0:
        return float("nan")
    distances = np.linalg.norm(pred[:, None, :] - target[None, :, :], axis=-1)
    return float(max(distances.min(axis=1).max(), distances.min(axis=0).max()))


def ade(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray, mask: torch.Tensor | np.ndarray) -> float:
    """Average displacement error over valid points."""

    pred_np, target_np, mask_np = _to_numpy(pred), _to_numpy(target), _to_numpy(mask)
    dists = np.linalg.norm(pred_np - target_np, axis=-1)
    valid = mask_np.astype(bool)
    if not valid.any():
        return float("nan")
    return float(dists[valid].mean())


def fde(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray, mask: torch.Tensor | np.ndarray) -> float:
    """Final displacement error at the final valid point of each trajectory."""

    pred_np, target_np, mask_np = _to_numpy(pred), _to_numpy(target), _to_numpy(mask)
    finals = []
    for p, t in _valid_points(pred_np, target_np, mask_np):
        if len(p):
            finals.append(np.linalg.norm(p[-1] - t[-1]))
    return float(np.mean(finals)) if finals else float("nan")


def smoothness_metric(coords: torch.Tensor | np.ndarray, mask: torch.Tensor | np.ndarray) -> float:
    """Mean squared second-order difference over valid coordinate triples."""

    coords_np, mask_np = _to_numpy(coords), _to_numpy(mask).astype(bool)
    values = []
    for c, m in zip(coords_np, mask_np):
        if len(c) < 3:
            continue
        triple = m[2:] & m[1:-1] & m[:-2]
        if triple.any():
            accel = c[2:] - 2.0 * c[1:-1] + c[:-2]
            values.extend(np.sum(accel[triple] ** 2, axis=-1).tolist())
    return float(np.mean(values)) if values else float("nan")


def trajectory_metrics(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    mask: torch.Tensor | np.ndarray,
    horizons: list[int] | None = None,
    prefix: str = "",
    pixel_scale: torch.Tensor | np.ndarray | None = None,
) -> dict[str, float]:
    """Compute normalized and optional pixel trajectory metrics."""

    pred_np = _to_numpy(pred).astype(np.float64)
    target_np = _to_numpy(target).astype(np.float64)
    mask_np = _to_numpy(mask).astype(bool)
    if pred_np.shape != target_np.shape:
        raise ValueError(f"Prediction and target shapes differ: {pred_np.shape} vs {target_np.shape}.")
    if mask_np.shape != pred_np.shape[:2]:
        raise ValueError(f"Mask shape {mask_np.shape} must match trajectory prefix {pred_np.shape[:2]}.")

    result: dict[str, float] = {
        f"{prefix}ade": ade(pred_np, target_np, mask_np),
        f"{prefix}fde": fde(pred_np, target_np, mask_np),
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

    for horizon in horizons or []:
        errors = []
        idx = horizon - 1
        for p, t, m in zip(pred_np, target_np, mask_np):
            valid_indices = np.flatnonzero(m)
            if len(valid_indices) > idx:
                point_idx = valid_indices[idx]
                errors.append(np.linalg.norm(p[point_idx] - t[point_idx]))
        result[f"{prefix}horizon_{horizon}_error"] = float(np.mean(errors)) if errors else float("nan")

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
