"""Evaluation entrypoint for the PyTorch SurgWMBench MWM baseline."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .config import load_config
from .data import SurgWMBenchClipDataset, collate_dense_variable_length, collate_sparse_anchors
from .metrics import average_metric_dicts, trajectory_metrics
from .train_surgwmbench import _append_dt, _data_root, build_model, gather_by_position
from .utils import get_device, move_to_device, seed_everything


def load_model(checkpoint: str | Path, config_path: str | Path | None, device: torch.device):
    config = load_config(config_path)
    model = build_model(config).to(device)
    state = torch.load(Path(checkpoint).expanduser(), map_location="cpu")
    if "model" not in state:
        raise ValueError(f"Checkpoint does not contain a full SurgWMBench model: {checkpoint}")
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    if unexpected:
        raise ValueError(f"Unexpected checkpoint keys: {unexpected}")
    if missing:
        print(f"warning: missing checkpoint keys: {missing}")
    return model, config


@torch.inference_mode()
def predict_sparse(model, batch: dict[str, torch.Tensor], use_time_delta: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z = model.encode_frames(batch["frames"])
    z_sparse, pos_mask = gather_by_position(z, batch["sparse_positions"])
    sparse_mask = batch["sparse_anchor_mask"] & pos_mask
    sparse_coords = batch["sparse_coords"]
    direct = model.decode_coords(z_sparse)
    if sparse_coords.shape[1] <= 1:
        return direct, sparse_coords, sparse_mask
    base_actions = sparse_coords[:, 1:] - sparse_coords[:, :-1]
    actions = _append_dt(base_actions, batch.get("anchor_dt"), use_time_delta)
    _, rollout_next = model.rollout(z_sparse[:, 0], sparse_coords[:, 0], actions)
    pred = torch.cat([direct[:, :1], rollout_next], dim=1)
    return pred, sparse_coords, sparse_mask


@torch.inference_mode()
def predict_dense(model, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z = model.encode_frames(batch["frames"])
    pred = model.decode_coords(z)
    mask = batch["dense_coord_mask"] & batch["frame_mask"]
    return pred, batch["dense_coords"], mask


def _per_item_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    batch: dict[str, Any],
    horizons: list[int],
    prefix: str,
    report_pixel_metrics: bool,
) -> list[dict[str, Any]]:
    rows = []
    for idx in range(pred.shape[0]):
        pixel_scale = batch["image_size"][idx : idx + 1] if report_pixel_metrics else None
        metrics = trajectory_metrics(
            pred[idx : idx + 1],
            target[idx : idx + 1],
            mask[idx : idx + 1],
            horizons=horizons,
            prefix=prefix,
            pixel_scale=pixel_scale,
        )
        rows.append(
            {
                "metrics": metrics,
                "difficulty": batch.get("difficulties", [None] * pred.shape[0])[idx],
                "source_video_id": batch.get("source_video_ids", [""] * pred.shape[0])[idx],
                "trajectory_id": batch.get("trajectory_ids", batch.get("clip_ids", [""] * pred.shape[0]))[idx],
            }
        )
    return rows


def _average_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    return average_metric_dicts([row["metrics"] for row in rows])


def _by_difficulty(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        difficulty = str(row.get("difficulty") if row.get("difficulty") is not None else "null")
        buckets.setdefault(difficulty, []).append(row)
    return {difficulty: _average_rows(items) for difficulty, items in sorted(buckets.items())}


def _flatten(prefix: str, metrics: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            flat.update(_flatten(f"{prefix}{key}.", value))
        elif isinstance(value, (int, float)):
            flat[f"{prefix}{key}"] = float(value)
    return flat


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = get_device()
    model, config = load_model(args.checkpoint, args.config, device)
    seed_everything(config.train.seed)
    data_root = _data_root(args, config)
    manifest = args.manifest or config.data.test_manifest
    if args.interpolation_method:
        config.data.interpolation_method = args.interpolation_method
    dense_eval = bool(args.dense_pseudo_eval or config.eval.dense_pseudo_eval)
    model.eval()

    sparse_dataset = SurgWMBenchClipDataset(
        data_root,
        manifest,
        interpolation_method=config.data.interpolation_method,
        image_size=config.model.image_size,
        frame_sampling="sparse_anchors",
        use_dense_pseudo=False,
    )
    sparse_loader = torch.utils.data.DataLoader(
        sparse_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.train.num_workers > 0,
        collate_fn=collate_sparse_anchors,
    )

    sparse_rows: list[dict[str, Any]] = []
    num_failed = 0
    for batch in sparse_loader:
        try:
            batch = move_to_device(batch, device)
            pred, target, mask = predict_sparse(model, batch, use_time_delta=config.model.use_time_delta)
            sparse_rows.extend(
                _per_item_metrics(
                    pred,
                    target,
                    mask,
                    batch,
                    config.eval.horizons,
                    "sparse_",
                    config.eval.report_pixel_metrics,
                )
            )
        except Exception as exc:
            num_failed += int(batch["frames"].shape[0])
            print(f"warning: failed sparse batch: {exc}")

    dense_rows: list[dict[str, Any]] = []
    if dense_eval:
        dense_dataset = SurgWMBenchClipDataset(
            data_root,
            manifest,
            interpolation_method=config.data.interpolation_method,
            image_size=config.model.image_size,
            frame_sampling="dense",
            use_dense_pseudo=True,
        )
        dense_loader = torch.utils.data.DataLoader(
            dense_dataset,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=config.train.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=config.train.num_workers > 0,
            collate_fn=collate_dense_variable_length,
        )
        dense_prefix = f"dense_pseudo_{config.data.interpolation_method}_"
        for batch in dense_loader:
            try:
                batch = move_to_device(batch, device)
                pred, target, mask = predict_dense(model, batch)
                dense_rows.extend(
                    _per_item_metrics(
                        pred,
                        target,
                        mask,
                        batch,
                        config.eval.horizons,
                        dense_prefix,
                        config.eval.report_pixel_metrics,
                    )
                )
            except Exception as exc:
                num_failed += int(batch["frames"].shape[0])
                print(f"warning: failed dense batch: {exc}")

    overall = _average_rows(sparse_rows)
    if dense_rows:
        overall.update(_average_rows(dense_rows))
    result: dict[str, Any] = {
        "dataset_root": str(data_root),
        "manifest": str(manifest),
        "checkpoint": str(args.checkpoint),
        "interpolation_method": config.data.interpolation_method,
        "primary_target": "sparse_human_anchors",
        "dense_target": "pseudo_coordinates" if dense_eval else None,
        "metrics_overall": overall,
        "metrics_by_difficulty": _by_difficulty(sparse_rows),
        "num_clips": len(sparse_dataset),
        "num_failed_clips": num_failed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if dense_rows:
        result["dense_metrics_by_difficulty"] = _by_difficulty(dense_rows)

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if args.csv_output:
        flat = _flatten("", result["metrics_overall"])
        csv_path = Path(args.csv_output).expanduser()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(flat))
            writer.writeheader()
            writer.writerow(flat)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", "--data-root", dest="dataset_root", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-output", default=None)
    parser.add_argument("--config", default="configs/surgwmbench_mwm.yaml")
    parser.add_argument("--interpolation-method", default=None)
    parser.add_argument("--dense-pseudo-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    result = evaluate(parse_args())
    for key in sorted(result["metrics_overall"]):
        print(f"{key}: {result['metrics_overall'][key]}")


if __name__ == "__main__":
    main()
