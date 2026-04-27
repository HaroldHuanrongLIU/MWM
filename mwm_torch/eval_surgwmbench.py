"""Evaluation entrypoint for the PyTorch SurgWMBench MWM baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import load_config
from .metrics import average_metric_dicts, save_metrics, trajectory_metrics
from .train_surgwmbench import build_model, gather_by_position, make_dataset, make_loader
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
def predict_sparse(model, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z = model.encode_frames(batch["frames"])
    z_sparse, pos_mask = gather_by_position(z, batch["sparse_positions"])
    sparse_mask = batch["sparse_anchor_mask"] & pos_mask
    sparse_coords = batch["sparse_coords"]
    direct = model.decode_coords(z_sparse)
    if sparse_coords.shape[1] <= 1:
        return direct, sparse_coords, sparse_mask
    actions = sparse_coords[:, 1:] - sparse_coords[:, :-1]
    _, rollout_next = model.rollout(z_sparse[:, 0], sparse_coords[:, 0], actions)
    pred = torch.cat([direct[:, :1], rollout_next], dim=1)
    return pred, sparse_coords, sparse_mask


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, float]:
    device = get_device()
    model, config = load_model(args.checkpoint, args.config, device)
    seed_everything(config.train.seed)
    dataset = make_dataset(config, args.manifest, args.data_root, "eval")
    loader = make_loader(config, dataset, shuffle=False)
    model.eval()

    sparse_metrics = []
    dense_metrics = []
    for batch in loader:
        batch = move_to_device(batch, device)
        pred, target, mask = predict_sparse(model, batch)
        pixel_scale = batch["image_size"] if config.eval.report_pixel_metrics else None
        sparse_metrics.append(
            trajectory_metrics(
                pred,
                target,
                mask,
                horizons=config.eval.horizons,
                prefix="sparse_",
                pixel_scale=pixel_scale,
            )
        )

        if batch["dense_coord_mask"].any():
            z = model.encode_frames(batch["frames"])
            dense_pred = model.decode_coords(z)
            dense_metrics.append(
                trajectory_metrics(
                    dense_pred,
                    batch["dense_coords"],
                    batch["dense_coord_mask"] & batch["frame_mask"],
                    horizons=config.eval.horizons,
                    prefix="dense_pseudo_",
                    pixel_scale=pixel_scale,
                )
            )

    metrics = average_metric_dicts(sparse_metrics)
    if dense_metrics:
        metrics.update(average_metric_dicts(dense_metrics))
    if "sparse_fde" in metrics:
        metrics["sparse_endpoint_error"] = metrics["sparse_fde"]
    save_metrics(metrics, args.output, args.csv_output)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-output", default=None)
    parser.add_argument("--config", default="configs/surgwmbench_mwm.yaml")
    return parser.parse_args()


def main() -> None:
    metrics = evaluate(parse_args())
    for key in sorted(metrics):
        print(f"{key}: {metrics[key]}")


if __name__ == "__main__":
    main()
