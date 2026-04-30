"""Training entrypoint for the PyTorch SurgWMBench MWM baseline."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .config import SurgWMBenchConfig, dataclass_to_dict, load_config
from .data import (
    SurgWMBenchClipDataset,
    SurgWMBenchRawVideoDataset,
    SurgWMBenchSSLFrameDataset,
    collate_dense_variable_length,
    collate_sparse_anchors,
    collate_ssl_video,
    surgwmbench_collate,
)
from .losses import coordinate_loss, latent_dynamics_loss, temporal_smoothness_loss
from .metrics import average_metric_dicts, trajectory_metrics
from .models import MWMSurgWMBenchModel, MaskedVisualAutoencoder
from .utils import autocast_context, ensure_dir, get_device, make_grad_scaler, move_to_device, seed_everything


def build_mae(config: SurgWMBenchConfig) -> MaskedVisualAutoencoder:
    model_cfg = config.model
    return MaskedVisualAutoencoder(
        image_size=model_cfg.image_size,
        patch_size=model_cfg.patch_size,
        latent_dim=model_cfg.latent_dim,
        hidden_dim=model_cfg.hidden_dim,
        encoder_depth=model_cfg.encoder_depth,
        decoder_depth=model_cfg.decoder_depth,
        num_heads=model_cfg.num_heads,
        decoder_num_heads=model_cfg.decoder_num_heads,
        conv_stem_channels=model_cfg.conv_stem_channels,
        mask_ratio=model_cfg.mask_ratio,
    )


def build_model(config: SurgWMBenchConfig) -> MWMSurgWMBenchModel:
    model_cfg = config.model
    return MWMSurgWMBenchModel(
        image_size=model_cfg.image_size,
        patch_size=model_cfg.patch_size,
        latent_dim=model_cfg.latent_dim,
        hidden_dim=model_cfg.hidden_dim,
        coord_dim=model_cfg.coord_dim,
        action_dim=model_cfg.action_dim if model_cfg.use_time_delta else model_cfg.coord_dim,
        coord_embed_dim=model_cfg.coord_embed_dim,
        action_embed_dim=model_cfg.action_embed_dim,
        action_type=model_cfg.action_type,
        dynamics_type=model_cfg.dynamics_type,
        mask_ratio=model_cfg.mask_ratio,
        encoder_depth=model_cfg.encoder_depth,
        decoder_depth=model_cfg.decoder_depth,
        num_heads=model_cfg.num_heads,
        decoder_num_heads=model_cfg.decoder_num_heads,
        conv_stem_channels=model_cfg.conv_stem_channels,
    )


def make_dataset(
    config: SurgWMBenchConfig,
    manifest: str | Path,
    data_root: str | Path,
    mode: str,
) -> SurgWMBenchClipDataset:
    dense = mode in {"train_dynamics_dense_aux", "dense_eval"} or config.data.use_dense_pseudo
    frame_sampling = "dense" if dense else "sparse_anchors"
    return SurgWMBenchClipDataset(
        dataset_root=data_root,
        manifest=manifest,
        interpolation_method=config.data.interpolation_method,
        image_size=config.model.image_size,
        frame_sampling=frame_sampling,
        max_frames=config.data.max_frames_per_clip,
        use_dense_pseudo=dense,
    )


def make_loader(config: SurgWMBenchConfig, dataset: torch.utils.data.Dataset, shuffle: bool) -> DataLoader:
    if isinstance(dataset, SurgWMBenchClipDataset):
        collate_fn = collate_dense_variable_length if dataset.frame_sampling in {"dense", "all", "window"} else collate_sparse_anchors
    elif isinstance(dataset, (SurgWMBenchRawVideoDataset, SurgWMBenchSSLFrameDataset)):
        collate_fn = collate_ssl_video
    else:
        collate_fn = surgwmbench_collate
    return DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=shuffle,
        num_workers=config.train.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.train.num_workers > 0,
        collate_fn=collate_fn,
    )


def _data_root(args: argparse.Namespace, config: SurgWMBenchConfig) -> str:
    return str(getattr(args, "dataset_root", None) or getattr(args, "data_root", None) or config.data.dataset_root)


def _manifest(args: argparse.Namespace, attr: str, fallback: str) -> str:
    value = getattr(args, attr, None)
    return str(value or fallback)


def build_ssl_dataset(args: argparse.Namespace, config: SurgWMBenchConfig):
    root = _data_root(args, config)
    ssl_cfg = config.data.ssl
    source = args.ssl_source or ssl_cfg.get("source", "raw_videos")
    manifest = args.manifest or args.train_manifest or config.data.train_manifest
    clip_length = int(args.clip_length or ssl_cfg.get("clip_length", 16))
    stride = int(args.stride or ssl_cfg.get("stride", 4))
    if source == "raw_videos":
        try:
            return SurgWMBenchRawVideoDataset(
                dataset_root=root,
                split="train",
                source_video_manifest=args.source_video_manifest,
                clip_length=clip_length,
                stride=stride,
                image_size=config.model.image_size,
                backend=args.ssl_backend or ssl_cfg.get("backend", "opencv"),
                max_videos=args.max_videos if args.max_videos is not None else ssl_cfg.get("max_videos"),
                max_clips_per_video=(
                    args.max_clips_per_video
                    if args.max_clips_per_video is not None
                    else ssl_cfg.get("max_clips_per_video")
                ),
            )
        except Exception:
            if not bool(ssl_cfg.get("fallback_to_clip_frames", True)):
                raise
            print("warning: raw video MAE dataset failed; falling back to extracted clip frames")
    return SurgWMBenchSSLFrameDataset(
        dataset_root=root,
        manifest=manifest,
        image_size=config.model.image_size,
        sequence_length=clip_length,
        stride=stride,
    )


def gather_by_position(sequence: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather `[B, A, D]` values from `[B, T, D]` using padded positions."""

    valid = positions >= 0
    safe = positions.clamp_min(0)
    index = safe.unsqueeze(-1).expand(-1, -1, sequence.shape[-1])
    gathered = torch.gather(sequence, dim=1, index=index)
    return gathered, valid


def _append_dt(actions: torch.Tensor, dt: torch.Tensor | None, use_time_delta: bool) -> torch.Tensor:
    if not use_time_delta:
        return actions
    if dt is None:
        dt = torch.ones(actions.shape[:2], dtype=actions.dtype, device=actions.device)
    return torch.cat([actions, dt.to(dtype=actions.dtype, device=actions.device).unsqueeze(-1)], dim=-1)


def _dense_dt(batch: dict[str, Any], width: int) -> torch.Tensor:
    if width <= 0:
        return torch.zeros(batch["frames"].shape[0], 0, dtype=torch.float32, device=batch["frames"].device)
    frame_indices = batch["frame_indices"].to(torch.float32)
    denom = torch.clamp(batch["num_frames"].to(torch.float32) - 1.0, min=1.0).unsqueeze(1)
    return (frame_indices[:, 1:] - frame_indices[:, :-1]).clamp_min(0.0) / denom


def supervised_losses(
    model: MWMSurgWMBenchModel,
    batch: dict[str, Any],
    config: SurgWMBenchConfig,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    """Compute sparse-primary and optional dense-pseudo world-model losses."""

    z = model.encode_frames(batch["frames"])
    z_sparse, pos_mask = gather_by_position(z, batch["sparse_positions"])
    sparse_mask = batch["sparse_anchor_mask"] & pos_mask
    sparse_coords = batch["sparse_coords"]

    direct_sparse = model.decode_coords(z_sparse)
    sparse_direct_loss = coordinate_loss(
        direct_sparse, sparse_coords, sparse_mask, config.loss.coord_loss
    )
    sparse_smooth = temporal_smoothness_loss(direct_sparse, sparse_mask)
    sparse_latent = z.sum() * 0.0
    sparse_transition = z.sum() * 0.0
    sparse_rollout = z.sum() * 0.0
    rollout_sparse = direct_sparse

    if sparse_coords.shape[1] > 1:
        base_actions = sparse_coords[:, 1:] - sparse_coords[:, :-1]
        actions = _append_dt(base_actions, batch.get("anchor_dt"), config.model.use_time_delta)
        pair_mask = sparse_mask[:, 1:] & sparse_mask[:, :-1]
        pred_next_z = model.predict_next_latents(z_sparse, sparse_coords, actions)
        sparse_latent = latent_dynamics_loss(pred_next_z, z_sparse[:, 1:], pair_mask)
        pred_next_coords = model.decode_coords(pred_next_z)
        sparse_transition = coordinate_loss(
            pred_next_coords, sparse_coords[:, 1:], pair_mask, config.loss.coord_loss
        )
        _, rollout_next = model.rollout(z_sparse[:, 0], sparse_coords[:, 0], actions)
        sparse_rollout = coordinate_loss(rollout_next, sparse_coords[:, 1:], pair_mask, config.loss.coord_loss)
        rollout_sparse = torch.cat([direct_sparse[:, :1], rollout_next], dim=1)

    dense_loss = z.sum() * 0.0
    dense_latent = z.sum() * 0.0
    dense_coord = z.sum() * 0.0
    if config.data.use_dense_pseudo and batch["dense_coord_mask"].any():
        dense_coords = batch["dense_coords"]
        dense_mask = batch["dense_coord_mask"] & batch["frame_mask"]
        direct_dense = model.decode_coords(z)
        dense_coord = coordinate_loss(
            direct_dense,
            dense_coords,
            dense_mask,
            config.loss.coord_loss,
            weights=batch.get("label_weight"),
        )
        if dense_coords.shape[1] > 1:
            dense_base_actions = dense_coords[:, 1:] - dense_coords[:, :-1]
            dense_actions = _append_dt(
                dense_base_actions,
                _dense_dt(batch, dense_base_actions.shape[1]),
                config.model.use_time_delta,
            )
            dense_pair = dense_mask[:, 1:] & dense_mask[:, :-1]
            pred_dense_z = model.predict_next_latents(z, dense_coords, dense_actions)
            dense_latent = latent_dynamics_loss(pred_dense_z, z[:, 1:], dense_pair)
            pred_dense_coords = model.decode_coords(pred_dense_z)
            dense_transition = coordinate_loss(
                pred_dense_coords, dense_coords[:, 1:], dense_pair, config.loss.coord_loss
            )
            dense_loss = dense_coord + dense_transition + 0.1 * dense_latent
        else:
            dense_loss = dense_coord

    dense_weight = getattr(config.loss, "dense_pseudo_coord_weight", config.loss.dense_coord_weight)
    sparse_coord_total = sparse_direct_loss + sparse_transition + sparse_rollout
    total = (
        config.loss.latent_weight * (sparse_latent + 0.1 * dense_latent)
        + config.loss.sparse_coord_weight * sparse_coord_total
        + dense_weight * dense_loss
        + config.loss.smoothness_weight * sparse_smooth
    )
    scalars = {
        "loss": float(total.detach().cpu()),
        "sparse_direct_loss": float(sparse_direct_loss.detach().cpu()),
        "sparse_transition_loss": float(sparse_transition.detach().cpu()),
        "sparse_rollout_loss": float(sparse_rollout.detach().cpu()),
        "sparse_latent_loss": float(sparse_latent.detach().cpu()),
        "dense_loss": float(dense_loss.detach().cpu()),
        "dense_latent_loss": float(dense_latent.detach().cpu()),
        "dense_coord_loss": float(dense_coord.detach().cpu()),
        "smoothness_loss": float(sparse_smooth.detach().cpu()),
    }
    outputs = {
        "sparse_pred": rollout_sparse.detach(),
        "sparse_target": sparse_coords.detach(),
        "sparse_mask": sparse_mask.detach(),
    }
    return total, scalars, outputs


def save_checkpoint(
    path: Path,
    config: SurgWMBenchConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    mode: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "mode": mode,
        "epoch": epoch,
        "config": dataclass_to_dict(config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if isinstance(model, MaskedVisualAutoencoder):
        state["mae"] = model.state_dict()
    elif isinstance(model, MWMSurgWMBenchModel):
        state["mae"] = model.visual.state_dict()
    torch.save(state, path)


def load_pretrained_encoder(model: MWMSurgWMBenchModel, checkpoint: str | Path) -> None:
    state = torch.load(Path(checkpoint).expanduser(), map_location="cpu")
    if "mae" in state:
        model.visual.load_state_dict(state["mae"], strict=False)
        return
    if "model" in state:
        visual_state = {
            key.removeprefix("visual."): value
            for key, value in state["model"].items()
            if key.startswith("visual.")
        }
        if visual_state:
            model.visual.load_state_dict(visual_state, strict=False)
            return
    model.visual.load_state_dict(state, strict=False)


def train_pretrain_mae(args: argparse.Namespace, config: SurgWMBenchConfig) -> Path:
    seed_everything(config.train.seed)
    device = get_device()
    dataset = build_ssl_dataset(args, config)
    loader = make_loader(config, dataset, shuffle=True)
    model = build_mae(config).to(device)
    train_model = torch.compile(model) if config.model.compile else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    scaler = make_grad_scaler(device, config.train.precision)
    output_dir = ensure_dir(args.output_dir or config.train.output_dir)
    ckpt_path = output_dir / "mwm_mae_surgwmbench.pt"

    start_epoch = 0
    if getattr(args, "resume", False) and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["mae"] if "mae" in state else state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state.get("epoch", 0))
        print(f"resumed_from={ckpt_path} start_epoch={start_epoch}")

    for epoch in range(start_epoch + 1, config.train.epochs + 1):
        model.train()
        start = time.time()
        running = []
        for step, batch in enumerate(loader, start=1):
            batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, config.train.precision):
                images = batch.get("images")
                if images is None:
                    frames = batch["frames"]
                    images = frames.reshape(frames.shape[0] * frames.shape[1], *frames.shape[2:])
                out = train_model(images, mask_ratio=config.model.mask_ratio)
                loss = config.loss.recon_weight * out["loss"]
            scaler.scale(loss).backward()
            if config.train.grad_clip_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            running.append(float(loss.detach().cpu()))
            if step % config.train.log_every == 0:
                print(f"epoch={epoch} step={step} mae_loss={sum(running) / len(running):.6f}")
        save_checkpoint(ckpt_path, config, model, optimizer, epoch, "pretrain_mae")
        print(f"epoch={epoch} mae_loss={sum(running) / max(1, len(running)):.6f} time={time.time() - start:.1f}s")
    return ckpt_path


@torch.inference_mode()
def evaluate_loss(
    model: MWMSurgWMBenchModel,
    loader: DataLoader,
    config: SurgWMBenchConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses = []
    metrics = []
    for batch in loader:
        batch = move_to_device(batch, device)
        loss, scalars, outputs = supervised_losses(model, batch, config)
        losses.append({**scalars, "loss": float(loss.detach().cpu())})
        pixel_scale = batch["image_size"] if config.eval.report_pixel_metrics else None
        metrics.append(
            trajectory_metrics(
                outputs["sparse_pred"],
                outputs["sparse_target"],
                outputs["sparse_mask"],
                horizons=config.eval.horizons,
                prefix="sparse_",
                pixel_scale=pixel_scale,
            )
        )
    return {**average_metric_dicts(losses), **average_metric_dicts(metrics)}


def train_dynamics(args: argparse.Namespace, config: SurgWMBenchConfig) -> Path:
    seed_everything(config.train.seed)
    device = get_device()
    data_root = _data_root(args, config)
    train_manifest = args.train_manifest or args.manifest or config.data.train_manifest
    val_manifest = args.val_manifest or config.data.val_manifest
    dense_mode = args.mode == "train_dynamics_dense_aux" or bool(args.use_dense_pseudo) or bool(config.data.use_dense_pseudo)
    config.data.use_dense_pseudo = dense_mode
    if args.interpolation_method:
        config.data.interpolation_method = args.interpolation_method
    train_dataset = make_dataset(
        config,
        train_manifest,
        data_root,
        "train_dynamics_dense_aux" if dense_mode else "train_dynamics_sparse",
    )
    train_loader = make_loader(config, train_dataset, shuffle=True)
    val_loader = None
    if val_manifest:
        val_dataset = make_dataset(config, val_manifest, data_root, "eval")
        val_loader = make_loader(config, val_dataset, shuffle=False)

    model = build_model(config).to(device)
    if args.pretrained_encoder:
        load_pretrained_encoder(model, args.pretrained_encoder)
    if config.train.freeze_encoder:
        for param in model.visual.parameters():
            param.requires_grad = False

    train_model = torch.compile(model) if config.model.compile else model
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )
    scaler = make_grad_scaler(device, config.train.precision)
    output_dir = ensure_dir(args.output_dir or config.train.output_dir)
    ckpt_path = output_dir / "mwm_surgwmbench.pt"

    for epoch in range(1, config.train.epochs + 1):
        model.train()
        start = time.time()
        running = []
        for step, batch in enumerate(train_loader, start=1):
            batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, config.train.precision):
                loss, scalars, _ = supervised_losses(train_model, batch, config)
            scaler.scale(loss).backward()
            if config.train.grad_clip_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            running.append(scalars)
            if step % config.train.log_every == 0:
                mean = average_metric_dicts(running)
                print(
                    f"epoch={epoch} step={step} loss={mean.get('loss', 0.0):.6f} "
                    f"sparse_rollout={mean.get('sparse_rollout_loss', 0.0):.6f}"
                )
        save_checkpoint(ckpt_path, config, model, optimizer, epoch, "train_dynamics")
        mean = average_metric_dicts(running)
        print(f"epoch={epoch} loss={mean.get('loss', 0.0):.6f} time={time.time() - start:.1f}s")
        if val_loader is not None:
            val = evaluate_loss(model, val_loader, config, device)
            print(
                f"epoch={epoch} val_sparse_ade={val.get('sparse_ade', float('nan')):.6f} "
                f"val_sparse_fde={val.get('sparse_fde', float('nan')):.6f}"
            )
    return ckpt_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["pretrain_mae", "train_dynamics_sparse", "train_dynamics_dense_aux", "train_dynamics"],
        required=True,
    )
    parser.add_argument("--dataset-root", "--data-root", dest="dataset_root", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--train-manifest", default=None)
    parser.add_argument("--val-manifest", default=None)
    parser.add_argument("--config", default="configs/surgwmbench_mwm.yaml")
    parser.add_argument("--interpolation-method", default=None)
    parser.add_argument("--use-dense-pseudo", action="store_true")
    parser.add_argument("--pretrained-encoder", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--ssl-source", choices=["raw_videos", "clip_frames"], default=None)
    parser.add_argument("--ssl-backend", default=None)
    parser.add_argument("--source-video-manifest", default=None)
    parser.add_argument("--clip-length", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--max-clips-per-video", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Continue MAE pretraining from existing checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.mode == "pretrain_mae":
        path = train_pretrain_mae(args, config)
    else:
        path = train_dynamics(args, config)
    print(f"saved_checkpoint={path}")


if __name__ == "__main__":
    main()
