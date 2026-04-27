"""Training entrypoint for the PyTorch SurgWMBench MWM baseline."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .config import SurgWMBenchConfig, dataclass_to_dict, load_config
from .data import SurgWMBenchDataset, surgwmbench_collate
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
) -> SurgWMBenchDataset:
    return SurgWMBenchDataset(
        manifest=manifest,
        data_root=data_root,
        mode=mode,  # type: ignore[arg-type]
        image_size=config.model.image_size,
        coordinate_normalization=config.data.coordinate_normalization,
        image_width=config.data.image_width,
        image_height=config.data.image_height,
        use_dense_pseudo=config.data.use_dense_pseudo,
        max_frames_per_clip=config.data.max_frames_per_clip,
        num_sparse_anchors=config.data.num_sparse_anchors,
    )


def make_loader(config: SurgWMBenchConfig, dataset: SurgWMBenchDataset, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=shuffle,
        num_workers=config.train.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.train.num_workers > 0,
        collate_fn=surgwmbench_collate,
    )


def gather_by_position(sequence: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather `[B, A, D]` values from `[B, T, D]` using padded positions."""

    valid = positions >= 0
    safe = positions.clamp_min(0)
    index = safe.unsqueeze(-1).expand(-1, -1, sequence.shape[-1])
    gathered = torch.gather(sequence, dim=1, index=index)
    return gathered, valid


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
        actions = sparse_coords[:, 1:] - sparse_coords[:, :-1]
        pair_mask = sparse_mask[:, 1:] & sparse_mask[:, :-1]
        pred_next_z = model.predict_next_latents(z_sparse[:, :-1], sparse_coords[:, :-1], actions)
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
        dense_coord = coordinate_loss(direct_dense, dense_coords, dense_mask, config.loss.coord_loss)
        if dense_coords.shape[1] > 1:
            dense_actions = dense_coords[:, 1:] - dense_coords[:, :-1]
            dense_pair = dense_mask[:, 1:] & dense_mask[:, :-1]
            pred_dense_z = model.predict_next_latents(z[:, :-1], dense_coords[:, :-1], dense_actions)
            dense_latent = latent_dynamics_loss(pred_dense_z, z[:, 1:], dense_pair)
            pred_dense_coords = model.decode_coords(pred_dense_z)
            dense_transition = coordinate_loss(
                pred_dense_coords, dense_coords[:, 1:], dense_pair, config.loss.coord_loss
            )
            dense_loss = dense_coord + dense_transition + 0.1 * dense_latent
        else:
            dense_loss = dense_coord

    sparse_coord_total = sparse_direct_loss + sparse_transition + sparse_rollout
    total = (
        config.loss.latent_weight * (sparse_latent + 0.1 * dense_latent)
        + config.loss.sparse_coord_weight * sparse_coord_total
        + config.loss.dense_coord_weight * dense_loss
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
    dataset = make_dataset(config, args.manifest, args.data_root, "pretrain_mae")
    loader = make_loader(config, dataset, shuffle=True)
    model = build_mae(config).to(device)
    train_model = torch.compile(model) if config.model.compile else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    scaler = make_grad_scaler(device, config.train.precision)
    output_dir = ensure_dir(args.output_dir or config.train.output_dir)
    ckpt_path = output_dir / "mwm_mae.pt"

    for epoch in range(1, config.train.epochs + 1):
        model.train()
        start = time.time()
        running = []
        for step, batch in enumerate(loader, start=1):
            batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, config.train.precision):
                out = train_model(batch["images"], mask_ratio=config.model.mask_ratio)
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
    train_dataset = make_dataset(config, args.manifest, args.data_root, "train_dynamics")
    train_loader = make_loader(config, train_dataset, shuffle=True)
    val_loader = None
    if args.val_manifest:
        val_dataset = make_dataset(config, args.val_manifest, args.data_root, "eval")
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
    parser.add_argument("--mode", choices=["pretrain_mae", "train_dynamics"], required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--val-manifest", default=None)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--config", default="configs/surgwmbench_mwm.yaml")
    parser.add_argument("--pretrained-encoder", default=None)
    parser.add_argument("--output-dir", default=None)
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
