"""Typed YAML configuration for the PyTorch SurgWMBench baseline."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    image_size: int = 224
    patch_size: int = 16
    latent_dim: int = 512
    hidden_dim: int = 512
    coord_dim: int = 2
    coord_embed_dim: int = 64
    action_embed_dim: int = 64
    action_type: str = "continuous_delta"
    dynamics_type: str = "gru"
    mask_ratio: float = 0.5
    encoder_depth: int = 4
    decoder_depth: int = 2
    num_heads: int = 8
    decoder_num_heads: int = 8
    conv_stem_channels: list[int] = field(default_factory=lambda: [64, 128])
    compile: bool = False


@dataclass
class DataConfig:
    coordinate_normalization: str = "image_size"
    use_dense_pseudo: bool = False
    max_frames_per_clip: int | None = None
    num_sparse_anchors: int = 20
    image_width: int | None = None
    image_height: int | None = None


@dataclass
class TrainConfig:
    batch_size: int = 16
    num_workers: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 100
    precision: str = "amp"
    seed: int = 42
    grad_clip_norm: float = 100.0
    output_dir: str = "checkpoints"
    log_every: int = 20
    freeze_encoder: bool = False


@dataclass
class LossConfig:
    recon_weight: float = 1.0
    latent_weight: float = 1.0
    sparse_coord_weight: float = 10.0
    dense_coord_weight: float = 1.0
    smoothness_weight: float = 0.1
    coord_loss: str = "smooth_l1"


@dataclass
class EvalConfig:
    report_pixel_metrics: bool = True
    horizons: list[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])


@dataclass
class SurgWMBenchConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def _merge_dataclass(instance: Any, values: dict[str, Any], path: str = "") -> Any:
    valid = {f.name for f in fields(instance)}
    for key, value in values.items():
        if key not in valid:
            raise ValueError(f"Unknown config key '{path + key}'.")
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge_dataclass(current, value, path=f"{path}{key}.")
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path | None = None) -> SurgWMBenchConfig:
    """Load a SurgWMBench YAML config into dataclasses.

    Missing keys use conservative defaults. Unknown keys raise so typos are not
    silently ignored.
    """

    config = SurgWMBenchConfig()
    if path is None:
        return config
    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return _merge_dataclass(config, raw)


def dataclass_to_dict(value: Any) -> Any:
    """Convert nested dataclasses to JSON/YAML-serializable containers."""

    if is_dataclass(value):
        return {f.name: dataclass_to_dict(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, list):
        return [dataclass_to_dict(v) for v in value]
    if isinstance(value, tuple):
        return [dataclass_to_dict(v) for v in value]
    if isinstance(value, dict):
        return {k: dataclass_to_dict(v) for k, v in value.items()}
    return value
