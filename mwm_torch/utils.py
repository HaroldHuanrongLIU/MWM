"""Runtime helpers shared by training and evaluation scripts."""

from __future__ import annotations

import contextlib
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Set deterministic seeds for Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    """Return CUDA, MPS, or CPU in that preference order."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@contextlib.contextmanager
def autocast_context(device: torch.device, precision: str) -> Iterator[None]:
    """Use current torch.amp autocast APIs when requested and supported."""

    enabled = precision == "amp" and device.type in {"cuda", "cpu"}
    if not enabled:
        yield
        return
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    with torch.amp.autocast(device_type=device.type, dtype=dtype):
        yield


def make_grad_scaler(device: torch.device, precision: str) -> torch.amp.GradScaler:
    """Create a GradScaler using the non-deprecated torch.amp namespace."""

    enabled = precision == "amp" and device.type == "cuda"
    scaler_device = device.type if device.type in {"cuda", "cpu"} else "cuda"
    return torch.amp.GradScaler(scaler_device, enabled=enabled)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""

    directory = Path(path).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def move_to_device(batch: dict, device: torch.device) -> dict:
    """Move tensors in a nested batch dictionary to a device."""

    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved
