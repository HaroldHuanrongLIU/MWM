"""SurgWMBench manifest dataset and variable-length collation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

DatasetMode = Literal["pretrain_mae", "train_dynamics", "eval"]


def _read_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(path).expanduser()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if manifest_path.suffix.lower() == ".jsonl":
        entries = []
        for line_no, line in enumerate(manifest_path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {manifest_path}:{line_no}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Manifest entry at {manifest_path}:{line_no} must be an object.")
            entries.append(item)
        return entries

    raw = json.loads(manifest_path.read_text())
    if isinstance(raw, dict) and "entries" in raw:
        raw = raw["entries"]
    if not isinstance(raw, list):
        raise ValueError(f"JSON manifest must be a list or contain an 'entries' list: {manifest_path}")
    if not all(isinstance(item, dict) for item in raw):
        raise ValueError(f"Every manifest entry must be an object: {manifest_path}")
    return list(raw)


def _resolve_path(data_root: Path, frame_path: str) -> Path:
    path = Path(frame_path).expanduser()
    return path if path.is_absolute() else data_root / path


def _pil_to_tensor(path: Path, image_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    if not path.exists():
        raise FileNotFoundError(f"Frame image not found: {path}")
    with Image.open(path) as img:
        img = img.convert("RGB")
        original_size = img.size
        if image_size > 0 and img.size != (image_size, image_size):
            img = img.resize((image_size, image_size), Image.BILINEAR)
        array = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor, original_size


def _as_float_coords(value: Any, field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[-1] != 2:
        raise ValueError(f"'{field_name}' must have shape [N, 2], got {array.shape}.")
    return array


class SurgWMBenchDataset(Dataset):
    """Load raw-video and IMP clip manifests for SurgWMBench.

    In `pretrain_mae` mode the dataset is frame-level: every frame path becomes
    one sample. In supervised/evaluation modes each sample is a variable-length
    clip, with sparse anchor labels kept separate from optional dense pseudo
    labels.
    """

    def __init__(
        self,
        manifest: str | Path,
        data_root: str | Path,
        mode: DatasetMode,
        image_size: int = 224,
        coordinate_normalization: str = "image_size",
        image_width: int | None = None,
        image_height: int | None = None,
        use_dense_pseudo: bool = False,
        max_frames_per_clip: int | None = None,
        num_sparse_anchors: int = 20,
        split: str | None = None,
    ) -> None:
        self.manifest_path = Path(manifest).expanduser()
        self.data_root = Path(data_root).expanduser()
        self.mode = mode
        self.image_size = image_size
        self.coordinate_normalization = coordinate_normalization
        self.image_width = image_width
        self.image_height = image_height
        self.use_dense_pseudo = use_dense_pseudo
        self.max_frames_per_clip = max_frames_per_clip
        self.num_sparse_anchors = num_sparse_anchors

        entries = _read_manifest(self.manifest_path)
        if split is not None:
            entries = [entry for entry in entries if entry.get("split") == split]
        if not entries:
            raise ValueError(f"No usable entries found in manifest: {self.manifest_path}")
        self.entries = entries

        self._frame_samples: list[tuple[int, int, Path, str]] = []
        if mode == "pretrain_mae":
            for entry_idx, entry in enumerate(self.entries):
                frame_paths = self._frame_paths(entry)
                identifier = str(entry.get("video_id") or entry.get("clip_id") or entry_idx)
                for frame_idx, frame_path in enumerate(frame_paths):
                    self._frame_samples.append(
                        (entry_idx, frame_idx, _resolve_path(self.data_root, frame_path), identifier)
                    )
            if not self._frame_samples:
                raise ValueError(f"Pretraining manifest contains no frame paths: {self.manifest_path}")
        else:
            for entry_idx, entry in enumerate(self.entries):
                self._validate_imp_entry(entry, entry_idx)

    def __len__(self) -> int:
        return len(self._frame_samples) if self.mode == "pretrain_mae" else len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.mode == "pretrain_mae":
            _, frame_idx, path, identifier = self._frame_samples[index]
            frame, original_size = _pil_to_tensor(path, self.image_size)
            return {
                "frame": frame,
                "frame_index": torch.tensor(frame_idx, dtype=torch.long),
                "sample_id": identifier,
                "image_size": torch.tensor(original_size, dtype=torch.long),
            }
        return self._load_imp_clip(index)

    @staticmethod
    def _frame_paths(entry: dict[str, Any]) -> list[str]:
        frame_paths = entry.get("frame_paths")
        if not isinstance(frame_paths, list) or not frame_paths:
            raise ValueError("Each manifest entry must contain a non-empty 'frame_paths' list.")
        if not all(isinstance(path, str) for path in frame_paths):
            raise ValueError("'frame_paths' must be a list of strings.")
        return frame_paths

    def _validate_imp_entry(self, entry: dict[str, Any], entry_idx: int) -> None:
        self._frame_paths(entry)
        if "sampled_indices" not in entry:
            raise ValueError(f"IMP entry {entry_idx} is missing required 'sampled_indices'.")
        if "human_labeled_coordinates" not in entry:
            raise ValueError(f"IMP entry {entry_idx} is missing required 'human_labeled_coordinates'.")
        sampled = np.asarray(entry["sampled_indices"], dtype=np.int64)
        human = _as_float_coords(entry["human_labeled_coordinates"], "human_labeled_coordinates")
        if sampled.ndim != 1:
            raise ValueError(f"'sampled_indices' must be one-dimensional for entry {entry_idx}.")
        if len(sampled) != len(human):
            raise ValueError(
                f"Entry {entry_idx} has {len(sampled)} sampled indices but {len(human)} human labels."
            )
        if len(sampled) == 0:
            raise ValueError(f"Entry {entry_idx} has no sparse anchors.")
        if self.num_sparse_anchors and len(sampled) != self.num_sparse_anchors:
            # Robustness is more useful than rejecting non-standard toy/test clips.
            pass

    def _load_imp_clip(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        frame_paths = self._frame_paths(entry)
        num_frames = int(entry.get("num_frames") or len(frame_paths))
        if num_frames <= 0:
            raise ValueError(f"Entry {index} has invalid num_frames={num_frames}.")
        if len(frame_paths) < num_frames:
            raise ValueError(
                f"Entry {index} declares num_frames={num_frames} but only has "
                f"{len(frame_paths)} frame_paths."
            )

        sampled_indices = np.asarray(entry["sampled_indices"], dtype=np.int64)
        sparse_coords = _as_float_coords(entry["human_labeled_coordinates"], "human_labeled_coordinates")
        pseudo_coords = None
        if "pseudo_coordinates" in entry and entry["pseudo_coordinates"] is not None:
            pseudo_coords = _as_float_coords(entry["pseudo_coordinates"], "pseudo_coordinates")
            if len(pseudo_coords) < num_frames:
                raise ValueError(
                    f"Entry {index} pseudo_coordinates has length {len(pseudo_coords)} "
                    f"but num_frames is {num_frames}."
                )

        selected_indices = self._select_frame_indices(num_frames, sampled_indices, pseudo_coords is not None)
        frames: list[torch.Tensor] = []
        original_size: tuple[int, int] | None = None
        for frame_idx in selected_indices:
            if frame_idx < 0 or frame_idx >= len(frame_paths):
                raise ValueError(f"Entry {index} references frame index {frame_idx}, out of range.")
            tensor, size = _pil_to_tensor(_resolve_path(self.data_root, frame_paths[frame_idx]), self.image_size)
            frames.append(tensor)
            original_size = original_size or size
        assert original_size is not None

        width = self.image_width or original_size[0]
        height = self.image_height or original_size[1]
        sparse_coords_norm = self._normalize_coords(sparse_coords, width, height)

        index_to_position = {frame_idx: pos for pos, frame_idx in enumerate(selected_indices)}
        sparse_positions = np.asarray([index_to_position.get(int(idx), -1) for idx in sampled_indices], dtype=np.int64)
        if np.any(sparse_positions < 0):
            raise RuntimeError("Sparse anchors were not included in selected frame indices.")

        dense_coords = np.zeros((len(selected_indices), 2), dtype=np.float32)
        dense_mask = np.zeros((len(selected_indices),), dtype=bool)
        if pseudo_coords is not None:
            selected_pseudo = pseudo_coords[np.asarray(selected_indices, dtype=np.int64)]
            dense_coords = self._normalize_coords(selected_pseudo, width, height)
            dense_mask[:] = True

        clip_id = str(entry.get("clip_id") or entry.get("video_id") or index)
        return {
            "frames": torch.stack(frames, dim=0),
            "frame_indices": torch.tensor(selected_indices, dtype=torch.long),
            "sparse_indices": torch.from_numpy(sampled_indices.astype(np.int64)),
            "sparse_positions": torch.from_numpy(sparse_positions),
            "sparse_coords": torch.from_numpy(sparse_coords_norm),
            "sparse_anchor_mask": torch.ones(len(sampled_indices), dtype=torch.bool),
            "dense_coords": torch.from_numpy(dense_coords),
            "dense_coord_mask": torch.from_numpy(dense_mask),
            "frame_mask": torch.ones(len(selected_indices), dtype=torch.bool),
            "num_frames": torch.tensor(num_frames, dtype=torch.long),
            "image_size": torch.tensor([width, height], dtype=torch.long),
            "clip_id": clip_id,
            "source_video_id": str(entry.get("source_video_id") or ""),
            "interpolation_method": str(entry.get("interpolation_method") or ""),
        }

    def _select_frame_indices(
        self, num_frames: int, sampled_indices: np.ndarray, has_pseudo: bool
    ) -> list[int]:
        sparse = sorted({int(idx) for idx in sampled_indices.tolist()})
        if any(idx < 0 or idx >= num_frames for idx in sparse):
            raise ValueError("'sampled_indices' contains an index outside [0, num_frames).")
        if not (self.use_dense_pseudo and has_pseudo):
            return sparse

        if self.max_frames_per_clip is None or num_frames <= self.max_frames_per_clip:
            return list(range(num_frames))

        budget = max(self.max_frames_per_clip, len(sparse))
        remaining = max(0, budget - len(sparse))
        if remaining == 0:
            return sparse
        uniform = np.linspace(0, num_frames - 1, num=remaining, dtype=np.int64).tolist()
        return sorted(set(sparse).union(int(idx) for idx in uniform))

    def _normalize_coords(self, coords: np.ndarray, width: int, height: int) -> np.ndarray:
        if self.coordinate_normalization == "none":
            return coords.astype(np.float32)
        if self.coordinate_normalization != "image_size":
            raise ValueError(
                "Unsupported coordinate_normalization="
                f"{self.coordinate_normalization!r}; expected 'image_size' or 'none'."
            )
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions for normalization: width={width}, height={height}.")
        scale = np.asarray([width, height], dtype=np.float32)
        return (coords.astype(np.float32) / scale).clip(0.0, 1.0)


def surgwmbench_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate pretraining frames or variable-length IMP clips."""

    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    if "frame" in batch[0]:
        return {
            "images": torch.stack([item["frame"] for item in batch], dim=0),
            "frame_indices": torch.stack([item["frame_index"] for item in batch], dim=0),
            "image_size": torch.stack([item["image_size"] for item in batch], dim=0),
            "sample_ids": [item["sample_id"] for item in batch],
        }

    batch_size = len(batch)
    max_frames = max(int(item["frames"].shape[0]) for item in batch)
    max_sparse = max(int(item["sparse_coords"].shape[0]) for item in batch)
    channels, height, width = batch[0]["frames"].shape[1:]

    frames = torch.zeros(batch_size, max_frames, channels, height, width, dtype=torch.float32)
    frame_mask = torch.zeros(batch_size, max_frames, dtype=torch.bool)
    frame_indices = torch.full((batch_size, max_frames), -1, dtype=torch.long)
    dense_coords = torch.zeros(batch_size, max_frames, 2, dtype=torch.float32)
    dense_coord_mask = torch.zeros(batch_size, max_frames, dtype=torch.bool)

    sparse_coords = torch.zeros(batch_size, max_sparse, 2, dtype=torch.float32)
    sparse_indices = torch.full((batch_size, max_sparse), -1, dtype=torch.long)
    sparse_positions = torch.full((batch_size, max_sparse), -1, dtype=torch.long)
    sparse_anchor_mask = torch.zeros(batch_size, max_sparse, dtype=torch.bool)

    image_size = torch.zeros(batch_size, 2, dtype=torch.long)
    num_frames = torch.zeros(batch_size, dtype=torch.long)
    clip_ids: list[str] = []
    source_video_ids: list[str] = []
    interpolation_methods: list[str] = []

    for row, item in enumerate(batch):
        t = int(item["frames"].shape[0])
        a = int(item["sparse_coords"].shape[0])
        frames[row, :t] = item["frames"]
        frame_mask[row, :t] = item["frame_mask"]
        frame_indices[row, :t] = item["frame_indices"]
        dense_coords[row, :t] = item["dense_coords"]
        dense_coord_mask[row, :t] = item["dense_coord_mask"]
        sparse_coords[row, :a] = item["sparse_coords"]
        sparse_indices[row, :a] = item["sparse_indices"]
        sparse_positions[row, :a] = item["sparse_positions"]
        sparse_anchor_mask[row, :a] = item["sparse_anchor_mask"]
        image_size[row] = item["image_size"]
        num_frames[row] = item["num_frames"]
        clip_ids.append(item["clip_id"])
        source_video_ids.append(item["source_video_id"])
        interpolation_methods.append(item["interpolation_method"])

    return {
        "frames": frames,
        "frame_mask": frame_mask,
        "frame_indices": frame_indices,
        "dense_coords": dense_coords,
        "dense_coord_mask": dense_coord_mask,
        "sparse_coords": sparse_coords,
        "sparse_indices": sparse_indices,
        "sparse_positions": sparse_positions,
        "sparse_anchor_mask": sparse_anchor_mask,
        "image_size": image_size,
        "num_frames": num_frames,
        "clip_ids": clip_ids,
        "source_video_ids": source_video_ids,
        "interpolation_methods": interpolation_methods,
    }
