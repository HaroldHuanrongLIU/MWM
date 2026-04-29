"""SurgWMBench final-layout datasets and collation."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

FINAL_DATASET_VERSION = "SurgWMBench"
LEGACY_DATASET_VERSION = "SurgWMBenchv2"
SUPPORTED_INTERPOLATION_METHODS = {"linear", "pchip", "akima", "cubic_spline"}
SOURCE_ENCODING = {"unlabeled": 0, "human": 1, "interpolated": 2}
FrameSampling = Literal["sparse_anchors", "dense", "window", "all"]


@dataclass(frozen=True)
class _DenseTrajectory:
    coords_px: np.ndarray
    coords_norm: np.ndarray
    source_labels: list[str]
    source_ids: np.ndarray
    label_weights: np.ndarray
    confidence: np.ndarray


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(path).expanduser()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if manifest_path.suffix.lower() == ".jsonl":
        entries: list[dict[str, Any]] = []
        for line_no, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
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

    raw = _read_json(manifest_path)
    if isinstance(raw, dict):
        raw = raw.get("entries", raw.get("clips", raw.get("items", raw)))
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise ValueError(f"Manifest must be a JSON list or object with an entries/clips/items list: {manifest_path}")
    return list(raw)


def _resolve_path(dataset_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else dataset_root / path


def _resolve_manifest_path(dataset_root: Path, manifest: str | Path) -> Path:
    path = Path(manifest).expanduser()
    if path.is_absolute() or path.exists():
        return path
    return dataset_root / path


def _resolve_frame_path(
    dataset_root: Path,
    frames_dir: Path,
    local_frame_idx: int,
    num_frames: int,
    strict: bool,
    frame_record: dict[str, Any] | None = None,
) -> Path:
    if local_frame_idx < 0 or local_frame_idx >= num_frames:
        raise ValueError(f"local_frame_idx={local_frame_idx} is outside [0, {num_frames}).")
    if isinstance(frame_record, dict) and frame_record.get("frame_path"):
        recorded = _resolve_path(dataset_root, frame_record["frame_path"])
        if recorded.exists() or strict:
            return recorded
    for suffix in (".jpg", ".png", ".jpeg"):
        candidate = frames_dir / f"{local_frame_idx:06d}{suffix}"
        if candidate.exists():
            return candidate
    canonical = frames_dir / f"{local_frame_idx:06d}.jpg"
    if strict:
        return canonical
    candidates = sorted(
        path
        for suffix in ("*.jpg", "*.png", "*.jpeg")
        for path in frames_dir.glob(suffix)
    )
    if local_frame_idx >= len(candidates):
        raise FileNotFoundError(f"Frame {local_frame_idx} not found under {frames_dir}.")
    return candidates[local_frame_idx]


def _target_hw(image_size: int | tuple[int, int]) -> tuple[int, int] | None:
    if isinstance(image_size, int):
        return None if image_size <= 0 else (image_size, image_size)
    if len(image_size) != 2:
        raise ValueError(f"image_size tuple must have length 2, got {image_size!r}.")
    height, width = int(image_size[0]), int(image_size[1])
    if height <= 0 or width <= 0:
        return None
    return height, width


def _pil_to_tensor(path: Path, image_size: int | tuple[int, int]) -> tuple[torch.Tensor, tuple[int, int]]:
    if not path.exists():
        raise FileNotFoundError(f"Frame image not found: {path}")
    with Image.open(path) as img:
        img = img.convert("RGB")
        original_width, original_height = img.size
        target = _target_hw(image_size)
        if target is not None:
            target_h, target_w = target
            if img.size != (target_w, target_h):
                img = img.resize((target_w, target_h), Image.BILINEAR)
        array = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor, (original_height, original_width)


def _coord_from_value(value: Any, field_name: str) -> np.ndarray:
    if isinstance(value, dict):
        if "coord_px" in value:
            value = value["coord_px"]
        elif "coordinate_px" in value:
            value = value["coordinate_px"]
        elif "coord" in value:
            value = value["coord"]
        elif "coordinate" in value:
            value = value["coordinate"]
        elif "point" in value:
            value = value["point"]
        elif "x" in value and "y" in value:
            value = [value["x"], value["y"]]
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (2,):
        raise ValueError(f"{field_name} must contain a 2D coordinate, got shape {array.shape}.")
    return array


def _coords_array(value: Any, field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{field_name} must have shape [N, 2], got {array.shape}.")
    return array


def _normalize_coords(coords_px: np.ndarray, image_size_hw: tuple[int, int]) -> np.ndarray:
    height, width = image_size_hw
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size for coordinate normalization: {(height, width)}.")
    return coords_px.astype(np.float32) / np.asarray([width, height], dtype=np.float32)


def _denormalize_coords(coords_norm: np.ndarray, image_size_hw: tuple[int, int]) -> np.ndarray:
    height, width = image_size_hw
    return coords_norm.astype(np.float32) * np.asarray([width, height], dtype=np.float32)


def _extract_image_size(annotation: dict[str, Any], row: dict[str, Any]) -> tuple[int, int] | None:
    for source in (annotation, row):
        if "image_height" in source and "image_width" in source:
            return int(source["image_height"]), int(source["image_width"])
        if "height" in source and "width" in source:
            return int(source["height"]), int(source["width"])
        image_size = source.get("image_size") or source.get("image_size_original")
        if isinstance(image_size, dict):
            if "height" in image_size and "width" in image_size:
                return int(image_size["height"]), int(image_size["width"])
        elif isinstance(image_size, (list, tuple)) and len(image_size) == 2:
            first, second = int(image_size[0]), int(image_size[1])
            # The final benchmark documents image sizes as height, width.
            return first, second
    metadata = annotation.get("metadata")
    if isinstance(metadata, dict):
        return _extract_image_size(metadata, {})
    return None


def _check_version(version: Any, *, allow_legacy_version: bool, context: str) -> None:
    if version == FINAL_DATASET_VERSION:
        return
    if version == LEGACY_DATASET_VERSION and allow_legacy_version:
        warnings.warn(
            f"{context} uses legacy dataset_version={LEGACY_DATASET_VERSION!r}; "
            f"final public datasets should use {FINAL_DATASET_VERSION!r}.",
            UserWarning,
            stacklevel=3,
        )
        return
    raise ValueError(
        f"{context} has dataset_version={version!r}; expected {FINAL_DATASET_VERSION!r}. "
        "Pass allow_legacy_version=True only for provisional legacy data."
    )


def _source_label(value: Any) -> str:
    label = str(value or "").lower()
    if label in {"human", "anchor", "human_anchor", "human_label"}:
        return "human"
    if label in {"unlabeled", "none", "null", "missing"}:
        return "unlabeled"
    return "interpolated"


class SurgWMBenchClipDataset(Dataset):
    """Load final public SurgWMBench clip samples from official manifests."""

    def __init__(
        self,
        dataset_root: str | Path,
        manifest: str | Path,
        interpolation_method: str | None = None,
        image_size: int | tuple[int, int] = 224,
        frame_sampling: FrameSampling = "sparse_anchors",
        max_frames: int | None = None,
        use_dense_pseudo: bool = False,
        return_images: bool = True,
        return_source_video_path: bool = True,
        normalize_coords: bool = True,
        cache_annotations: bool = True,
        strict: bool = True,
        allow_legacy_version: bool = False,
    ) -> None:
        if frame_sampling not in {"sparse_anchors", "dense", "window", "all"}:
            raise ValueError(f"Unsupported frame_sampling={frame_sampling!r}.")
        if interpolation_method is not None and interpolation_method not in SUPPORTED_INTERPOLATION_METHODS:
            raise ValueError(f"Unsupported interpolation_method={interpolation_method!r}.")
        self.dataset_root = Path(dataset_root).expanduser()
        self.manifest_path = _resolve_manifest_path(self.dataset_root, manifest)
        self.interpolation_method = interpolation_method
        self.image_size = image_size
        self.frame_sampling = frame_sampling
        self.max_frames = max_frames
        self.use_dense_pseudo = use_dense_pseudo
        self.return_images = return_images
        self.return_source_video_path = return_source_video_path
        self.normalize_coords = normalize_coords
        self.cache_annotations = cache_annotations
        self.strict = strict
        self.allow_legacy_version = allow_legacy_version

        self.entries = _read_manifest(self.manifest_path)
        if not self.entries:
            raise ValueError(f"No entries found in manifest: {self.manifest_path}")
        self._annotation_cache: dict[Path, dict[str, Any]] = {}
        self._interpolation_cache: dict[Path, Any] = {}
        for idx, row in enumerate(self.entries):
            version = row.get("dataset_version")
            if version is not None:
                _check_version(version, allow_legacy_version=allow_legacy_version, context=f"Manifest row {idx}")
            elif strict:
                raise ValueError(f"Manifest row {idx} is missing dataset_version.")
            for key in ("annotation_path", "frames_dir", "interpolation_files", "num_frames"):
                if key not in row and strict:
                    raise ValueError(f"Manifest row {idx} is missing required field {key!r}.")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.entries[index]
        annotation_path = _resolve_path(self.dataset_root, row["annotation_path"])
        annotation = self._load_annotation(annotation_path, index)
        human_px, human_norm_from_ann, human_local_indices = self._load_human_anchors(
            annotation, row, index
        )
        num_frames = int(row.get("num_frames") or annotation.get("num_frames") or 0)
        if num_frames <= 0:
            raise ValueError(f"Entry {index} has invalid num_frames={num_frames}.")
        if len(human_local_indices) != 20 and self.strict:
            raise ValueError(f"Entry {index} has {len(human_local_indices)} human anchors; expected exactly 20.")
        if np.any(human_local_indices < 0) or np.any(human_local_indices >= num_frames):
            raise ValueError(f"Entry {index} has human anchor local_frame_idx outside [0, num_frames).")

        selected_indices = self._select_indices(num_frames, human_local_indices)
        frames_dir = _resolve_path(self.dataset_root, row["frames_dir"])
        frame_records = self._frame_records_by_local_idx(annotation)
        frame_paths = [
            self._frame_path(frames_dir, int(frame_idx), num_frames, frame_records.get(int(frame_idx)))
            for frame_idx in selected_indices
        ]

        frames_tensor: torch.Tensor | None = None
        image_size_from_frames: tuple[int, int] | None = None
        if self.return_images:
            frame_tensors: list[torch.Tensor] = []
            for frame_path in frame_paths:
                tensor, original_hw = _pil_to_tensor(frame_path, self.image_size)
                frame_tensors.append(tensor)
                image_size_from_frames = image_size_from_frames or original_hw
            frames_tensor = torch.stack(frame_tensors, dim=0)

        image_size_original = _extract_image_size(annotation, row) or image_size_from_frames
        if image_size_original is None:
            if self.strict:
                raise ValueError(f"Entry {index} cannot determine original image size.")
            image_size_original = (1, 1)

        human_norm = human_norm_from_ann
        if human_norm is None or not self.normalize_coords:
            human_norm = _normalize_coords(human_px, image_size_original)

        method = self._select_interpolation_method(row)
        interpolation_path = self._interpolation_path(row, method, index)
        dense = self._load_dense_trajectory(
            interpolation_path, method, num_frames, image_size_original, human_local_indices
        )

        anchor_positions = {int(frame_idx): pos for pos, frame_idx in enumerate(human_local_indices.tolist())}
        selected_source_ids = dense.source_ids[selected_indices].copy()
        selected_source_labels = [dense.source_labels[int(frame_idx)] for frame_idx in selected_indices]
        selected_weights = dense.label_weights[selected_indices].copy()
        selected_confidence = dense.confidence[selected_indices].copy()
        if self.frame_sampling == "sparse_anchors":
            selected_coords_px = human_px.astype(np.float32)
            selected_coords_norm = human_norm.astype(np.float32)
            selected_source_ids = np.full(len(selected_indices), SOURCE_ENCODING["human"], dtype=np.int64)
            selected_source_labels = ["human"] * len(selected_indices)
            selected_weights = np.ones(len(selected_indices), dtype=np.float32)
            selected_confidence = np.ones(len(selected_indices), dtype=np.float32)
        else:
            selected_coords_px = dense.coords_px[selected_indices]
            selected_coords_norm = dense.coords_norm[selected_indices]

        sparse_positions = np.asarray(
            [idx if self.frame_sampling == "sparse_anchors" else selected_indices.index(int(local_idx))
             if int(local_idx) in set(selected_indices) else -1
             for idx, local_idx in enumerate(human_local_indices.tolist())],
            dtype=np.int64,
        )
        if self.frame_sampling != "sparse_anchors":
            selected_lookup = {int(frame_idx): pos for pos, frame_idx in enumerate(selected_indices)}
            sparse_positions = np.asarray(
                [selected_lookup.get(int(local_idx), -1) for local_idx in human_local_indices.tolist()],
                dtype=np.int64,
            )

        source_video_path = row.get("source_video_path")
        resolved_source_video_path = (
            str(_resolve_path(self.dataset_root, source_video_path))
            if source_video_path and self.return_source_video_path
            else ""
        )
        return {
            "patient_id": str(row.get("patient_id") or annotation.get("patient_id") or ""),
            "source_video_id": str(row.get("source_video_id") or annotation.get("source_video_id") or ""),
            "source_video_path": resolved_source_video_path,
            "trajectory_id": str(row.get("trajectory_id") or annotation.get("trajectory_id") or index),
            "difficulty": row.get("difficulty", annotation.get("difficulty")),
            "num_frames": int(num_frames),
            "image_size_original": tuple(int(v) for v in image_size_original),
            "sampled_indices": torch.from_numpy(human_local_indices.astype(np.int64)),
            "frames": frames_tensor,
            "frame_paths": [str(path) for path in frame_paths],
            "frame_indices": torch.tensor(selected_indices, dtype=torch.long),
            "human_anchor_coords_px": torch.from_numpy(human_px.astype(np.float32)),
            "human_anchor_coords_norm": torch.from_numpy(human_norm.astype(np.float32)),
            "human_anchor_local_indices": torch.from_numpy(human_local_indices.astype(np.int64)),
            "dense_coords_px": torch.from_numpy(dense.coords_px.astype(np.float32)),
            "dense_coords_norm": torch.from_numpy(dense.coords_norm.astype(np.float32)),
            "dense_coord_sources": torch.from_numpy(dense.source_ids.astype(np.int64)),
            "dense_coord_source_labels": dense.source_labels,
            "dense_label_weights": torch.from_numpy(dense.label_weights.astype(np.float32)),
            "dense_confidence": torch.from_numpy(dense.confidence.astype(np.float32)),
            "selected_coords_px": torch.from_numpy(selected_coords_px.astype(np.float32)),
            "selected_coords_norm": torch.from_numpy(selected_coords_norm.astype(np.float32)),
            "selected_coord_sources": torch.from_numpy(selected_source_ids.astype(np.int64)),
            "selected_coord_source_labels": selected_source_labels,
            "selected_label_weights": torch.from_numpy(selected_weights.astype(np.float32)),
            "selected_confidence": torch.from_numpy(selected_confidence.astype(np.float32)),
            "interpolation_method": method,
            "annotation_path": str(annotation_path),
            "interpolation_path": str(interpolation_path),
            "frame_sampling": self.frame_sampling,
            "sparse_positions": torch.from_numpy(sparse_positions),
            # Compatibility keys used by the provisional training path.
            "sparse_coords": torch.from_numpy(human_norm.astype(np.float32)),
            "sparse_indices": torch.from_numpy(human_local_indices.astype(np.int64)),
            "sparse_anchor_mask": torch.ones(len(human_local_indices), dtype=torch.bool),
            "dense_coords": torch.from_numpy(selected_coords_norm.astype(np.float32)),
            "dense_coord_mask": torch.from_numpy((selected_source_ids != SOURCE_ENCODING["unlabeled"]).astype(bool)),
            "frame_mask": torch.ones(len(selected_indices), dtype=torch.bool),
            "image_size": torch.tensor([image_size_original[1], image_size_original[0]], dtype=torch.long),
            "clip_id": str(row.get("trajectory_id") or annotation.get("trajectory_id") or index),
        }

    def _load_annotation(self, path: Path, index: int) -> dict[str, Any]:
        if self.cache_annotations and path in self._annotation_cache:
            return self._annotation_cache[path]
        if not path.exists():
            raise FileNotFoundError(f"Annotation file not found for entry {index}: {path}")
        annotation = _read_json(path)
        if not isinstance(annotation, dict):
            raise ValueError(f"Annotation must be a JSON object: {path}")
        version = annotation.get("dataset_version")
        if version is not None:
            _check_version(version, allow_legacy_version=self.allow_legacy_version, context=f"Annotation {path}")
        elif self.strict:
            raise ValueError(f"Annotation is missing dataset_version: {path}")
        if self.cache_annotations:
            self._annotation_cache[path] = annotation
        return annotation

    def _load_human_anchors(
        self, annotation: dict[str, Any], row: dict[str, Any], index: int
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        anchors = annotation.get("human_anchors")
        if not isinstance(anchors, list):
            raise ValueError(f"Entry {index} annotation is missing list field 'human_anchors'.")
        sampled_indices = row.get("sampled_indices")
        if sampled_indices is not None:
            sampled_array = np.asarray(sampled_indices, dtype=np.int64)
            if sampled_array.ndim != 1:
                raise ValueError(f"Entry {index} sampled_indices must be one-dimensional.")
        else:
            sampled_array = None

        coords_px: list[np.ndarray] = []
        coords_norm: list[np.ndarray] = []
        has_all_norm = True
        local_indices: list[int] = []
        for anchor_pos, anchor in enumerate(anchors):
            if not isinstance(anchor, dict):
                raise ValueError(f"Entry {index} human_anchors[{anchor_pos}] must be an object.")
            anchor_idx = int(anchor.get("anchor_idx", anchor_pos))
            if self.strict and anchor_idx != anchor_pos:
                raise ValueError(f"Entry {index} anchor_idx sequence is not 0..19 at position {anchor_pos}.")
            if "local_frame_idx" in anchor:
                local_idx = int(anchor["local_frame_idx"])
            elif sampled_array is not None and anchor_pos < len(sampled_array):
                local_idx = int(sampled_array[anchor_pos])
            else:
                raise ValueError(
                    f"Entry {index} human anchor {anchor_pos} is missing local_frame_idx. "
                    "old_frame_idx is not a dense local frame index and is intentionally ignored."
                )
            local_indices.append(local_idx)
            coords_px.append(_coord_from_value(anchor, f"human_anchors[{anchor_pos}]"))
            norm_value = anchor.get("coord_norm", anchor.get("coordinate_norm"))
            if norm_value is None:
                has_all_norm = False
            else:
                coords_norm.append(_coord_from_value(norm_value, f"human_anchors[{anchor_pos}].coord_norm"))

        local_array = np.asarray(local_indices, dtype=np.int64)
        if sampled_array is not None and self.strict and not np.array_equal(sampled_array, local_array):
            raise ValueError(
                f"Entry {index} manifest sampled_indices differ from human_anchors[].local_frame_idx."
            )
        norm_array = np.stack(coords_norm).astype(np.float32) if has_all_norm else None
        return np.stack(coords_px).astype(np.float32), norm_array, local_array

    def _select_indices(self, num_frames: int, human_local_indices: np.ndarray) -> list[int]:
        if self.frame_sampling == "sparse_anchors":
            return [int(idx) for idx in human_local_indices.tolist()]
        if self.frame_sampling in {"dense", "all"}:
            return list(range(num_frames))
        window = num_frames if self.max_frames is None else min(num_frames, int(self.max_frames))
        return list(range(max(0, window)))

    @staticmethod
    def _frame_records_by_local_idx(annotation: dict[str, Any]) -> dict[int, dict[str, Any]]:
        records = annotation.get("frames")
        if not isinstance(records, list):
            return {}
        result: dict[int, dict[str, Any]] = {}
        for fallback_idx, record in enumerate(records):
            if isinstance(record, dict):
                local_idx = int(record.get("local_frame_idx", fallback_idx))
                result[local_idx] = record
        return result

    def _frame_path(
        self,
        frames_dir: Path,
        local_frame_idx: int,
        num_frames: int,
        frame_record: dict[str, Any] | None = None,
    ) -> Path:
        return _resolve_frame_path(
            self.dataset_root,
            frames_dir,
            local_frame_idx,
            num_frames,
            self.strict,
            frame_record,
        )

    def _select_interpolation_method(self, row: dict[str, Any]) -> str:
        method = self.interpolation_method or row.get("default_interpolation_method") or "linear"
        method = str(method)
        if method not in SUPPORTED_INTERPOLATION_METHODS:
            raise ValueError(f"Unsupported interpolation method {method!r}.")
        return method

    def _interpolation_path(self, row: dict[str, Any], method: str, index: int) -> Path:
        files = row.get("interpolation_files")
        if not isinstance(files, dict):
            raise ValueError(f"Entry {index} interpolation_files must be an object keyed by method.")
        if method not in files:
            raise FileNotFoundError(f"Entry {index} has no interpolation file for method {method!r}.")
        path = _resolve_path(self.dataset_root, files[method])
        if not path.exists():
            raise FileNotFoundError(f"Interpolation file for method {method!r} not found: {path}")
        return path

    def _load_dense_trajectory(
        self,
        path: Path,
        method: str,
        num_frames: int,
        image_size_hw: tuple[int, int],
        human_local_indices: np.ndarray,
    ) -> _DenseTrajectory:
        if self.cache_annotations and path in self._interpolation_cache:
            raw = self._interpolation_cache[path]
        else:
            raw = _read_json(path)
            if self.cache_annotations:
                self._interpolation_cache[path] = raw
        entries = self._interpolation_entries(raw, path)
        if len(entries) != num_frames:
            raise ValueError(f"{path} has {len(entries)} interpolation entries; expected num_frames={num_frames}.")

        coords_px = np.zeros((num_frames, 2), dtype=np.float32)
        coords_norm = np.zeros((num_frames, 2), dtype=np.float32)
        source_labels: list[str] = []
        source_ids = np.zeros(num_frames, dtype=np.int64)
        label_weights = np.full(num_frames, 0.5, dtype=np.float32)
        confidence = np.ones(num_frames, dtype=np.float32)
        anchor_set = {int(idx) for idx in human_local_indices.tolist()}

        for default_idx, entry in enumerate(entries):
            local_idx = default_idx
            source = "interpolated"
            weight: float | None = None
            conf = 1.0
            coord_px: np.ndarray | None = None
            coord_norm: np.ndarray | None = None
            if isinstance(entry, dict):
                local_idx = int(entry.get("local_frame_idx", entry.get("frame_idx", default_idx)))
                source = _source_label(entry.get("source", entry.get("coord_source", entry.get("label_source"))))
                if "label_weight" in entry:
                    weight = float(entry["label_weight"])
                elif "weight" in entry:
                    weight = float(entry["weight"])
                if "confidence" in entry:
                    conf = float(entry["confidence"])
                norm_value = entry.get("coord_norm", entry.get("coordinate_norm"))
                if norm_value is not None:
                    coord_norm = _coord_from_value(norm_value, f"{path}[{default_idx}].coord_norm")
                if any(key in entry for key in ("coord_px", "coordinate_px", "coord", "coordinate", "point", "x")):
                    coord_px = _coord_from_value(entry, f"{path}[{default_idx}]")
            else:
                coord_px = _coord_from_value(entry, f"{path}[{default_idx}]")
            if local_idx != default_idx and self.strict:
                raise ValueError(f"{path} interpolation entry {default_idx} has local_frame_idx={local_idx}.")
            if coord_px is None and coord_norm is None:
                raise ValueError(f"{path} interpolation entry {default_idx} has no coordinate.")
            if coord_px is None:
                assert coord_norm is not None
                coord_px = _denormalize_coords(coord_norm[None, :], image_size_hw)[0]
            if coord_norm is None:
                coord_norm = _normalize_coords(coord_px[None, :], image_size_hw)[0]

            if default_idx in anchor_set:
                source = "human"
            source_id = SOURCE_ENCODING[source]
            coords_px[default_idx] = coord_px
            coords_norm[default_idx] = coord_norm
            source_labels.append(source)
            source_ids[default_idx] = source_id
            label_weights[default_idx] = float(weight) if weight is not None else (1.0 if source == "human" else 0.5)
            confidence[default_idx] = conf

        return _DenseTrajectory(coords_px, coords_norm, source_labels, source_ids, label_weights, confidence)

    def _interpolation_entries(self, raw: Any, path: Path) -> list[Any]:
        if isinstance(raw, list):
            return raw
        if not isinstance(raw, dict):
            raise ValueError(f"Interpolation file must be a list or object: {path}")
        version = raw.get("dataset_version")
        if version is not None:
            _check_version(version, allow_legacy_version=self.allow_legacy_version, context=f"Interpolation {path}")
        for key in ("coordinates", "dense_coordinates", "points", "frames", "trajectory"):
            value = raw.get(key)
            if isinstance(value, list):
                return value
        raise ValueError(
            f"Interpolation file {path} must contain one of: coordinates, dense_coordinates, points, frames, trajectory."
        )


class SurgWMBenchSSLFrameDataset(Dataset):
    """Self-supervised frame or short-sequence dataset from extracted clip frames."""

    def __init__(
        self,
        dataset_root: str | Path,
        manifest: str | Path = "manifests/all.jsonl",
        image_size: int | tuple[int, int] = 224,
        sequence_length: int = 1,
        stride: int = 1,
        max_samples: int | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser()
        self.manifest_path = _resolve_manifest_path(self.dataset_root, manifest)
        self.image_size = image_size
        self.sequence_length = max(1, int(sequence_length))
        self.stride = max(1, int(stride))
        self.entries = _read_manifest(self.manifest_path)
        self.samples: list[tuple[dict[str, Any], int, Path]] = []
        for row in self.entries:
            frames_dir = _resolve_path(self.dataset_root, row["frames_dir"])
            num_frames = int(row.get("num_frames") or 0)
            max_start = max(0, num_frames - self.sequence_length)
            for start in range(0, max_start + 1, self.stride):
                self.samples.append((row, start, frames_dir))
                if max_samples is not None and len(self.samples) >= max_samples:
                    return
        if not self.samples:
            raise ValueError(f"No SSL frame samples found in manifest: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row, start, frames_dir = self.samples[index]
        num_frames = int(row.get("num_frames") or 0)
        frame_paths = [
            _resolve_frame_path(self.dataset_root, frames_dir, idx, num_frames, strict=True)
            for idx in range(start, start + self.sequence_length)
        ]
        tensors = [_pil_to_tensor(path, self.image_size)[0] for path in frame_paths]
        frames = torch.stack(tensors, dim=0)
        item = {
            "frames": frames,
            "patient_id": str(row.get("patient_id") or ""),
            "trajectory_id": str(row.get("trajectory_id") or ""),
            "local_frame_idx": torch.tensor(start, dtype=torch.long),
            "frame_indices": torch.arange(start, start + self.sequence_length, dtype=torch.long),
            "frame_path": str(frame_paths[0]),
            "frame_paths": [str(path) for path in frame_paths],
        }
        if self.sequence_length == 1:
            item["image"] = frames[0]
        return item


class SurgWMBenchRawVideoDataset(Dataset):
    """Self-supervised raw source-video clip dataset."""

    def __init__(
        self,
        dataset_root: str | Path,
        split: Literal["train", "val", "test", "all"] | None = "all",
        source_video_manifest: str | Path | None = None,
        clip_length: int = 16,
        stride: int = 1,
        image_size: int | tuple[int, int] = 224,
        backend: Literal["opencv", "torchvision", "decord"] = "opencv",
        max_videos: int | None = None,
        max_clips_per_video: int | None = None,
    ) -> None:
        if backend != "opencv":
            raise ValueError("Only backend='opencv' is implemented for SurgWMBenchRawVideoDataset.")
        self.dataset_root = Path(dataset_root).expanduser()
        self.clip_length = max(1, int(clip_length))
        self.stride = max(1, int(stride))
        self.image_size = image_size
        self.backend = backend
        try:
            import cv2  # noqa: F401
        except ImportError as exc:
            raise ImportError("OpenCV is required for raw video decoding. Install opencv-python or use SSL frames.") from exc

        self.videos = self._load_video_entries(split, source_video_manifest)
        if max_videos is not None:
            self.videos = self.videos[: int(max_videos)]
        self.samples: list[tuple[dict[str, Any], int]] = []
        for video in self.videos:
            frame_count = int(video.get("num_frames") or video.get("frame_count") or 0)
            if frame_count <= 0:
                frame_count = self._probe_frame_count(Path(video["source_video_path"]))
                video["num_frames"] = frame_count
            max_start = max(0, frame_count - self.clip_length)
            starts = list(range(0, max_start + 1, self.stride))
            if max_clips_per_video is not None:
                starts = starts[: int(max_clips_per_video)]
            for start in starts:
                self.samples.append((video, start))
        if not self.samples:
            raise ValueError("No raw video clips available.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import cv2

        video, start = self.samples[index]
        path = Path(video["source_video_path"])
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open source video: {path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        tensors: list[torch.Tensor] = []
        frame_indices: list[int] = []
        for frame_idx in range(start, start + self.clip_length):
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            original = img.size
            target = _target_hw(self.image_size)
            if target is not None:
                target_h, target_w = target
                if img.size != (target_w, target_h):
                    img = img.resize((target_w, target_h), Image.BILINEAR)
            array = np.asarray(img, dtype=np.float32) / 255.0
            tensors.append(torch.from_numpy(array).permute(2, 0, 1).contiguous())
            frame_indices.append(frame_idx)
        cap.release()
        if len(tensors) != self.clip_length:
            raise RuntimeError(f"Decoded {len(tensors)} frames from {path}; expected {self.clip_length}.")
        return {
            "frames": torch.stack(tensors, dim=0),
            "source_video_id": str(video["source_video_id"]),
            "source_video_path": str(path),
            "start_frame": torch.tensor(start, dtype=torch.long),
            "frame_indices": torch.tensor(frame_indices, dtype=torch.long),
        }

    def _load_video_entries(
        self,
        split: Literal["train", "val", "test", "all"] | None,
        source_video_manifest: str | Path | None,
    ) -> list[dict[str, Any]]:
        if source_video_manifest is not None:
            path = _resolve_path(self.dataset_root, source_video_manifest)
            raw = _read_json(path) if path.suffix.lower() != ".jsonl" else _read_manifest(path)
            return self._normalize_video_entries(raw)
        manifest_name = "all" if split is None else split
        manifest = self.dataset_root / "manifests" / f"{manifest_name}.jsonl"
        if manifest.exists():
            rows = _read_manifest(manifest)
            by_id: dict[str, dict[str, Any]] = {}
            for row in rows:
                source_id = str(row.get("source_video_id") or "")
                if not source_id:
                    continue
                path_value = row.get("source_video_path") or f"videos/{source_id}/video_left.avi"
                by_id[source_id] = {
                    "source_video_id": source_id,
                    "source_video_path": str(_resolve_path(self.dataset_root, path_value)),
                    "num_frames": row.get("source_num_frames") or row.get("video_num_frames"),
                }
            if by_id:
                return list(by_id.values())
        metadata = self.dataset_root / "metadata" / "source_videos.json"
        if metadata.exists():
            return self._normalize_video_entries(_read_json(metadata))
        video_paths = sorted(self.dataset_root.glob("videos/*/video_left.avi"))
        return [
            {"source_video_id": path.parent.name, "source_video_path": str(path), "num_frames": None}
            for path in video_paths
        ]

    def _normalize_video_entries(self, raw: Any) -> list[dict[str, Any]]:
        if isinstance(raw, dict):
            if "source_videos" in raw:
                raw = raw["source_videos"]
            elif "videos" in raw:
                raw = raw["videos"]
            else:
                raw = [
                    {"source_video_id": key, **value} if isinstance(value, dict) else {"source_video_id": key}
                    for key, value in raw.items()
                ]
        if not isinstance(raw, list):
            raise ValueError("Source video manifest must be a list or object.")
        entries = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            source_id = str(item.get("source_video_id") or item.get("video_id") or item.get("id") or "")
            if not source_id:
                continue
            path_value = item.get("source_video_path") or item.get("video_path") or f"videos/{source_id}/video_left.avi"
            entries.append(
                {
                    "source_video_id": source_id,
                    "source_video_path": str(_resolve_path(self.dataset_root, path_value)),
                    "num_frames": item.get("num_frames") or item.get("frame_count"),
                }
            )
        return entries

    def _probe_frame_count(self, path: Path) -> int:
        import cv2

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open source video: {path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count


def collate_sparse_anchors(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    frames = torch.stack([item["frames"] for item in batch], dim=0)
    coords = torch.stack([item["human_anchor_coords_norm"] for item in batch], dim=0)
    sampled_indices = torch.stack([item["sampled_indices"] for item in batch], dim=0)
    num_frames = torch.tensor([int(item["num_frames"]) for item in batch], dtype=torch.long)
    denom = torch.clamp(num_frames - 1, min=1).to(torch.float32).unsqueeze(1)
    anchor_dt = (sampled_indices[:, 1:] - sampled_indices[:, :-1]).to(torch.float32) / denom
    actions_delta = coords[:, 1:] - coords[:, :-1]
    sparse_anchor_mask = torch.ones(coords.shape[:2], dtype=torch.bool)
    image_size_original = torch.tensor([item["image_size_original"] for item in batch], dtype=torch.long)
    image_size = torch.stack([item["image_size"] for item in batch], dim=0)
    sparse_positions = torch.arange(coords.shape[1], dtype=torch.long).unsqueeze(0).expand(len(batch), -1)
    return {
        "frames": frames,
        "coords": coords,
        "sampled_indices": sampled_indices,
        "actions_delta": actions_delta,
        "anchor_dt": anchor_dt,
        "sparse_anchor_mask": sparse_anchor_mask,
        "frame_mask": sparse_anchor_mask.clone(),
        "frame_indices": sampled_indices,
        "num_frames": num_frames,
        "image_size_original": image_size_original,
        "image_size": image_size,
        "patient_ids": [item["patient_id"] for item in batch],
        "source_video_ids": [item["source_video_id"] for item in batch],
        "trajectory_ids": [item["trajectory_id"] for item in batch],
        "difficulties": [item["difficulty"] for item in batch],
        "interpolation_methods": [item["interpolation_method"] for item in batch],
        "annotation_paths": [item["annotation_path"] for item in batch],
        "interpolation_paths": [item["interpolation_path"] for item in batch],
        # Compatibility keys.
        "sparse_coords": coords,
        "sparse_indices": sampled_indices,
        "sparse_positions": sparse_positions,
        "dense_coords": coords,
        "dense_coord_mask": sparse_anchor_mask.clone(),
        "clip_ids": [item["trajectory_id"] for item in batch],
    }


def collate_dense_variable_length(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    batch_size = len(batch)
    max_t = max(int(item["frame_indices"].shape[0]) for item in batch)
    channels, height, width = batch[0]["frames"].shape[1:]
    frames = torch.zeros(batch_size, max_t, channels, height, width, dtype=torch.float32)
    coords = torch.zeros(batch_size, max_t, 2, dtype=torch.float32)
    frame_mask = torch.zeros(batch_size, max_t, dtype=torch.bool)
    coord_source = torch.zeros(batch_size, max_t, dtype=torch.long)
    label_weight = torch.zeros(batch_size, max_t, dtype=torch.float32)
    confidence = torch.zeros(batch_size, max_t, dtype=torch.float32)
    frame_indices = torch.full((batch_size, max_t), -1, dtype=torch.long)
    image_size = torch.stack([item["image_size"] for item in batch], dim=0)
    image_size_original = torch.tensor([item["image_size_original"] for item in batch], dtype=torch.long)
    num_frames = torch.tensor([int(item["num_frames"]) for item in batch], dtype=torch.long)
    sparse_coords = torch.stack([item["human_anchor_coords_norm"] for item in batch], dim=0)
    sparse_indices = torch.stack([item["sampled_indices"] for item in batch], dim=0)
    sparse_anchor_mask = torch.ones(sparse_coords.shape[:2], dtype=torch.bool)
    sparse_positions = torch.full(sparse_indices.shape, -1, dtype=torch.long)

    for row, item in enumerate(batch):
        t = int(item["frame_indices"].shape[0])
        frames[row, :t] = item["frames"]
        coords[row, :t] = item["selected_coords_norm"]
        frame_mask[row, :t] = True
        coord_source[row, :t] = item["selected_coord_sources"]
        label_weight[row, :t] = item["selected_label_weights"]
        confidence[row, :t] = item["selected_confidence"]
        frame_indices[row, :t] = item["frame_indices"]
        sparse_positions[row] = item["sparse_positions"]

    dense_coord_mask = frame_mask & (coord_source != SOURCE_ENCODING["unlabeled"])
    actions_delta = coords[:, 1:] - coords[:, :-1] if max_t > 1 else torch.zeros(batch_size, 0, 2)
    action_mask = dense_coord_mask[:, 1:] & dense_coord_mask[:, :-1] if max_t > 1 else torch.zeros(batch_size, 0, dtype=torch.bool)
    return {
        "frames": frames,
        "coords": coords,
        "frame_mask": frame_mask,
        "coord_source": coord_source,
        "label_weight": label_weight,
        "confidence": confidence,
        "frame_indices": frame_indices,
        "actions_delta": actions_delta,
        "action_mask": action_mask,
        "dense_coord_mask": dense_coord_mask,
        "dense_coords": coords,
        "sparse_coords": sparse_coords,
        "sparse_indices": sparse_indices,
        "sparse_positions": sparse_positions,
        "sparse_anchor_mask": sparse_anchor_mask,
        "sampled_indices": sparse_indices,
        "num_frames": num_frames,
        "image_size": image_size,
        "image_size_original": image_size_original,
        "patient_ids": [item["patient_id"] for item in batch],
        "source_video_ids": [item["source_video_id"] for item in batch],
        "trajectory_ids": [item["trajectory_id"] for item in batch],
        "difficulties": [item["difficulty"] for item in batch],
        "interpolation_methods": [item["interpolation_method"] for item in batch],
        "annotation_paths": [item["annotation_path"] for item in batch],
        "interpolation_paths": [item["interpolation_path"] for item in batch],
        "clip_ids": [item["trajectory_id"] for item in batch],
    }


def collate_ssl_video(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    frames = torch.stack([item["frames"] for item in batch], dim=0)
    result = {
        "frames": frames,
        "frame_indices": torch.stack([item["frame_indices"] for item in batch], dim=0),
    }
    if "source_video_id" in batch[0]:
        result.update(
            {
                "source_video_ids": [item["source_video_id"] for item in batch],
                "source_video_paths": [item["source_video_path"] for item in batch],
                "start_frames": torch.stack([item["start_frame"] for item in batch], dim=0),
            }
        )
    else:
        result.update(
            {
                "images": frames[:, 0],
                "patient_ids": [item["patient_id"] for item in batch],
                "trajectory_ids": [item["trajectory_id"] for item in batch],
                "frame_paths": [item["frame_path"] for item in batch],
            }
        )
    return result


def surgwmbench_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Compatibility collate wrapper for staged training/eval migration."""

    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    first = batch[0]
    if "human_anchor_coords_norm" in first and first.get("frame_sampling") == "sparse_anchors":
        return collate_sparse_anchors(batch)
    if "selected_coords_norm" in first:
        return collate_dense_variable_length(batch)
    return collate_ssl_video(batch)


class SurgWMBenchDataset(SurgWMBenchClipDataset):
    """Backward-compatible name for the final-layout clip dataset."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if "data_root" in kwargs or "mode" in kwargs:
            manifest = kwargs.pop("manifest", args[0] if args else None)
            dataset_root = kwargs.pop("data_root", args[1] if len(args) > 1 else None)
            if manifest is None or dataset_root is None:
                raise TypeError("SurgWMBenchDataset compatibility mode requires manifest and data_root.")
            mode = kwargs.pop("mode", "train_dynamics")
            use_dense = bool(kwargs.pop("use_dense_pseudo", False))
            frame_sampling: FrameSampling = "dense" if use_dense else "sparse_anchors"
            if mode == "pretrain_mae":
                frame_sampling = "dense"
            image_size = kwargs.pop("image_size", 224)
            max_frames = kwargs.pop("max_frames_per_clip", None)
            coordinate_normalization = kwargs.pop("coordinate_normalization", "image_size")
            kwargs.pop("image_width", None)
            kwargs.pop("image_height", None)
            kwargs.pop("num_sparse_anchors", None)
            kwargs.pop("split", None)
            super().__init__(
                dataset_root=dataset_root,
                manifest=manifest,
                image_size=image_size,
                frame_sampling=frame_sampling,
                max_frames=max_frames,
                use_dense_pseudo=use_dense,
                normalize_coords=coordinate_normalization != "none",
                **kwargs,
            )
            return
        super().__init__(*args, **kwargs)
