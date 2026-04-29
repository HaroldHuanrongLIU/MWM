"""Lightweight downstream sanity checks for final SurgWMBench releases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .surgwmbench import SUPPORTED_INTERPOLATION_METHODS, _read_manifest, _resolve_path


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _coord(value: dict[str, Any]) -> np.ndarray:
    if "coord_px" in value:
        return np.asarray(value["coord_px"], dtype=np.float32)
    if "human_coord_px" in value:
        return np.asarray(value["human_coord_px"], dtype=np.float32)
    if "x" in value and "y" in value:
        return np.asarray([value["x"], value["y"]], dtype=np.float32)
    raise ValueError(f"Cannot find pixel coordinate in {value}")


def _interpolation_entries(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("coordinates", "dense_coordinates", "points", "frames", "trajectory"):
            value = raw.get(key)
            if isinstance(value, list):
                return value
    raise ValueError("Interpolation file has no supported coordinate list.")


def _check_manifest(dataset_root: Path, manifest: str | Path, check_files: bool, check_interpolations: bool) -> tuple[int, set[str]]:
    rows = _read_manifest(_resolve_path(dataset_root, manifest))
    source_ids: set[str] = set()
    failures = 0
    for row_idx, row in enumerate(rows):
        try:
            if row.get("dataset_version") != "SurgWMBench":
                raise ValueError(f"row {row_idx}: dataset_version must be SurgWMBench")
            source_ids.add(str(row.get("source_video_id") or ""))
            annotation_path = _resolve_path(dataset_root, row["annotation_path"])
            if check_files and not annotation_path.exists():
                raise FileNotFoundError(annotation_path)
            annotation = _read_json(annotation_path)
            anchors = annotation.get("human_anchors")
            if not isinstance(anchors, list) or len(anchors) != 20:
                raise ValueError(f"{annotation_path}: expected exactly 20 human anchors")
            num_frames = int(row.get("num_frames") or annotation.get("num_frames") or 0)
            if num_frames <= 0:
                raise ValueError(f"{annotation_path}: invalid num_frames={num_frames}")
            frames_dir = _resolve_path(dataset_root, row["frames_dir"])
            if check_files:
                frames = annotation.get("frames")
                for idx in sorted({0, num_frames - 1, *[int(a["local_frame_idx"]) for a in anchors]}):
                    frame_path = None
                    if isinstance(frames, list) and idx < len(frames) and isinstance(frames[idx], dict):
                        raw = frames[idx].get("frame_path")
                        if raw:
                            frame_path = _resolve_path(dataset_root, raw)
                    if frame_path is None:
                        for suffix in (".jpg", ".png", ".jpeg"):
                            candidate = frames_dir / f"{idx:06d}{suffix}"
                            if candidate.exists():
                                frame_path = candidate
                                break
                    if frame_path is None or not frame_path.exists():
                        raise FileNotFoundError(f"missing frame {idx} under {frames_dir}")
            if check_interpolations:
                files = row.get("interpolation_files")
                if not isinstance(files, dict):
                    raise ValueError(f"row {row_idx}: interpolation_files must be a method map")
                for method in SUPPORTED_INTERPOLATION_METHODS:
                    if method not in files:
                        raise FileNotFoundError(f"row {row_idx}: missing {method} interpolation")
                    interp_path = _resolve_path(dataset_root, files[method])
                    if check_files and not interp_path.exists():
                        raise FileNotFoundError(interp_path)
                    entries = _interpolation_entries(_read_json(interp_path))
                    if len(entries) != num_frames:
                        raise ValueError(f"{interp_path}: expected {num_frames} entries, got {len(entries)}")
                    for anchor in anchors:
                        idx = int(anchor["local_frame_idx"])
                        interp = entries[idx]
                        if not np.allclose(_coord(anchor), _coord(interp), atol=1e-3):
                            raise ValueError(f"{interp_path}: anchor mismatch at local_frame_idx={idx}")
        except Exception as exc:
            failures += 1
            print(f"FAIL {manifest} row={row_idx}: {exc}")
    print(f"{manifest}: checked={len(rows)} failures={failures}")
    return failures, source_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--train-manifest", default=None)
    parser.add_argument("--val-manifest", default=None)
    parser.add_argument("--test-manifest", default=None)
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--check-interpolations", action="store_true")
    args = parser.parse_args()

    root = Path(args.dataset_root).expanduser()
    manifests = [m for m in [args.manifest, args.train_manifest, args.val_manifest, args.test_manifest] if m]
    if not manifests:
        manifests = ["manifests/train.jsonl"]
    total_failures = 0
    source_sets: dict[str, set[str]] = {}
    for manifest in manifests:
        failures, sources = _check_manifest(root, manifest, args.check_files, args.check_interpolations)
        total_failures += failures
        source_sets[str(manifest)] = sources
    if args.train_manifest and args.val_manifest and source_sets[str(args.train_manifest)] & source_sets[str(args.val_manifest)]:
        raise SystemExit("source-video leakage between train and val manifests")
    if args.train_manifest and args.test_manifest and source_sets[str(args.train_manifest)] & source_sets[str(args.test_manifest)]:
        raise SystemExit("source-video leakage between train and test manifests")
    if args.val_manifest and args.test_manifest and source_sets[str(args.val_manifest)] & source_sets[str(args.test_manifest)]:
        raise SystemExit("source-video leakage between val and test manifests")
    if total_failures:
        raise SystemExit(f"validation failed with {total_failures} failed rows")
    print("SurgWMBench sanity check passed")


if __name__ == "__main__":
    main()
