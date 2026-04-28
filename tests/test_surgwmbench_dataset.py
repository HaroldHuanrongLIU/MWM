import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from mwm_torch.data import (
    SOURCE_ENCODING,
    SurgWMBenchClipDataset,
    SurgWMBenchSSLFrameDataset,
    collate_dense_variable_length,
    collate_sparse_anchors,
)


IMAGE_HEIGHT = 48
IMAGE_WIDTH = 64
METHOD_OFFSETS = {
    "linear": 0.0,
    "pchip": 1.0,
    "akima": 2.0,
    "cubic_spline": 3.0,
}


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), value, dtype=np.uint8)
    Image.fromarray(array).save(path)


def _coord(frame_idx: int, clip_offset: float, method_offset: float = 0.0) -> list[float]:
    return [float(clip_offset + frame_idx + method_offset), float(clip_offset + frame_idx * 0.5 + method_offset)]


def _norm(coord: list[float]) -> list[float]:
    return [coord[0] / IMAGE_WIDTH, coord[1] / IMAGE_HEIGHT]


def _write_clip(root: Path, patient_id: str, source_video_id: str, trajectory_id: str, length: int, offset: float) -> dict:
    frames_dir = root / "clips" / patient_id / trajectory_id / "frames"
    for idx in range(length):
        _write_image(frames_dir / f"{idx:06d}.jpg", value=(idx * 7) % 255)

    sampled = np.linspace(0, length - 1, num=20, dtype=np.int64).tolist()
    human_anchors = []
    for anchor_idx, local_idx in enumerate(sampled):
        coord = _coord(local_idx, offset)
        human_anchors.append(
            {
                "anchor_idx": anchor_idx,
                "local_frame_idx": int(local_idx),
                "old_frame_idx": anchor_idx,
                "source_frame_idx": int(local_idx + 1000),
                "coord_px": coord,
                "coord_norm": _norm(coord),
            }
        )

    annotation_rel = f"clips/{patient_id}/{trajectory_id}/annotation.json"
    annotation = {
        "dataset_version": "SurgWMBench",
        "patient_id": patient_id,
        "source_video_id": source_video_id,
        "trajectory_id": trajectory_id,
        "difficulty": "low" if length == 25 else "high",
        "num_frames": length,
        "image_height": IMAGE_HEIGHT,
        "image_width": IMAGE_WIDTH,
        "human_anchors": human_anchors,
    }
    (root / annotation_rel).write_text(json.dumps(annotation), encoding="utf-8")

    anchor_set = set(sampled)
    interpolation_files = {}
    for method, method_offset in METHOD_OFFSETS.items():
        entries = []
        for frame_idx in range(length):
            coord = _coord(frame_idx, offset, 0.0 if frame_idx in anchor_set else method_offset)
            source = "human" if frame_idx in anchor_set else "interpolated"
            entries.append(
                {
                    "local_frame_idx": frame_idx,
                    "coord_px": coord,
                    "coord_norm": _norm(coord),
                    "source": source,
                    "confidence": 1.0 if source == "human" else 0.75,
                }
            )
        interp_rel = f"interpolations/{patient_id}/{trajectory_id}.{method}.json"
        interpolation_files[method] = interp_rel
        interp_path = root / interp_rel
        interp_path.parent.mkdir(parents=True, exist_ok=True)
        interp_path.write_text(
            json.dumps(
                {
                    "dataset_version": "SurgWMBench",
                    "interpolation_method": method,
                    "coordinates": entries,
                }
            ),
            encoding="utf-8",
        )

    return {
        "dataset_version": "SurgWMBench",
        "patient_id": patient_id,
        "source_video_id": source_video_id,
        "source_video_path": f"videos/{source_video_id}/video_left.avi",
        "trajectory_id": trajectory_id,
        "difficulty": annotation["difficulty"],
        "num_frames": length,
        "annotation_path": annotation_rel,
        "frames_dir": f"clips/{patient_id}/{trajectory_id}/frames",
        "interpolation_files": interpolation_files,
        "default_interpolation_method": "linear",
        "num_human_anchors": 20,
        "sampled_indices": sampled,
    }


@pytest.fixture()
def surgwmbench_root(tmp_path: Path) -> Path:
    root = tmp_path / "SurgWMBench"
    (root / "videos" / "video_01").mkdir(parents=True)
    (root / "videos" / "video_01" / "video_left.avi").write_bytes(b"synthetic")
    rows = [
        _write_clip(root, "patient_001", "video_01", "traj_001", 25, 1.0),
        _write_clip(root, "patient_002", "video_01", "traj_002", 31, 5.0),
    ]
    manifests = root / "manifests"
    manifests.mkdir(parents=True)
    for split in ("train", "val", "test", "all"):
        (manifests / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows),
            encoding="utf-8",
        )
    metadata = root / "metadata"
    metadata.mkdir()
    (metadata / "source_videos.json").write_text(
        json.dumps(
            {
                "source_videos": [
                    {
                        "source_video_id": "video_01",
                        "source_video_path": "videos/video_01/video_left.avi",
                        "num_frames": 100,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (metadata / "validation_report.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    return root


def test_final_manifest_loads_and_resolves_paths(surgwmbench_root: Path) -> None:
    dataset = SurgWMBenchClipDataset(surgwmbench_root, "manifests/train.jsonl", image_size=32)
    item = dataset[0]
    assert len(dataset) == 2
    assert item["patient_id"] == "patient_001"
    assert item["image_size_original"] == (IMAGE_HEIGHT, IMAGE_WIDTH)
    assert Path(item["annotation_path"]).is_absolute()
    assert Path(item["interpolation_path"]).is_absolute()
    assert item["source_video_path"].endswith("videos/video_01/video_left.avi")


def test_sparse_anchor_mode_returns_20_human_anchors(surgwmbench_root: Path) -> None:
    dataset = SurgWMBenchClipDataset(
        surgwmbench_root,
        "manifests/train.jsonl",
        frame_sampling="sparse_anchors",
        image_size=32,
    )
    item = dataset[0]
    assert item["frames"].shape == (20, 3, 32, 32)
    assert item["human_anchor_coords_px"].shape == (20, 2)
    assert item["human_anchor_coords_norm"].shape == (20, 2)
    assert item["sampled_indices"].shape == (20,)
    assert torch.equal(item["frame_indices"], item["sampled_indices"])
    assert item["dense_coords_norm"].shape == (25, 2)
    assert torch.all(item["selected_coord_sources"] == SOURCE_ENCODING["human"])


def test_dense_mode_returns_variable_length_sources(surgwmbench_root: Path) -> None:
    dataset = SurgWMBenchClipDataset(
        surgwmbench_root,
        "manifests/train.jsonl",
        frame_sampling="dense",
        image_size=(24, 32),
        use_dense_pseudo=True,
    )
    first = dataset[0]
    second = dataset[1]
    assert first["frames"].shape == (25, 3, 24, 32)
    assert second["frames"].shape == (31, 3, 24, 32)
    assert first["selected_coords_norm"].shape == (25, 2)
    assert (first["selected_coord_sources"] == SOURCE_ENCODING["human"]).any()
    assert (first["selected_coord_sources"] == SOURCE_ENCODING["interpolated"]).any()
    assert first["selected_label_weights"][first["selected_coord_sources"] == SOURCE_ENCODING["human"]].eq(1.0).all()
    assert first["selected_label_weights"][first["selected_coord_sources"] == SOURCE_ENCODING["interpolated"]].eq(0.5).all()


def test_window_mode_is_deterministic_from_clip_start(surgwmbench_root: Path) -> None:
    dataset = SurgWMBenchClipDataset(
        surgwmbench_root,
        "manifests/train.jsonl",
        frame_sampling="window",
        max_frames=8,
        image_size=32,
    )
    item = dataset[0]
    assert item["frames"].shape == (8, 3, 32, 32)
    assert item["frame_indices"].tolist() == list(range(8))
    assert item["selected_coords_norm"].shape == (8, 2)
    assert item["sparse_positions"][0].item() == 0
    assert (item["sparse_positions"] >= 8).sum() == 0
    assert (item["sparse_positions"] == -1).any()


def test_collate_sparse_stacks_and_computes_actions(surgwmbench_root: Path) -> None:
    dataset = SurgWMBenchClipDataset(surgwmbench_root, "manifests/train.jsonl", image_size=32)
    batch = collate_sparse_anchors([dataset[0], dataset[1]])
    assert batch["frames"].shape == (2, 20, 3, 32, 32)
    assert batch["coords"].shape == (2, 20, 2)
    assert batch["sampled_indices"].shape == (2, 20)
    assert batch["actions_delta"].shape == (2, 19, 2)
    assert batch["anchor_dt"].shape == (2, 19)
    expected_dt = (batch["sampled_indices"][0, 1:] - batch["sampled_indices"][0, :-1]).float() / 24.0
    assert torch.allclose(batch["anchor_dt"][0], expected_dt)
    assert batch["sparse_anchor_mask"].all()


def test_collate_dense_pads_variable_length_clips(surgwmbench_root: Path) -> None:
    dataset = SurgWMBenchClipDataset(
        surgwmbench_root,
        "manifests/train.jsonl",
        frame_sampling="dense",
        image_size=32,
        use_dense_pseudo=True,
    )
    batch = collate_dense_variable_length([dataset[0], dataset[1]])
    assert batch["frames"].shape == (2, 31, 3, 32, 32)
    assert batch["coords"].shape == (2, 31, 2)
    assert batch["frame_mask"][0].sum() == 25
    assert batch["frame_mask"][1].sum() == 31
    assert not batch["frame_mask"][0, 25:].any()
    assert batch["coord_source"].shape == (2, 31)
    assert batch["actions_delta"].shape == (2, 30, 2)
    assert batch["action_mask"][0].sum() == 24


def test_interpolation_method_switching_loads_different_files(surgwmbench_root: Path) -> None:
    linear = SurgWMBenchClipDataset(
        surgwmbench_root,
        "manifests/train.jsonl",
        interpolation_method="linear",
        frame_sampling="dense",
        image_size=32,
    )[0]
    pchip = SurgWMBenchClipDataset(
        surgwmbench_root,
        "manifests/train.jsonl",
        interpolation_method="pchip",
        frame_sampling="dense",
        image_size=32,
    )[0]
    non_anchor = torch.nonzero(linear["selected_coord_sources"] == SOURCE_ENCODING["interpolated"])[0].item()
    assert not torch.allclose(linear["selected_coords_px"][non_anchor], pchip["selected_coords_px"][non_anchor])
    assert linear["interpolation_method"] == "linear"
    assert pchip["interpolation_method"] == "pchip"


def test_missing_interpolation_file_fails_in_strict_mode(surgwmbench_root: Path) -> None:
    missing = surgwmbench_root / "interpolations" / "patient_001" / "traj_001.pchip.json"
    missing.unlink()
    dataset = SurgWMBenchClipDataset(
        surgwmbench_root,
        "manifests/train.jsonl",
        interpolation_method="pchip",
        frame_sampling="dense",
        image_size=32,
        strict=True,
    )
    with pytest.raises(FileNotFoundError, match="Interpolation file"):
        dataset[0]


def test_wrong_dataset_version_requires_legacy_opt_in(surgwmbench_root: Path) -> None:
    annotation_path = surgwmbench_root / "clips" / "patient_001" / "traj_001" / "annotation.json"
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotation["dataset_version"] = "SurgWMBenchv2"
    annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
    dataset = SurgWMBenchClipDataset(surgwmbench_root, "manifests/train.jsonl", image_size=32)
    with pytest.raises(ValueError, match="SurgWMBenchv2"):
        dataset[0]
    legacy = SurgWMBenchClipDataset(
        surgwmbench_root,
        "manifests/train.jsonl",
        image_size=32,
        allow_legacy_version=True,
    )
    with pytest.warns(UserWarning, match="legacy dataset_version"):
        assert legacy[0]["patient_id"] == "patient_001"


def test_ssl_frame_dataset_returns_frames_without_coordinates(surgwmbench_root: Path) -> None:
    dataset = SurgWMBenchSSLFrameDataset(
        surgwmbench_root,
        "manifests/train.jsonl",
        image_size=32,
        sequence_length=2,
        stride=3,
    )
    item = dataset[0]
    assert item["frames"].shape == (2, 3, 32, 32)
    assert item["patient_id"] == "patient_001"
    assert item["trajectory_id"] == "traj_001"
    assert "human_anchor_coords_norm" not in item
    assert "selected_coords_norm" not in item


def test_data_layer_does_not_randomly_split_clips() -> None:
    source = Path("mwm_torch/data/surgwmbench.py").read_text(encoding="utf-8")
    assert "random_split" not in source
    assert "train_test_split" not in source
