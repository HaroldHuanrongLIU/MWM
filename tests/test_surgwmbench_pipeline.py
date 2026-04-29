import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mwm_torch.data import SurgWMBenchRawVideoDataset


def _write_image(path: Path, value: int, height: int = 32, width: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((height, width, 3), value, dtype=np.uint8)).save(path)


def _coord(frame_idx: int, offset: float = 0.0) -> list[float]:
    return [float(4 + offset + frame_idx * 0.2), float(6 + offset + frame_idx * 0.1)]


def _norm(coord: list[float], height: int = 32, width: int = 32) -> list[float]:
    return [coord[0] / width, coord[1] / height]


def _write_tiny_avi(path: Path, num_frames: int = 8) -> None:
    cv2 = pytest.importorskip("cv2")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (32, 32))
    assert writer.isOpened()
    for idx in range(num_frames):
        frame = np.full((32, 32, 3), idx * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _write_clip(root: Path, patient_id: str, trajectory_id: str, length: int, offset: float) -> dict:
    frames_dir = root / "clips" / patient_id / trajectory_id / "frames"
    for idx in range(length):
        _write_image(frames_dir / f"{idx:06d}.png", idx)
    sampled = np.linspace(0, length - 1, 20, dtype=np.int64).tolist()
    anchors = []
    frames = []
    for frame_idx in range(length):
        coord = _coord(frame_idx, offset)
        is_anchor = frame_idx in sampled
        anchor_idx = sampled.index(frame_idx) if is_anchor else None
        frames.append(
            {
                "local_frame_idx": frame_idx,
                "frame_path": f"clips/{patient_id}/{trajectory_id}/frames/{frame_idx:06d}.png",
                "coord_source": "human" if is_anchor else "unlabeled",
                "is_human_labeled": is_anchor,
                "anchor_idx": anchor_idx,
                "human_coord_px": coord if is_anchor else None,
                "human_coord_norm": _norm(coord) if is_anchor else None,
            }
        )
        if is_anchor:
            anchors.append(
                {
                    "anchor_idx": anchor_idx,
                    "local_frame_idx": frame_idx,
                    "old_frame_idx": anchor_idx,
                    "source_frame_idx": frame_idx,
                    "coord_px": coord,
                    "coord_norm": _norm(coord),
                }
            )
    annotation_rel = f"clips/{patient_id}/{trajectory_id}/annotation.json"
    (root / annotation_rel).write_text(
        json.dumps(
            {
                "dataset_version": "SurgWMBench",
                "patient_id": patient_id,
                "source_video_id": patient_id,
                "source_video_path": f"videos/{patient_id}/video_left.avi",
                "trajectory_id": trajectory_id,
                "difficulty": "low",
                "num_frames": length,
                "image_size": {"height": 32, "width": 32},
                "sampled_indices": sampled,
                "human_anchors": anchors,
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )
    interpolation_files = {}
    for method in ("linear", "pchip", "akima", "cubic_spline"):
        rel = f"interpolations/{patient_id}/{trajectory_id}.{method}.json"
        interpolation_files[method] = rel
        entries = []
        for frame_idx in range(length):
            coord = _coord(frame_idx, offset)
            source = "human" if frame_idx in sampled else "interpolated"
            entries.append(
                {
                    "local_frame_idx": frame_idx,
                    "coord_px": coord,
                    "coord_norm": _norm(coord),
                    "source": source,
                    "label_weight": 1.0 if source == "human" else 0.5,
                    "confidence": 1.0 if source == "human" else 0.6,
                }
            )
        interp_path = root / rel
        interp_path.parent.mkdir(parents=True, exist_ok=True)
        interp_path.write_text(json.dumps({"dataset_version": "SurgWMBench", "coordinates": entries}), encoding="utf-8")
    return {
        "dataset_version": "SurgWMBench",
        "patient_id": patient_id,
        "source_video_id": patient_id,
        "source_video_path": f"videos/{patient_id}/video_left.avi",
        "trajectory_id": trajectory_id,
        "difficulty": "low",
        "num_frames": length,
        "annotation_path": annotation_rel,
        "frames_dir": f"clips/{patient_id}/{trajectory_id}/frames",
        "interpolation_files": interpolation_files,
        "default_interpolation_method": "linear",
        "num_human_anchors": 20,
        "sampled_indices": sampled,
    }


@pytest.fixture()
def tiny_surgwmbench(tmp_path: Path) -> Path:
    root = tmp_path / "SurgWMBench"
    rows = [
        _write_clip(root, "video_01", "traj_001", 25, 0.0),
        _write_clip(root, "video_02", "traj_002", 26, 1.0),
    ]
    for row in rows:
        _write_tiny_avi(root / row["source_video_path"], num_frames=8)
    manifests = root / "manifests"
    manifests.mkdir(parents=True)
    for split in ("train", "val", "test", "all"):
        (manifests / f"{split}.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    metadata = root / "metadata"
    metadata.mkdir()
    (metadata / "source_videos.json").write_text(
        json.dumps(
            {
                "dataset_version": "SurgWMBench",
                "videos": [
                    {
                        "source_video_id": row["source_video_id"],
                        "source_video_path": row["source_video_path"],
                        "num_frames": 8,
                    }
                    for row in rows
                ],
            }
        ),
        encoding="utf-8",
    )
    return root


def _config(path: Path, root: Path, epochs: int = 1) -> Path:
    config = path / "config.yaml"
    config.write_text(
        f"""
model:
  image_size: 32
  patch_size: 8
  latent_dim: 32
  hidden_dim: 32
  coord_dim: 2
  action_dim: 3
  coord_embed_dim: 8
  action_embed_dim: 8
  action_type: continuous_delta
  use_time_delta: true
  dynamics_type: gru
  mask_ratio: 0.5
  encoder_depth: 1
  decoder_depth: 1
  num_heads: 4
  decoder_num_heads: 4
  conv_stem_channels: [8, 16]
  compile: false
data:
  dataset_root: {root}
  train_manifest: manifests/train.jsonl
  val_manifest: manifests/val.jsonl
  test_manifest: manifests/test.jsonl
  interpolation_method: linear
  use_dense_pseudo: false
  max_frames_per_clip: null
  num_sparse_anchors: 20
  ssl:
    source: clip_frames
    backend: opencv
    clip_length: 2
    stride: 2
    fallback_to_clip_frames: true
    max_videos: 1
    max_clips_per_video: 1
train:
  batch_size: 2
  num_workers: 0
  lr: 1.0e-4
  weight_decay: 1.0e-4
  epochs: {epochs}
  precision: "none"
  seed: 42
  grad_clip_norm: 100.0
  output_dir: "{path / 'checkpoints'}"
  log_every: 1
  freeze_encoder: false
loss:
  recon_weight: 1.0
  latent_weight: 1.0
  sparse_coord_weight: 1.0
  dense_coord_weight: 1.0
  dense_pseudo_coord_weight: 1.0
  smoothness_weight: 0.0
  coord_loss: smooth_l1
eval:
  primary_target: sparse_human_anchors
  report_pixel_metrics: true
  horizons: [1, 3, 5, 10, 20]
  stratify_by_difficulty: true
  dense_pseudo_eval: false
""",
        encoding="utf-8",
    )
    return config


def test_raw_video_dataset_decodes_source_video(tiny_surgwmbench: Path) -> None:
    dataset = SurgWMBenchRawVideoDataset(
        tiny_surgwmbench,
        split="train",
        clip_length=4,
        stride=4,
        image_size=32,
        max_videos=1,
        max_clips_per_video=1,
    )
    item = dataset[0]
    assert item["frames"].shape == (4, 3, 32, 32)
    assert item["source_video_path"].endswith("video_left.avi")


def test_final_train_eval_cli_smoke(tiny_surgwmbench: Path, tmp_path: Path) -> None:
    config = _config(tmp_path, tiny_surgwmbench)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mwm_torch.train_surgwmbench",
            "--mode",
            "train_dynamics_sparse",
            "--dataset-root",
            str(tiny_surgwmbench),
            "--train-manifest",
            "manifests/train.jsonl",
            "--val-manifest",
            "manifests/val.jsonl",
            "--config",
            str(config),
        ],
        check=True,
    )
    checkpoint = tmp_path / "checkpoints" / "mwm_surgwmbench.pt"
    output = tmp_path / "metrics.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mwm_torch.eval_surgwmbench",
            "--dataset-root",
            str(tiny_surgwmbench),
            "--manifest",
            "manifests/test.jsonl",
            "--checkpoint",
            str(checkpoint),
            "--interpolation-method",
            "linear",
            "--dense-pseudo-eval",
            "--output",
            str(output),
            "--config",
            str(config),
        ],
        check=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["primary_target"] == "sparse_human_anchors"
    assert result["dense_target"] == "pseudo_coordinates"
    assert result["num_clips"] == 2
    assert "sparse_ade" in result["metrics_overall"]
    assert "dense_pseudo_linear_ade" in result["metrics_overall"]


def test_pretrain_mae_clip_frames_cli_smoke(tiny_surgwmbench: Path, tmp_path: Path) -> None:
    config = _config(tmp_path, tiny_surgwmbench)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mwm_torch.train_surgwmbench",
            "--mode",
            "pretrain_mae",
            "--dataset-root",
            str(tiny_surgwmbench),
            "--train-manifest",
            "manifests/train.jsonl",
            "--ssl-source",
            "clip_frames",
            "--config",
            str(config),
        ],
        check=True,
    )
    assert (tmp_path / "checkpoints" / "mwm_mae_surgwmbench.pt").exists()
