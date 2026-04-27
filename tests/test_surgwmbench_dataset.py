import json

import numpy as np
import pytest
import torch
from PIL import Image

from mwm_torch.data import SurgWMBenchDataset, surgwmbench_collate


def _write_image(path, value=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((16, 20, 3), value, dtype=np.uint8)
    Image.fromarray(array).save(path)


def test_jsonl_manifest_sparse_and_dense_collate(tmp_path):
    root = tmp_path / "data"
    for clip, length in [("clip_a", 5), ("clip_b", 3)]:
        for idx in range(length):
            _write_image(root / "clips" / clip / f"{idx:06d}.jpg", value=idx)

    entries = [
        {
            "clip_id": "clip_a",
            "frame_paths": [f"clips/clip_a/{idx:06d}.jpg" for idx in range(5)],
            "num_frames": 5,
            "sampled_indices": [0, 2, 4],
            "human_labeled_coordinates": [[0, 0], [10, 8], [20, 16]],
            "pseudo_coordinates": [[idx * 5, idx * 4] for idx in range(5)],
            "split": "train",
        },
        {
            "clip_id": "clip_b",
            "frame_paths": [f"clips/clip_b/{idx:06d}.jpg" for idx in range(3)],
            "num_frames": 3,
            "sampled_indices": [0, 1, 2],
            "human_labeled_coordinates": [[0, 0], [5, 4], [10, 8]],
            "split": "train",
        },
    ]
    manifest = tmp_path / "imp.jsonl"
    manifest.write_text("\n".join(json.dumps(entry) for entry in entries))

    dataset = SurgWMBenchDataset(
        manifest,
        root,
        mode="train_dynamics",
        image_size=32,
        use_dense_pseudo=True,
        max_frames_per_clip=4,
        num_sparse_anchors=3,
    )
    first = dataset[0]
    assert first["frames"].shape[1:] == (3, 32, 32)
    assert first["sparse_coords"].max() <= 1.0
    assert first["dense_coord_mask"].any()

    batch = surgwmbench_collate([dataset[0], dataset[1]])
    assert batch["frames"].shape[0] == 2
    assert batch["frame_mask"].dtype == torch.bool
    assert batch["sparse_anchor_mask"].shape == (2, 3)
    assert batch["dense_coord_mask"][1].sum() == 0


def test_raw_pretrain_manifest_flattens_frames(tmp_path):
    root = tmp_path / "data"
    for idx in range(2):
        _write_image(root / "raw" / "case001" / f"{idx:06d}.jpg", value=idx)
    manifest = tmp_path / "raw.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "video_id": "case001",
                    "frame_paths": [f"raw/case001/{idx:06d}.jpg" for idx in range(2)],
                    "split": "train",
                }
            ]
        )
    )
    dataset = SurgWMBenchDataset(manifest, root, mode="pretrain_mae", image_size=32)
    assert len(dataset) == 2
    batch = surgwmbench_collate([dataset[0], dataset[1]])
    assert batch["images"].shape == (2, 3, 32, 32)


def test_imp_manifest_requires_sparse_labels(tmp_path):
    manifest = tmp_path / "bad.json"
    manifest.write_text(json.dumps([{"clip_id": "bad", "frame_paths": ["missing.jpg"]}]))
    with pytest.raises(ValueError, match="sampled_indices"):
        SurgWMBenchDataset(manifest, tmp_path, mode="train_dynamics")
