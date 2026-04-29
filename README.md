# SurgWMBench MWM PyTorch Baseline

This repository is adapted for **SurgWMBench: A Dataset and World-Model Benchmark for Surgical Instrument Motion Planning**. The active implementation is the PyTorch code under `mwm_torch/`; the original TensorFlow MWM code under `mwm/` is kept only as historical reference.

The final public dataset version is `SurgWMBench`, not `SurgWMBenchv2`.

## Environment

Recommended reproducible setup:

```bash
uv sync --locked
uv run --locked python -m pytest -q
```

Pip fallback:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

The lock file targets Python `3.13` and includes PyTorch, torchvision, OpenCV, PIL, PyYAML, NumPy, and pytest.

## Dataset

Expected final dataset root:

```text
SurgWMBench/
  videos/<source_video_id>/video_left.avi
  clips/<patient_id>/<trajectory_id>/frames/
  clips/<patient_id>/<trajectory_id>/annotation.json
  interpolations/<patient_id>/<trajectory_id>.<method>.json
  manifests/{train,val,test,all}.jsonl
  metadata/
```

All paths inside JSON/JSONL files are relative to the dataset root. Use official manifests only; the code should not create random train/val/test splits.

The primary benchmark target is the sparse set of exactly 20 human-labeled anchors in `annotation.json`. Dense pseudo coordinates from interpolation files are auxiliary only. Supported interpolation methods are `linear`, `pchip`, `akima`, and `cubic_spline`.

## Data Smoke Check

```bash
uv run --locked python - <<'PY'
from pathlib import Path
from mwm_torch.data import SurgWMBenchClipDataset, collate_sparse_anchors

root = Path("/mnt/hdd1/neurips2026_dataset_track/SurgWMBench")
dataset = SurgWMBenchClipDataset(root, "manifests/train.jsonl", frame_sampling="sparse_anchors")
item = dataset[0]
batch = collate_sparse_anchors([dataset[0], dataset[1]])
print(len(dataset), item["trajectory_id"], item["frames"].shape, batch["frames"].shape)
PY
```

## Commands

Sparse dynamics training with the current CLI:

```bash
uv run --locked python -m mwm_torch.train_surgwmbench \
  --mode train_dynamics \
  --data-root /path/to/SurgWMBench \
  --manifest manifests/train.jsonl \
  --val-manifest manifests/val.jsonl \
  --config configs/surgwmbench_mwm.yaml
```

Evaluation:

```bash
uv run --locked python -m mwm_torch.eval_surgwmbench \
  --data-root /path/to/SurgWMBench \
  --manifest manifests/test.jsonl \
  --checkpoint checkpoints/mwm_surgwmbench.pt \
  --output results/mwm_linear_test_metrics.json
```

MAE pretraining and dense auxiliary training use the data loaders in `mwm_torch.data`; their CLI integration is staged separately from the current data-loader migration.

## Notes

- Sparse human-anchor metrics are primary.
- Dense interpolation metrics must be reported as pseudo-coordinate metrics.
- `old_frame_idx` is preserved only as legacy metadata and is not the dense local frame index.
- Clip frames are real video frames; only coordinates are interpolated.
