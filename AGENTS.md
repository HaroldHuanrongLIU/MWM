# Repository Guidelines

## Project Scope

This repository is the PyTorch MWM-style baseline for the final public
SurgWMBench dataset. The active code lives under `mwm_torch/`. The original
TensorFlow-style code under `mwm/` is historical reference only and should not
be extended unless explicitly requested.

Treat the final dataset contract as canonical:

- Dataset version is `SurgWMBench`, not `SurgWMBenchv2`.
- Official manifests under `manifests/{train,val,test,all}.jsonl` define splits.
- Do not create random train/val/test splits in data, training, or eval code.
- Sparse 20-anchor human labels are the primary benchmark target.
- Dense interpolation files are auxiliary pseudo-coordinate targets only.
- Supported interpolation methods are `linear`, `pchip`, `akima`, and
  `cubic_spline`.
- Clip frames are real video frames. Only coordinates may be interpolated.
- Do not use legacy `old_frame_idx` as a dense local frame index.

## Environment

Use the reproducible uv environment by default:

```bash
uv sync --locked
uv run --locked python -m pytest -q
```

The project targets Python 3.13 through `pyproject.toml` and `uv.lock`.
`requirements.txt` is kept as a pip fallback, but prefer uv for local work and
reproducible smoke tests.

The real dataset is commonly available at:

```text
/mnt/hdd1/neurips2026_dataset_track/SurgWMBench
```

Do not modify files inside the dataset root.

## Common Commands

Dataset sanity check:

```bash
uv run --locked python -m mwm_torch.data.validate_surgwmbench \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --train-manifest manifests/train.jsonl \
  --val-manifest manifests/val.jsonl \
  --test-manifest manifests/test.jsonl \
  --check-files \
  --check-interpolations
```

MAE pretraining from raw source videos:

```bash
uv run --locked python -m mwm_torch.train_surgwmbench \
  --mode pretrain_mae \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --config configs/surgwmbench_mwm.yaml \
  --ssl-source raw_videos
```

Sparse human-anchor dynamics training:

```bash
uv run --locked python -m mwm_torch.train_surgwmbench \
  --mode train_dynamics_sparse \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --train-manifest manifests/train.jsonl \
  --val-manifest manifests/val.jsonl \
  --interpolation-method linear \
  --pretrained-encoder checkpoints/mwm_mae_surgwmbench.pt \
  --config configs/surgwmbench_mwm.yaml
```

Dense pseudo-coordinate auxiliary training:

```bash
uv run --locked python -m mwm_torch.train_surgwmbench \
  --mode train_dynamics_dense_aux \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --train-manifest manifests/train.jsonl \
  --val-manifest manifests/val.jsonl \
  --interpolation-method linear \
  --use-dense-pseudo \
  --pretrained-encoder checkpoints/mwm_mae_surgwmbench.pt \
  --config configs/surgwmbench_mwm.yaml
```

Evaluation:

```bash
uv run --locked python -m mwm_torch.eval_surgwmbench \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --manifest manifests/test.jsonl \
  --checkpoint checkpoints/mwm_surgwmbench.pt \
  --interpolation-method linear \
  --output results/mwm_linear_test_metrics.json
```

Add `--dense-pseudo-eval` only when reporting auxiliary interpolation metrics.

## Implementation Notes

- Prefer `pathlib`, type hints, and small focused helpers.
- Use PIL for image loading and OpenCV only for raw video decoding.
- Keep normalized coordinates as model targets; keep pixel coordinates for
  reporting/debugging.
- Never silently clip coordinates into image bounds.
- Sparse collate should stack exactly 20 anchors per clip.
- Dense collate should pad variable-length clips and use masks for loss/metrics.
- If loading legacy `SurgWMBenchv2` annotations, require an explicit
  compatibility flag and emit a warning.

## Testing Expectations

Before committing data/model changes, run:

```bash
uv run --locked python -m pytest -q
```

For GPU/training smoke tests, use temporary manifests and checkpoints under
`/tmp` rather than writing small experimental split files into the repository or
dataset root.

