# SurgWMBench PyTorch MWM Baseline

This repository keeps the original TensorFlow 2 MWM implementation in `mwm/`
unchanged and adds a separate PyTorch baseline under `mwm_torch/` for
SurgWMBench. The new path is for offline surgical video data, not online RL
environment interaction.

## Setup

```bash
pip install -r requirements_torch.txt
```

The implementation uses PyTorch 2.x APIs, `torch.amp` mixed precision, and an
optional config flag for `torch.compile`.

## Manifest Format

Raw SAR-RARP50 pretraining entries may contain only video frames:

```json
{"video_id": "case001", "frame_paths": ["raw/case001/000000.jpg"], "split": "train"}
```

Instrument Motion Planning entries require sparse human labels:

```json
{
  "clip_id": "case001_clip0007",
  "source_video_id": "case001",
  "frame_paths": ["clips/case001_clip0007/000000.jpg"],
  "num_frames": 73,
  "sampled_indices": [0, 4, 8, 12],
  "human_labeled_coordinates": [[10, 20], [12, 22], [14, 24], [16, 26]],
  "pseudo_coordinates": [[10, 20], [10.5, 20.5]],
  "interpolation_method": "linear",
  "split": "train"
}
```

JSONL files should contain one entry per line. JSON files may contain a list of
entries or an object with an `entries` field. Relative frame paths are resolved
against `--data-root`.

Sparse 20-anchor human labels are the primary supervision and evaluation target.
Dense pseudo coordinates are optional auxiliary labels and are always reported
separately.

## Example Commands

```bash
python -m mwm_torch.train_surgwmbench \
  --mode pretrain_mae \
  --manifest data/raw_train.jsonl \
  --data-root /path/to/SurgWMBench \
  --config configs/surgwmbench_mwm.yaml
```

```bash
python -m mwm_torch.train_surgwmbench \
  --mode train_dynamics \
  --manifest data/imp_train.jsonl \
  --val-manifest data/imp_val.jsonl \
  --data-root /path/to/SurgWMBench \
  --pretrained-encoder checkpoints/mwm_mae.pt \
  --config configs/surgwmbench_mwm.yaml
```

```bash
python -m mwm_torch.eval_surgwmbench \
  --manifest data/imp_test.jsonl \
  --data-root /path/to/SurgWMBench \
  --checkpoint checkpoints/mwm_surgwmbench.pt \
  --output results/mwm_surgwmbench_metrics.json
```

## Original Repo Audit

The original code is compact and TensorFlow-specific:

- `mwm/train.py` creates online RL environments and replay buffers for DMC,
  MetaWorld, and RLBench.
- `mwm/common/mae.py` implements masked visual autoencoding with TF/Keras and
  `tfimm`.
- `mwm/common/nets.py` implements the TensorFlow RSSM and distribution heads.
- `mwm/common/replay.py` stores online episodes and exposes `tf.data.Dataset`.

The PyTorch SurgWMBench baseline replaces these runtime pieces with offline
manifest datasets, PyTorch modules, and trajectory metrics while preserving the
MWM design pattern: masked visual representation learning plus latent dynamics
and multi-step rollout.
