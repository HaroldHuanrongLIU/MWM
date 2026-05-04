from __future__ import annotations

import sys
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if (parent / "surgwmbench_benchmark").is_dir():
        sys.path.insert(0, str(parent))
        break

import torch
from torch import nn

from mwm_torch.models.mwm_surgwmbench import MWMSurgWMBenchModel
from surgwmbench_benchmark.future_model_helpers import FutureFrameDecoder, zero_actions
from surgwmbench_benchmark.future_prediction import FutureProtocolConfig, main


class MWMFuturePredictionModel(nn.Module):
    """Future-prediction wrapper that uses the MWM visual encoder and dynamics core."""

    def __init__(self, config: FutureProtocolConfig) -> None:
        super().__init__()
        self.core = MWMSurgWMBenchModel(
            image_size=config.image_size,
            patch_size=16,
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            action_dim=2,
            encoder_depth=4,
            decoder_depth=2,
            num_heads=8,
            decoder_num_heads=8,
        )
        self.frame_decoder = FutureFrameDecoder(config.latent_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        frames = batch["context_frames"]
        _, _, _, height, width = frames.shape
        z_context = self.core.encode_frames(frames)
        actions = zero_actions(batch, action_dim=2)
        z_future, pred_coords = self.core.rollout(z_context[:, -1], batch["context_coords_norm"][:, -1], actions)
        pred_frames = self.frame_decoder(z_future, (height, width))
        return {"pred_frames": pred_frames, "pred_coords_norm": pred_coords}


def make_model(config: FutureProtocolConfig) -> nn.Module:
    return MWMFuturePredictionModel(config)


if __name__ == "__main__":
    raise SystemExit(main("mwm", "MWMFuturePredictionCore", "mwm_torch.data.surgwmbench", make_model))
