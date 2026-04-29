"""End-to-end PyTorch MWM baseline for SurgWMBench."""

from __future__ import annotations

import torch
from torch import nn

from .dynamics import GRUDynamics, MLPDynamics
from .masked_autoencoder import MaskedVisualAutoencoder


def _head(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
        nn.Sigmoid(),
    )


class MWMSurgWMBenchModel(nn.Module):
    """Masked visual encoder, latent dynamics, and coordinate decoder."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        latent_dim: int = 512,
        hidden_dim: int = 512,
        coord_dim: int = 2,
        action_dim: int = 2,
        coord_embed_dim: int = 64,
        action_embed_dim: int = 64,
        action_type: str = "continuous_delta",
        dynamics_type: str = "gru",
        mask_ratio: float = 0.5,
        encoder_depth: int = 4,
        decoder_depth: int = 2,
        num_heads: int = 8,
        decoder_num_heads: int = 8,
        conv_stem_channels: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.coord_dim = coord_dim
        self.visual = MaskedVisualAutoencoder(
            image_size=image_size,
            patch_size=patch_size,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            encoder_depth=encoder_depth,
            decoder_depth=decoder_depth,
            num_heads=num_heads,
            decoder_num_heads=decoder_num_heads,
            conv_stem_channels=conv_stem_channels,
            mask_ratio=mask_ratio,
        )
        dynamics_cls = {"mlp": MLPDynamics, "gru": GRUDynamics}.get(dynamics_type)
        if dynamics_cls is None:
            raise ValueError(f"Unsupported dynamics_type={dynamics_type!r}; expected 'mlp' or 'gru'.")
        self.dynamics_type = dynamics_type
        self.dynamics = dynamics_cls(
            latent_dim=latent_dim,
            coord_dim=coord_dim,
            action_dim=action_dim,
            action_type=action_type,
            action_embed_dim=action_embed_dim,
            coord_embed_dim=coord_embed_dim,
            hidden_dim=hidden_dim,
        )
        self.coord_head = _head(latent_dim, hidden_dim, coord_dim)

    def forward_mae(self, images: torch.Tensor, mask_ratio: float | None = None) -> dict[str, torch.Tensor]:
        """Run masked visual autoencoder pretraining forward pass."""

        return self.visual(images, mask_ratio=mask_ratio)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode image batch `[B, C, H, W]` into visual latents."""

        return self.visual.encode(images)

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode video frames `[B, T, C, H, W]` into latents `[B, T, D]`."""

        assert frames.ndim == 5, f"Expected frames [B, T, C, H, W], got {tuple(frames.shape)}."
        batch, time, channels, height, width = frames.shape
        flat = frames.reshape(batch * time, channels, height, width)
        z = self.encode_images(flat)
        return z.reshape(batch, time, -1)

    def decode_coords(self, z: torch.Tensor) -> torch.Tensor:
        """Decode normalized coordinates from latent states."""

        return self.coord_head(z)

    def predict_next_latents(self, z: torch.Tensor, coords: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Teacher-forced transition predictions for a latent sequence."""

        assert z.shape[:2] == coords.shape[:2], "Latents and coordinates must share [B, T]."
        assert actions.shape[:2] == (z.shape[0], max(z.shape[1] - 1, 0)), (
            f"Expected actions [B, T-1, A], got {tuple(actions.shape)} for z {tuple(z.shape)}."
        )
        return self.dynamics(z, actions, coords)

    def rollout(
        self, initial_z: torch.Tensor, initial_coord: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Open-loop rollout from an initial latent and coordinate.

        Returns predicted future latents and coordinates for each action step,
        both with horizon length `actions.shape[1]`.
        """

        assert actions.ndim == 3, f"Expected actions [B, H, A], got {tuple(actions.shape)}."
        z = initial_z
        coord = initial_coord
        hidden = None
        z_preds = []
        coord_preds = []
        for step in range(actions.shape[1]):
            if isinstance(self.dynamics, GRUDynamics):
                z, hidden = self.dynamics.step(z, actions[:, step], coord, hidden)
            else:
                z = self.dynamics.step(z, actions[:, step], coord)
            pred_coord = self.decode_coords(z)
            z_preds.append(z)
            coord_preds.append(pred_coord)
            if actions.shape[-1] >= self.coord_dim:
                coord = (coord + actions[:, step, : self.coord_dim]).clamp(0.0, 1.0)
            else:
                coord = pred_coord
        if not z_preds:
            empty_z = initial_z[:, :0].reshape(initial_z.shape[0], 0, initial_z.shape[-1])
            empty_c = initial_coord[:, :0].reshape(initial_coord.shape[0], 0, initial_coord.shape[-1])
            return empty_z, empty_c
        return torch.stack(z_preds, dim=1), torch.stack(coord_preds, dim=1)

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode frames and decode coordinates without applying dynamics."""

        z = self.encode_frames(frames)
        coords = self.decode_coords(z)
        return {"z": z, "coords": coords}
