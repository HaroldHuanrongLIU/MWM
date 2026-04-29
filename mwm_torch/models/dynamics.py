"""Latent transition models for SurgWMBench world-model rollouts."""

from __future__ import annotations

import torch
from torch import nn


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, depth: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for _ in range(depth):
        layers.extend([nn.Linear(current, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()])
        current = hidden_dim
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class ContinuousActionEmbedding(nn.Module):
    """Embed continuous `(dx, dy)` actions."""

    def __init__(self, action_dim: int, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = _mlp(action_dim, hidden_dim, embed_dim, depth=1)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        return self.net(actions)


class DiscreteContinuousActionEmbedding(nn.Module):
    """Embed `(direction_id, magnitude)` actions for 8 compass directions plus stay."""

    def __init__(self, embed_dim: int, hidden_dim: int, num_directions: int = 9) -> None:
        super().__init__()
        self.direction = nn.Embedding(num_directions, embed_dim)
        self.magnitude = _mlp(1, hidden_dim, embed_dim, depth=1)
        self.out = _mlp(embed_dim * 2, hidden_dim, embed_dim, depth=1)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        assert actions.shape[-1] == 2, (
            "Discrete-continuous actions must be [..., 2] with direction id and magnitude."
        )
        direction = actions[..., 0].long().clamp(0, self.direction.num_embeddings - 1)
        magnitude = actions[..., 1:].float().clamp_min(0)
        return self.out(torch.cat([self.direction(direction), self.magnitude(magnitude)], dim=-1))


class CoordinateEmbedding(nn.Module):
    """Embed normalized 2D coordinates."""

    def __init__(self, coord_dim: int, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.coord_dim = coord_dim
        self.net = _mlp(coord_dim, hidden_dim, embed_dim, depth=1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        assert coords.shape[-1] == self.coord_dim, (
            f"Expected coordinates [..., {self.coord_dim}], got {tuple(coords.shape)}."
        )
        return self.net(coords)


class MLPDynamics(nn.Module):
    """Residual MLP transition model `z_t, a_t, p_t -> z_{t+1}`."""

    def __init__(
        self,
        latent_dim: int,
        coord_dim: int = 2,
        action_dim: int = 2,
        action_type: str = "continuous_delta",
        action_embed_dim: int = 64,
        coord_embed_dim: int = 64,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_embed = (
            ContinuousActionEmbedding(action_dim, action_embed_dim, hidden_dim)
            if action_type == "continuous_delta"
            else DiscreteContinuousActionEmbedding(action_embed_dim, hidden_dim)
        )
        self.coord_embed = CoordinateEmbedding(coord_dim, coord_embed_dim, hidden_dim)
        self.transition = _mlp(latent_dim + action_embed_dim + coord_embed_dim, hidden_dim, latent_dim, depth=2)

    def step(self, z_t: torch.Tensor, action_t: torch.Tensor, coord_t: torch.Tensor) -> torch.Tensor:
        action = self.action_embed(action_t)
        coord = self.coord_embed(coord_t)
        delta = self.transition(torch.cat([z_t, action, coord], dim=-1))
        return z_t + delta

    def forward(self, z: torch.Tensor, actions: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Teacher-forced one-step predictions for sequences.

        Args:
            z: `[B, T, D]` latent sequence.
            actions: `[B, T-1, 2]` actions.
            coords: `[B, T, 2]` coordinates.
        """

        assert z.ndim == 3 and actions.ndim == 3 and coords.ndim == 3
        preds = [self.step(z[:, t], actions[:, t], coords[:, t]) for t in range(actions.shape[1])]
        return torch.stack(preds, dim=1) if preds else z[:, :0]


class GRUDynamics(nn.Module):
    """GRU transition model for recurrent latent rollouts."""

    def __init__(
        self,
        latent_dim: int,
        coord_dim: int = 2,
        action_dim: int = 2,
        action_type: str = "continuous_delta",
        action_embed_dim: int = 64,
        coord_embed_dim: int = 64,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_embed = (
            ContinuousActionEmbedding(action_dim, action_embed_dim, hidden_dim)
            if action_type == "continuous_delta"
            else DiscreteContinuousActionEmbedding(action_embed_dim, hidden_dim)
        )
        self.coord_embed = CoordinateEmbedding(coord_dim, coord_embed_dim, hidden_dim)
        self.input = nn.Sequential(
            nn.Linear(latent_dim + action_embed_dim + coord_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.z_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.hidden_to_z = nn.Linear(hidden_dim, latent_dim)

    def step(
        self,
        z_t: torch.Tensor,
        action_t: torch.Tensor,
        coord_t: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action = self.action_embed(action_t)
        coord = self.coord_embed(coord_t)
        inp = self.input(torch.cat([z_t, action, coord], dim=-1))
        h_prev = self.z_to_hidden(z_t) if hidden is None else hidden
        h_next = self.cell(inp, h_prev)
        return self.hidden_to_z(h_next), h_next

    def forward(self, z: torch.Tensor, actions: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Teacher-forced recurrent one-step predictions."""

        assert z.ndim == 3 and actions.ndim == 3 and coords.ndim == 3
        hidden = None
        preds = []
        for t in range(actions.shape[1]):
            pred, hidden = self.step(z[:, t], actions[:, t], coords[:, t], hidden)
            preds.append(pred)
        return torch.stack(preds, dim=1) if preds else z[:, :0]
