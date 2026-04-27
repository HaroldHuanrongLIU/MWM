"""Masked visual autoencoder with convolutional feature masking."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


def _get_1d_sincos_pos_embed(embed_dim: int, positions: np.ndarray) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError("Sine-cosine positional embedding dimension must be even.")
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)
    out = np.einsum("m,d->md", positions.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool) -> np.ndarray:
    """Return fixed 2D sine-cosine positional embeddings."""

    if embed_dim % 2 != 0:
        raise ValueError("2D positional embedding dimension must be even.")
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape(2, 1, grid_size, grid_size)
    emb_h = _get_1d_sincos_pos_embed(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed(embed_dim // 2, grid[1])
    pos = np.concatenate([emb_h, emb_w], axis=1)
    if cls_token:
        pos = np.concatenate([np.zeros((1, embed_dim), dtype=np.float32), pos], axis=0)
    return pos


class ConvStem(nn.Module):
    """Small convolutional stem used before token masking."""

    def __init__(self, in_channels: int, channels: list[int], embed_dim: int, patch_size: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = in_channels
        stem_stride = 1
        for width in channels:
            groups = min(8, width)
            while width % groups != 0:
                groups -= 1
            layers.extend(
                [
                    nn.Conv2d(current, width, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(num_groups=groups, num_channels=width),
                    nn.GELU(),
                ]
            )
            current = width
            stem_stride *= 2
        feature_patch = max(1, patch_size // stem_stride)
        layers.append(nn.Conv2d(current, embed_dim, kernel_size=feature_patch, stride=feature_patch))
        self.net = nn.Sequential(*layers)
        self.output_stride = stem_stride * feature_patch

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.net(images)
        return features.flatten(2).transpose(1, 2).contiguous()


class MaskedVisualAutoencoder(nn.Module):
    """MWM-style MAE that masks convolutional feature tokens and reconstructs RGB patches."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        latent_dim: int = 512,
        hidden_dim: int = 512,
        encoder_depth: int = 4,
        decoder_depth: int = 2,
        num_heads: int = 8,
        decoder_num_heads: int = 8,
        conv_stem_channels: list[int] | None = None,
        mask_ratio: float = 0.5,
        norm_pix_loss: bool = False,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        conv_stem_channels = conv_stem_channels or [64, 128]
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size**2

        self.stem = ConvStem(in_channels, conv_stem_channels, latent_dim, patch_size)
        if self.stem.output_stride != patch_size:
            raise ValueError(
                "Conv stem output stride must equal patch_size. "
                f"Got output_stride={self.stem.output_stride}, patch_size={patch_size}."
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)
        self.encoder_norm = nn.LayerNorm(latent_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, latent_dim))
        self.latent_proj = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, latent_dim))

        self.decoder_embed = nn.Linear(latent_dim, hidden_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=decoder_num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder_pred = nn.Linear(hidden_dim, patch_size * patch_size * in_channels)

        pos_embed = get_2d_sincos_pos_embed(latent_dim, self.grid_size, cls_token=True)
        decoder_pos_embed = get_2d_sincos_pos_embed(hidden_dim, self.grid_size, cls_token=True)
        self.register_buffer("pos_embed", torch.from_numpy(pos_embed).unsqueeze(0), persistent=False)
        self.register_buffer(
            "decoder_pos_embed", torch.from_numpy(decoder_pos_embed).unsqueeze(0), persistent=False
        )
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        """Convert images [B, C, H, W] into flattened RGB patches [B, L, P*P*C]."""

        batch, channels, height, width = images.shape
        p = self.patch_size
        assert channels == self.in_channels, f"Expected {self.in_channels} channels, got {channels}."
        assert height == width == self.image_size, (
            f"Expected square images of size {self.image_size}, got {(height, width)}."
        )
        patches = images.reshape(batch, channels, height // p, p, width // p, p)
        patches = patches.permute(0, 2, 4, 3, 5, 1).reshape(batch, self.num_patches, p * p * channels)
        return patches

    def random_masking(
        self, tokens: torch.Tensor, mask_ratio: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly mask feature tokens, returning visible tokens, mask, and restore ids."""

        batch, length, dim = tokens.shape
        len_keep = int(length * (1.0 - mask_ratio))
        noise = torch.rand(batch, length, device=tokens.device)
        if mask_ratio == 0.0:
            noise = torch.arange(length, device=tokens.device, dtype=tokens.dtype).repeat(batch, 1)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        visible = torch.gather(tokens, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, dim))
        mask = torch.ones(batch, length, device=tokens.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return visible, mask, ids_restore

    def _embed_tokens(self, images: torch.Tensor) -> torch.Tensor:
        assert images.ndim == 4, f"Expected images [B, C, H, W], got {tuple(images.shape)}."
        tokens = self.stem(images)
        assert tokens.shape[1] == self.num_patches, (
            f"Expected {self.num_patches} tokens, got {tokens.shape[1]}. "
            "Check image_size, patch_size, and conv stem stride."
        )
        return tokens + self.pos_embed[:, 1:, :].to(dtype=tokens.dtype)

    def forward_encoder(
        self, images: torch.Tensor, mask_ratio: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode images with optional random token masking."""

        ratio = self.mask_ratio if mask_ratio is None else mask_ratio
        tokens = self._embed_tokens(images)
        tokens, mask, ids_restore = self.random_masking(tokens, ratio)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1).to(dtype=tokens.dtype)
        encoded = torch.cat([cls + self.pos_embed[:, :1, :].to(dtype=tokens.dtype), tokens], dim=1)
        encoded = self.encoder(encoded)
        encoded = self.encoder_norm(encoded)
        z_img = self.latent_proj(encoded[:, 0])
        return encoded, mask, ids_restore, z_img

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Return unmasked visual latents `z_img` for images."""

        _, _, _, z_img = self.forward_encoder(images, mask_ratio=0.0)
        return z_img

    def forward_decoder(self, encoded: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """Decode visible tokens and mask tokens into RGB patch predictions."""

        decoded = self.decoder_embed(encoded)
        batch, _, dim = decoded.shape
        mask_tokens = self.mask_token.expand(batch, ids_restore.shape[1] + 1 - decoded.shape[1], -1)
        tokens = torch.cat([decoded[:, 1:, :], mask_tokens.to(dtype=decoded.dtype)], dim=1)
        tokens = torch.gather(tokens, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, dim))
        decoded = torch.cat([decoded[:, :1, :], tokens], dim=1)
        decoded = decoded + self.decoder_pos_embed.to(device=decoded.device, dtype=decoded.dtype)
        decoded = self.decoder(decoded)
        decoded = self.decoder_norm(decoded)
        return self.decoder_pred(decoded[:, 1:, :])

    def reconstruction_loss(self, images: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """MAE reconstruction loss over masked patches when any are masked."""

        target = self.patchify(images).to(dtype=pred.dtype)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / torch.sqrt(var + 1.0e-6)
        loss = (pred - target).pow(2).mean(dim=-1)
        denom = mask.sum()
        if denom > 0:
            return (loss * mask).sum() / denom.clamp_min(1.0)
        return loss.mean()

    def forward(self, images: torch.Tensor, mask_ratio: float | None = None) -> dict[str, torch.Tensor]:
        """Run masked autoencoding and return loss, prediction, mask, and latent."""

        encoded, mask, ids_restore, z_img = self.forward_encoder(images, mask_ratio)
        pred = self.forward_decoder(encoded, ids_restore)
        loss = self.reconstruction_loss(images, pred, mask)
        return {"loss": loss, "pred": pred, "mask": mask, "z_img": z_img}
