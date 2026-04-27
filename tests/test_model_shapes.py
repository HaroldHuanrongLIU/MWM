import torch

from mwm_torch.models import MWMSurgWMBenchModel, MaskedVisualAutoencoder


def test_masked_autoencoder_shapes():
    model = MaskedVisualAutoencoder(
        image_size=32,
        patch_size=8,
        latent_dim=64,
        hidden_dim=64,
        encoder_depth=1,
        decoder_depth=1,
        num_heads=4,
        decoder_num_heads=4,
        conv_stem_channels=[16, 32],
        mask_ratio=0.5,
    )
    images = torch.rand(2, 3, 32, 32)
    out = model(images)
    assert out["z_img"].shape == (2, 64)
    assert out["pred"].shape == (2, 16, 8 * 8 * 3)
    assert out["mask"].shape == (2, 16)
    assert out["loss"].ndim == 0


def test_mwm_rollout_shapes():
    model = MWMSurgWMBenchModel(
        image_size=32,
        patch_size=8,
        latent_dim=64,
        hidden_dim=64,
        encoder_depth=1,
        decoder_depth=1,
        num_heads=4,
        decoder_num_heads=4,
        conv_stem_channels=[16, 32],
        dynamics_type="gru",
    )
    frames = torch.rand(2, 3, 3, 32, 32)
    z = model.encode_frames(frames)
    assert z.shape == (2, 3, 64)
    coords = torch.rand(2, 3, 2)
    actions = coords[:, 1:] - coords[:, :-1]
    pred_next = model.predict_next_latents(z, coords, actions)
    assert pred_next.shape == (2, 2, 64)
    z_roll, coord_roll = model.rollout(z[:, 0], coords[:, 0], actions)
    assert z_roll.shape == (2, 2, 64)
    assert coord_roll.shape == (2, 2, 2)
