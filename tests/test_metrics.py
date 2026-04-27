import math

import torch

from mwm_torch.metrics import discrete_frechet, hausdorff_distance, trajectory_metrics


def test_basic_trajectory_metrics_are_zero_for_identical_paths():
    pred = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
    target = pred.clone()
    mask = torch.tensor([[True, True, True]])
    metrics = trajectory_metrics(pred, target, mask, horizons=[1, 3], prefix="sparse_")
    assert metrics["sparse_ade"] == 0.0
    assert metrics["sparse_fde"] == 0.0
    assert metrics["sparse_frechet"] == 0.0
    assert metrics["sparse_hausdorff"] == 0.0
    assert metrics["sparse_horizon_3_error"] == 0.0


def test_frechet_and_hausdorff_known_offset():
    a = torch.tensor([[0.0, 0.0], [1.0, 0.0]]).numpy()
    b = torch.tensor([[0.0, 1.0], [1.0, 1.0]]).numpy()
    assert math.isclose(discrete_frechet(a, b), 1.0)
    assert math.isclose(hausdorff_distance(a, b), 1.0)


def test_pixel_metrics_scale_coordinates():
    pred = torch.tensor([[[0.5, 0.5]]])
    target = torch.tensor([[[0.0, 0.0]]])
    mask = torch.tensor([[True]])
    metrics = trajectory_metrics(pred, target, mask, pixel_scale=torch.tensor([[20, 10]]))
    assert math.isclose(metrics["ade"], math.sqrt(0.5))
    assert math.isclose(metrics["pixel_ade"], math.sqrt(10**2 + 5**2))
