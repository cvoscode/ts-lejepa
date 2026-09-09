"""Smoke tests for the augmenttime-backed transform wrappers.

These tests verify that every wrapper in `timeseries.transforms.augmenttime_ops`
runs on a random tensor, preserves shape, returns finite values, and integrates
with the Compose / RandomApply / OneOf pipeline.
"""

from __future__ import annotations

import torch

from timeseries.transforms import (
    AddGaussianNoise,
    Bias,
    Compose,
    FeatureJitter,
    FrequencyMask,
    MagnitudeWarp,
    OneOf,
    RandomApply,
    Scaling,
    Smoothing,
    TemporalBlockMask,
    build_from_config,
)
from timeseries.transforms.augmenttime_ops import _AugmenttimeOp
from timeseries.transforms.base import Transform


def _batch() -> torch.Tensor:
    """A reproducible 3D batch for smoke tests."""
    g = torch.Generator().manual_seed(0)
    return torch.randn(4, 3, 32, generator=g)


def _assert_finite_preserves_shape(out: torch.Tensor, x: torch.Tensor) -> None:
    assert out.shape == x.shape, f"shape changed: {out.shape} != {x.shape}"
    assert torch.is_floating_point(out), f"output not floating-point: {out.dtype}"
    assert torch.isfinite(out).all(), "output contains non-finite values"


def test_scaling_runs() -> None:
    op = Scaling(scale_range=(0.9, 1.1))
    x = _batch()
    out = op(x)
    _assert_finite_preserves_shape(out, x)
    # Scaling should change magnitudes within the requested range.
    assert not torch.allclose(out, x)


def test_bias_runs() -> None:
    op = Bias(bias_std=0.05)
    x = _batch()
    out = op(x)
    _assert_finite_preserves_shape(out, x)
    assert not torch.allclose(out, x)


def test_feature_jitter_runs() -> None:
    op = FeatureJitter(jitter_std=0.05)
    x = _batch()
    out = op(x)
    _assert_finite_preserves_shape(out, x)
    assert not torch.allclose(out, x)


def test_add_gaussian_noise_runs() -> None:
    op = AddGaussianNoise(scale=0.05, p=1.0)  # force apply for determinism
    x = _batch()
    out = op(x)
    _assert_finite_preserves_shape(out, x)
    assert not torch.allclose(out, x)


def test_frequency_mask_runs() -> None:
    op = FrequencyMask(p=1.0, max_freq_ratio=0.1)
    x = _batch()
    out = op(x)
    _assert_finite_preserves_shape(out, x)


def test_magnitude_warp_runs() -> None:
    op = MagnitudeWarp(p=1.0, num_knots=4, noise_scale=0.1)
    x = _batch()
    out = op(x)
    _assert_finite_preserves_shape(out, x)


def test_temporal_block_mask_runs() -> None:
    op = TemporalBlockMask(p=1.0, block_size_ratio=0.1)
    x = _batch()
    out = op(x)
    _assert_finite_preserves_shape(out, x)
    # Augmenttime's WindowMask fills with a uniform random value drawn from
    # the masked window's [min, max] range, so the output must differ from
    # the input in at least some positions.
    assert not torch.allclose(out, x)


def test_smoothing_runs() -> None:
    op = Smoothing(kernel_range=(3, 7), sigma=1.0)
    x = _batch()
    out = op(x)
    _assert_finite_preserves_shape(out, x)


def test_wrapper_accepts_2d_and_3d() -> None:
    """The wrapper base class should handle 2D `[C, T]` inputs as well."""
    op = Scaling(scale_range=(0.9, 1.1))
    x2 = torch.randn(3, 32)
    out2 = op(x2)
    assert out2.shape == x2.shape
    x3 = x2.unsqueeze(0)
    out3 = op(x3)
    assert out3.shape == x3.shape


def test_compose_random_apply_oneof_chain() -> None:
    """The full Compose / RandomApply / OneOf pipeline should run end-to-end."""
    pipeline = Compose([
        RandomApply(Scaling(scale_range=(0.9, 1.1)), p=0.5),
        RandomApply(OneOf([Bias(bias_std=0.05), FeatureJitter(jitter_std=0.05)]), p=0.5),
        TemporalBlockMask(p=0.5),
    ])
    x = _batch()
    out = pipeline(x)
    _assert_finite_preserves_shape(out, x)


def test_build_from_config_registry_routes_to_augmenttime() -> None:
    """The config-driven factory should produce augmenttime-backed transforms."""
    cfg = {
        "name": "Compose",
        "transforms": [
            {"name": "Scaling", "scale_range": (0.9, 1.1)},
            {"name": "Bias", "bias_std": 0.05},
            {"name": "RandomApply", "p": 0.5,
             "transform": {"name": "FeatureJitter", "jitter_std": 0.05}},
        ],
    }
    t = build_from_config(cfg)
    assert isinstance(t, Compose)
    assert isinstance(t.transforms[0], Scaling)
    assert isinstance(t.transforms[1], Bias)
    assert isinstance(t.transforms[2], RandomApply)

    x = _batch()
    out = t(x)
    _assert_finite_preserves_shape(out, x)


def test_all_wrappers_are_transform_subclasses() -> None:
    """Sanity: every wrapper in the module should implement the Transform ABC."""
    from timeseries.transforms import augmenttime_ops as mod
    augmenttime_backed = {
        "Scaling", "Bias", "FeatureJitter", "AddGaussianNoise",
        "FrequencyMask", "MagnitudeWarp", "TemporalBlockMask", "Smoothing",
    }
    for name in mod.__all__:
        cls = getattr(mod, name)
        assert issubclass(cls, Transform), f"{name} is not a Transform"
        if name in augmenttime_backed:
            assert issubclass(cls, _AugmenttimeOp), (
                f"{name} should be an _AugmenttimeOp subclass"
            )