from __future__ import annotations

"""Registry and factory for transform configs."""

from typing import Callable

from .augmenttime_ops import (
    AddGaussianNoise,
    Bias,
    Drift,
    FeatureJitter,
    FrequencyMask,
    MagnitudeWarp,
    Scaling,
    Smoothing,
    TemporalBlockMask,
    TemporalCrop,
)
from .base import Transform
from .compose import Compose, OneOf, RandomApply
from .ops.channel_mixup import ChannelMixup


_REGISTRY: dict[str, Callable[..., Transform]] = {
    "Compose": Compose,
    "RandomApply": RandomApply,
    "OneOf": OneOf,
    "Scaling": Scaling,
    "Bias": Bias,
    "Drift": Drift,
    "FeatureJitter": FeatureJitter,
    "AddGaussianNoise": AddGaussianNoise,
    "FrequencyMask": FrequencyMask,
    "MagnitudeWarp": MagnitudeWarp,
    "TemporalCrop": TemporalCrop,
    "TemporalBlockMask": TemporalBlockMask,
    "Smoothing": Smoothing,
    "ChannelMixup": ChannelMixup,
}


def build_from_config(cfg: dict) -> Transform:
    """Build a transform from a dict (Ray Tune friendly).

    Expected format:
      {"name": "Compose", "transforms": [ ... ]}
    """
    if "name" not in cfg:
        raise ValueError("Transform config must include 'name'")
    name = cfg["name"]
    if name not in _REGISTRY:
        raise ValueError(f"Unknown transform: {name}")

    ctor = _REGISTRY[name]
    kwargs = {k: v for k, v in cfg.items() if k != "name"}

    if name in {"Compose", "OneOf"}:
        transforms_cfg = kwargs.pop("transforms", [])
        transforms = [build_from_config(t) for t in transforms_cfg]
        return ctor(transforms=transforms, **kwargs)
    if name == "RandomApply":
        inner_cfg = kwargs.pop("transform")
        inner = build_from_config(inner_cfg)
        return ctor(transform=inner, **kwargs)

    return ctor(**kwargs)


__all__ = ["build_from_config"]
