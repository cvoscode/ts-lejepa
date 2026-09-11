"""Augmentation ops backed by the `augmenttime` package.

See :mod:`timeseries.transforms.augmenttime_ops` for the actual classes.
This module is kept so legacy ``from timeseries.transforms.ops import X``
imports continue to work.
"""

from ..augmenttime_ops import (
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
from .channel_mixup import ChannelMixup

__all__ = [
    "Scaling",
    "Bias",
    "Drift",
    "FeatureJitter",
    "AddGaussianNoise",
    "FrequencyMask",
    "MagnitudeWarp",
    "TemporalBlockMask",
    "Smoothing",
    "TemporalCrop",
    "ChannelMixup",
]
