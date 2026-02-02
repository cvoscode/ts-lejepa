from __future__ import annotations

"""Shared data types for time-series SSL."""

from dataclasses import dataclass
from typing import Optional, Literal

import torch


ViewTensor = torch.Tensor


@dataclass(frozen=True)
class InvarianceMix:
    """Configuration for invariance loss view sampling.

    p_t0: proportion of t0 views used in invariance loss.
    p_tminus1: proportion of t-1 views used in invariance loss.
    sample_policy: behavior when available views are fewer than requested.
    """

    p_t0: float = 0.9
    p_tminus1: float = 0.1
    sample_policy: Literal["at_least_one_tminus1", "t0_only_if_missing"] = "at_least_one_tminus1"


@dataclass
class Batch:
    """Unified batch structure for SSL and probe tasks.

    views: [B, V, C, T] or [B, V, C, T, F]
    view_times: optional time covariates aligned to each view
    targets: optional forecast targets [B, H, C]
    future_times: optional time features [B, H, F_time]
    """

    views: ViewTensor
    view_times: Optional[torch.Tensor] = None
    targets: Optional[torch.Tensor] = None
    future_times: Optional[torch.Tensor] = None
    time_index: Optional[torch.Tensor] = None
    future_time_index: Optional[torch.Tensor] = None

    def validate(self) -> None:
        """Validate tensor shapes for views and time features."""
        assert_view_shape(self.views)
        if self.view_times is not None:
            assert_time_shape(self.view_times, self.views)


def assert_view_shape(views: torch.Tensor) -> None:
    """Ensure views follow [B, V, C, T] or [B, V, C, T, F]."""
    if views.dim() not in (4, 5):
        raise ValueError(f"views must be 4D or 5D, got shape {tuple(views.shape)}")
    if views.shape[0] <= 0 or views.shape[1] <= 0:
        raise ValueError("views must have positive batch and view dimensions")


def assert_time_shape(view_times: torch.Tensor, views: torch.Tensor) -> None:
    """Ensure view_times align with views along batch, view, and time dims."""
    if view_times.dim() not in (4, 5):
        raise ValueError(f"view_times must be 4D or 5D, got shape {tuple(view_times.shape)}")
    if view_times.shape[0] != views.shape[0]:
        raise ValueError("view_times batch size must match views")
    if view_times.shape[1] != views.shape[1]:
        raise ValueError("view_times view count must match views")
    if view_times.shape[2] != views.shape[3]:
        raise ValueError("view_times time dimension must match views time dimension")


__all__ = [
    "Batch",
    "InvarianceMix",
    "assert_time_shape",
    "assert_view_shape",
    "ViewTensor",
]
