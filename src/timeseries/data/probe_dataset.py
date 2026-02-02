"""Datasets for forecast probe evaluation (no encoder gradients)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ProbeSample:
    """Single sample for forecasting probe evaluation."""

    window: torch.Tensor
    target: torch.Tensor
    future_times: Optional[torch.Tensor] = None


class ForecastProbeDataset(Dataset):
    """Creates past-only windows with future targets for probe evaluation.

    data: [C, T]
    time_features: [T, F_time] (optional)
    """

    def __init__(
        self,
        data: torch.Tensor,
        time_features: Optional[torch.Tensor],
        *,
        window_size: int,
        horizon: int,
        stride: int = 1,
        sensors: Optional[list[int]] = None,
    ) -> None:
        super().__init__()
        if data.dim() != 2:
            raise ValueError(f"data must be [C, T], got shape {tuple(data.shape)}")
        self.data = data
        self.time_features = time_features
        self.window_size = int(window_size)
        self.horizon = int(horizon)
        self.stride = int(stride)
        self.sensors = sensors if sensors is not None else list(range(self.data.shape[0]))

        max_start = self.data.shape[1] - (self.window_size + self.horizon)
        self.num_windows = max(0, (max_start // self.stride) + 1)

    def __len__(self) -> int:
        """Number of available probe windows."""
        return self.num_windows

    def __getitem__(self, idx: int) -> ProbeSample:
        """Return a probe window and its future target.

        target is returned as [H, C] to match existing convention.
        """
        start = idx * self.stride
        window = self.data[self.sensors, start : start + self.window_size]
        target_start = start + self.window_size
        target = self.data[self.sensors, target_start : target_start + self.horizon].t()

        future_times = None
        if self.time_features is not None:
            future_times = self.time_features[target_start : target_start + self.horizon]

        return ProbeSample(window=window, target=target, future_times=future_times)


__all__ = ["ForecastProbeDataset", "ProbeSample"]
