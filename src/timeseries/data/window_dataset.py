from __future__ import annotations

"""Window slicing datasets for SSL view construction."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class WindowSample:
    """Container for a single window sample and optional t-1 context."""
    window: torch.Tensor
    time_features: Optional[torch.Tensor] = None
    prev_window: Optional[torch.Tensor] = None
    prev_time_features: Optional[torch.Tensor] = None
    targets: Optional[torch.Tensor] = None
    future_times: Optional[torch.Tensor] = None
    time_index: Optional[torch.Tensor] = None
    future_time_index: Optional[torch.Tensor] = None


class WindowDataset(Dataset):
    """Slices raw time series into past-only windows.

    data: [C, T]
    time_features: [T, F_time] (optional)
    """

    def __init__(
        self,
        data: torch.Tensor,
        time_features: Optional[torch.Tensor],
        *,
        window_size: int,
        stride: int = 1,
        include_prev: bool = False,
        prev_shift: Optional[int] = None,
        sensors: Optional[list[int]] = None,
        horizon: int = 0,
    ) -> None:
        """Create a window dataset from raw [C, T] data.

        Args:
            data: Sensor data with shape [C, T].
            time_features: Optional time covariates [T, F_time].
            window_size: Window length.
            stride: Step between window starts.
            include_prev: Whether to also expose a t-1 window.
            prev_shift: Shift between t0 and t-1 windows.
            sensors: Optional subset of sensors.
            horizon: Forecast horizon. If > 0, returns targets and future_times.
        """
        super().__init__()
        if data.dim() != 2:
            raise ValueError(f"data must be [C, T], got shape {tuple(data.shape)}")
        self.data = data
        self.time_features = time_features
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.include_prev = bool(include_prev)
        self.prev_shift = int(prev_shift) if prev_shift is not None else 0
        self.sensors = sensors if sensors is not None else list(range(self.data.shape[0]))
        self.horizon = int(horizon)

        if self.include_prev and self.prev_shift <= 0:
            raise ValueError("prev_shift must be > 0 when include_prev is True")

        total_shift = self.prev_shift if self.include_prev else 0
        max_start = self.data.shape[1] - self.window_size - self.horizon
        self.num_windows = (max_start - total_shift) // self.stride + 1
        self.num_windows = max(0, self.num_windows)

    def __len__(self) -> int:
        """Number of windows available."""
        return self.num_windows

    def __getitem__(self, idx: int) -> WindowSample:
        """Return a window and optional previous window for invariance mixing."""
        base_start = (self.prev_shift if self.include_prev else 0) + idx * self.stride

        window = self.data[self.sensors, base_start : base_start + self.window_size]
        time_feats = (
            None
            if self.time_features is None
            else self.time_features[base_start : base_start + self.window_size]
        )

        prev_window = None
        prev_time = None
        if self.include_prev:
            prev_start = base_start - self.prev_shift
            prev_window = self.data[self.sensors, prev_start : prev_start + self.window_size]
            if self.time_features is not None:
                prev_time = self.time_features[prev_start : prev_start + self.window_size]
        
        targets = None
        future_times = None
        time_index = torch.tensor(base_start, dtype=torch.long)
        future_time_index = None
        if self.horizon > 0:
            target_start = base_start + self.window_size
            targets = self.data[self.sensors, target_start : target_start + self.horizon].t() # [H, C]
            if self.time_features is not None:
                future_times = self.time_features[target_start : target_start + self.horizon]
            future_time_index = torch.arange(
                target_start, target_start + self.horizon, dtype=torch.long
            )

        return WindowSample(
            window=window,
            time_features=time_feats,
            prev_window=prev_window,
            prev_time_features=prev_time,
            targets=targets,
            future_times=future_times,
            time_index=time_index,
            future_time_index=future_time_index,
        )


__all__ = [
    "WindowDataset",
    "WindowSample",
]
