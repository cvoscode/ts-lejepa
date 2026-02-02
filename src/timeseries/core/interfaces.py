from __future__ import annotations

"""Core interfaces for encoders, projectors, and data builders."""

from abc import ABC, abstractmethod
from typing import Optional, Protocol, Literal

import torch


class BaseEncoder(ABC):
    """Encoder interface for time series backbones."""

    output_dim: int
    output_mode: Literal["pooled", "token"]

    @abstractmethod
    def forward(self, x: torch.Tensor, time_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode input windows.

        Expected input: [B, C, T] or [B, C, T, F] depending on encoder.
        Output:
          - pooled: [B, D]
          - token: [B, T, D]
        """
        raise NotImplementedError


class BaseProjector(ABC):
    """Projector interface for pooled or token embeddings."""

    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Project embeddings while preserving structure.

        Input:
          - pooled: [B, D]
          - token: [B, T, D]
        Output:
          - pooled: [B, P]
          - token: [B, T, P]
        """
        raise NotImplementedError


class BaseViewBuilder(ABC):
    """Builds SSL views from a raw window."""

    @abstractmethod
    def build_views(self, window: torch.Tensor, time_features: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return views and aligned time features.

        views: [V, C, T] or [V, C, T, F]
        view_times: [V, T, F_time] or None
        """
        raise NotImplementedError


class BaseWindowDataset(Protocol):
    """Window slicing dataset interface."""

    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return (window, time_features) for SSL view construction."""
        ...


__all__ = [
    "BaseEncoder",
    "BaseProjector",
    "BaseViewBuilder",
    "BaseWindowDataset",
]
