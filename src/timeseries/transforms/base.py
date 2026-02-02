from __future__ import annotations

"""Base class for time-series transforms."""

from abc import ABC, abstractmethod

import torch


class Transform(ABC):
    """Base class for time-series transforms."""

    def train(self, mode: bool = True) -> "Transform":
        """Set training mode (kept for API parity)."""
        return self

    def eval(self) -> "Transform":
        """Set eval mode (kept for API parity)."""
        return self.train(False)

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transform to input tensor."""
        raise NotImplementedError


__all__ = ["Transform"]
