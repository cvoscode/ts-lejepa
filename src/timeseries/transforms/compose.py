from __future__ import annotations

"""Composable transform utilities."""

import random
from typing import Iterable

import torch

from .base import Transform


class Compose(Transform):
    """Apply a sequence of transforms in order."""
    def __init__(self, transforms: Iterable[Transform]):
        self.transforms = list(transforms)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x


class RandomApply(Transform):
    """Apply a transform with probability p."""
    def __init__(self, transform: Transform, p: float = 0.5):
        self.transform = transform
        self.p = float(p)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return self.transform(x)
        return x


class OneOf(Transform):
    """Randomly select one transform to apply."""
    def __init__(self, transforms: Iterable[Transform], p: Iterable[float] | None = None):
        self.transforms = list(transforms)
        if p is None:
            self.p = None
        else:
            probs = list(p)
            if len(probs) != len(self.transforms):
                raise ValueError("p must match transforms length")
            total = sum(probs)
            if total <= 0:
                raise ValueError("p must sum to > 0")
            self.p = [v / total for v in probs]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not self.transforms:
            return x
        if self.p is None:
            t = random.choice(self.transforms)
        else:
            t = random.choices(self.transforms, weights=self.p, k=1)[0]
        return t(x)


__all__ = ["Compose", "RandomApply", "OneOf"]
