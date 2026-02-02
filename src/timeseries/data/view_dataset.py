from __future__ import annotations

"""Datasets and collators that produce SSL view batches."""

from typing import Iterable

import torch
from torch.utils.data import Dataset

from ..core.types import Batch
from .window_dataset import WindowSample
from .view_builders import AugmentationViewBuilder


def collate_batches(items: Iterable[Batch]) -> Batch:
    """Collate a list of Batch items into a single Batch."""
    items = list(items)
    views = torch.stack([item.views for item in items], dim=0)
    view_times = None
    if items and items[0].view_times is not None:
        view_times = torch.stack([item.view_times for item in items], dim=0)
    
    targets = None
    if items and items[0].targets is not None:
        targets = torch.stack([item.targets for item in items], dim=0)
    
    future_times = None
    if items and items[0].future_times is not None:
        future_times = torch.stack([item.future_times for item in items], dim=0)

    time_index = None
    if items and items[0].time_index is not None:
        time_index = torch.stack([item.time_index for item in items], dim=0)

    future_time_index = None
    if items and items[0].future_time_index is not None:
        future_time_index = torch.stack([item.future_time_index for item in items], dim=0)

    result = Batch(
        views=views,
        view_times=view_times,
        targets=targets,
        future_times=future_times,
        time_index=time_index,
        future_time_index=future_time_index,
    )
    result.validate()  # Validate after batching
    return result


class ViewDataset(Dataset):
    """Wraps a WindowDataset and a ViewBuilder to produce SSL batches."""

    def __init__(self, window_dataset: Dataset, view_builder: AugmentationViewBuilder):
        """Wrap a window dataset with a view builder."""
        super().__init__()
        self.window_dataset = window_dataset
        self.view_builder = view_builder

    def __len__(self) -> int:
        """Number of available windows."""
        return len(self.window_dataset)

    def __getitem__(self, idx: int) -> Batch:
        """Return a Batch of views and optional time features (unbatched)."""
        sample: WindowSample = self.window_dataset[idx]
        views, view_times = self.view_builder.build_views(
            sample.window,
            sample.time_features,
            prev_window=sample.prev_window,
            prev_time_features=sample.prev_time_features,
        )
        
        targets = getattr(sample, "targets", None)
        future_times = getattr(sample, "future_times", None)
        time_index = getattr(sample, "time_index", None)
        future_time_index = getattr(sample, "future_time_index", None)

        # Note: views are [V, C, T] here; validation happens after collation
        batch = Batch(
            views=views,
            view_times=view_times,
            targets=targets,
            future_times=future_times,
            time_index=time_index,
            future_time_index=future_time_index,
        )
        return batch


__all__ = [
    "ViewDataset",
    "collate_batches",
]
