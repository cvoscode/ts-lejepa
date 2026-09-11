from __future__ import annotations

"""Datasets and collators that produce SSL view batches."""

from typing import Iterable, Optional

import torch
from torch.utils.data import Dataset

from ..core.types import Batch
from ..transforms.ops import ChannelMixup
from .window_dataset import WindowSample
from .view_builders import AugmentationViewBuilder


class BatchChannelMixup:
    """Holds a :class:`ChannelMixup` instance that is seeded per-batch.

    Channel-mixup replaces random channels in a view with channels from a
    partner window in the same batch. It needs cross-sample access, so it
    runs at the collate level (not inside :class:`AugmentationViewBuilder`).

    The partner pool is taken from the CLEAN t0 view of each sample
    (``batch.views[:, clean_t0_index, ...]``), which is the
    ground-truth t0 of each sample and is shared across all augmented views.
    This avoids a view being mixed with its own augmentation.

    Wire-up:
        mixup = BatchChannelMixup(p=0.3, clean_t0_index=1)
        dataloader = DataLoader(..., collate_fn=lambda items: mixup(collate_batches(items)))
    """

    def __init__(
        self,
        *,
        p: float = 0.3,
        max_channels: Optional[int] = None,
        clean_t0_index: int = 1,
    ) -> None:
        self.mixup = ChannelMixup(p=p, max_channels=max_channels)
        self.clean_t0_index = int(clean_t0_index)

    def __call__(self, batch: Batch) -> Batch:
        if batch.views is None or batch.views.shape[0] < 2:
            return batch
        if batch.views.dim() != 4:
            return batch
        idx = min(self.clean_t0_index, batch.views.shape[1] - 1)
        # Partner pool: [B, C, T] from the clean t0 view of every sample.
        partner_pool = batch.views[:, idx, ...].clone()
        # We mix only the AUGMENTED views; the clean t0 itself is the partner
        # pool source and must stay untouched so the probe keeps a clean
        # ground-truth anchor.
        views = batch.views
        B, V, C, T = views.shape
        # Build a list of view indices to mix (everything except clean t0).
        target_indices = [i for i in range(V) if i != idx]
        if not target_indices:
            return batch
        # Stack the targets into [B*V_mix, C, T] for the mixup op.
        target_views = views[:, target_indices, ...]  # [B, V_mix, C, T]
        # The op handles [B, V, C, T] internally by collapsing V into B.
        self.mixup.set_batch(partner_pool)
        mixed_targets = self.mixup(target_views)
        # Write back.
        for j, i in enumerate(target_indices):
            views[:, i, ...] = mixed_targets[:, j, ...]
        batch.views = views
        return batch


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
            prev_windows=sample.prev_windows,
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
    "BatchChannelMixup",
]
