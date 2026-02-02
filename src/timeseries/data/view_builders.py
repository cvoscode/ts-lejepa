from __future__ import annotations

"""View builders for SSL multi-view construction."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class AugmentationViewBuilder:
    """Builds multiple augmented views from a window.

    - t0 views are repeated `repeat_factor` times.
    - optionally include a t-1 view if provided by the dataset.
    """

    transform: torch.nn.Module
    repeat_factor: int = 2
    include_prev: bool = False
    prev_transform: Optional[torch.nn.Module] = None

    def build_views(
        self,
        window: torch.Tensor,
        time_features: Optional[torch.Tensor] = None,
        *,
        prev_window: Optional[torch.Tensor] = None,
        prev_time_features: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Generate t0 views and optional t-1 view.

        Returns:
            views: [V, C, T]
            view_times: [V, T, F_time] or None
        """
        if self.repeat_factor < 1:
            raise ValueError("repeat_factor must be >= 1")

        # Optimization: Batch processing for t0 views
        # Create batch of windows [Repeat, C, T] first
        # We need independent augmentations, so we clone to have distinct memory if transform is in-place
        # But even if not in-place, we need a batch.
        # expand() creates views. If transform modifies in-place, all would change.
        # But 'Scaling', 'Drift' create new tensors.
        # However, to be safe and robust (e.g. Cutout might be in-place), we clone.
        # Cloning a batch is faster than cloning N times in loop.
        
        # [C, T] -> [1, C, T] -> [R, C, T]
        t0_batch = window.unsqueeze(0).repeat(self.repeat_factor, 1, 1).clone()
        
        # Apply transform to the batch
        # Note: self.transform must handle batched input [B, C, T]
        t0_processed = self.transform(t0_batch)
        
        # If transform returns [R, C, T], we are good.
        views_list = [t0_processed] 
        # Wait, we need to handle prev_window
        
        view_times_list = []
        if time_features is not None:
            # Time features are usually static across views of same window
            # [R, T, F]
            t0_times = time_features.unsqueeze(0).repeat(self.repeat_factor, 1, 1)
            view_times_list.append(t0_times)

        if self.include_prev:
            if prev_window is None:
                raise ValueError("include_prev=True requires prev_window")
            prev_t = self.prev_transform if self.prev_transform is not None else self.transform
            
            # prev_window: [C, T] -> [1, C, T] for batch consistency
            prev_batch = prev_window.clone().unsqueeze(0)
            prev_view = prev_t(prev_batch)
            
            views_list.insert(0, prev_view)
            
            if time_features is not None:
                if prev_time_features is None:
                    raise ValueError("include_prev=True requires prev_time_features when time_features is provided")
                # [1, T, F]
                prev_times = prev_time_features.unsqueeze(0)
                view_times_list.insert(0, prev_times)

        # Concatenate all batches
        views = torch.cat(views_list, dim=0)
        
        if not view_times_list:
            final_times = None
        else:
            final_times = torch.cat(view_times_list, dim=0)

        return views, final_times


__all__ = [
    "AugmentationViewBuilder",
]
