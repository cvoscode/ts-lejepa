from __future__ import annotations

"""View builders for SSL multi-view construction."""

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class AugmentationViewBuilder:
    """Builds multiple augmented views from a window.

    View structure (with num_prev=2, repeat_factor=10):
        [t-1_aug1, t-1_aug2, t0_clean, t0_aug1, t0_aug2, ..., t0_aug10]
        
    Key design:
        1. Previous views are num_prev AUGMENTED copies of the t-1 window only.
           This removes implicit time ordering (t-2, t-3, etc.) from the views,
           reducing the risk of the encoder learning positional shortcuts instead
           of temporal dynamics.
        2. First t0 view is CLEAN (unaugmented) for probe training
        3. Remaining t0 views are augmented for SSL invariance learning
        
    The probe should use index `num_prev` (the clean t0 view).
    SSL invariance uses all views but benefits from clean t0 anchor.
    """

    transform: torch.nn.Module
    repeat_factor: int = 2  # Number of AUGMENTED t0 views
    num_prev: int = 0  # Number of augmented t-1 views to include
    include_clean_t0: bool = True  # Whether to include a clean t0 view for probe
    prev_transform: Optional[torch.nn.Module] = None  # Optional separate transform for prev views

    @property
    def clean_t0_index(self) -> int:
        """Index of the clean t0 view in the view tensor.
        
        Returns num_prev (the first t0 view after all previous views).
        """
        return self.num_prev

    @property
    def total_views(self) -> int:
        """Total number of views generated."""
        base = self.num_prev + self.repeat_factor
        if self.include_clean_t0:
            return base + 1
        return base

    def build_views(
        self,
        window: torch.Tensor,
        time_features: Optional[torch.Tensor] = None,
        *,
        prev_windows: Optional[list[torch.Tensor]] = None,
        prev_time_features: Optional[list[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Generate t0 views and optional augmented t-1 views.

        View order:
            [t-1_aug1, ..., t-1_augN, t0_clean, t0_aug1, ..., t0_augR]
            where N = num_prev, R = repeat_factor
            
        Previous views are all augmented copies of the SAME t-1 window.
        This avoids implicit time encoding from t-2, t-3, etc.
        The clean t0 view is at index `num_prev` for easy probe access.

        Returns:
            views: [V, C, T] where V = num_prev + 1 (clean) + repeat_factor (augmented)
            view_times: [V, T, F_time] or None
        """
        if self.repeat_factor < 1:
            raise ValueError("repeat_factor must be >= 1")

        views_list = []
        view_times_list = []

        # 1. Add num_prev AUGMENTED copies of the t-1 window
        #    Only uses the most recent previous window (t-1) and creates
        #    multiple augmented views of it, removing implicit time ordering.
        if self.num_prev > 0:
            if prev_windows is None or len(prev_windows) < 1:
                raise ValueError(f"num_prev={self.num_prev} requires at least 1 prev_window (t-1)")
            
            # Use only the t-1 window (index 0 in prev_windows list)
            t_minus_1 = prev_windows[0]
            aug_transform = self.prev_transform if self.prev_transform is not None else self.transform
            
            for _ in range(self.num_prev):
                prev_view = aug_transform(t_minus_1.clone().unsqueeze(0)).squeeze(0)  # [C, T]
                views_list.append(prev_view.unsqueeze(0))  # [1, C, T]
                
                if time_features is not None:
                    if prev_time_features is None or len(prev_time_features) < 1:
                        raise ValueError(
                            "num_prev > 0 requires at least 1 prev_time_features "
                            "when time_features is provided"
                        )
                    # All t-1 augmented views share the same t-1 time features
                    prev_times = prev_time_features[0].unsqueeze(0)
                    view_times_list.append(prev_times)
        
        # 2. Add CLEAN t0 view (unaugmented) for probe training
        #    This is the ground truth embedding the probe should learn from
        if self.include_clean_t0:
            clean_t0 = window.clone().unsqueeze(0)  # [1, C, T]
            views_list.append(clean_t0)
            
            if time_features is not None:
                clean_t0_times = time_features.unsqueeze(0)  # [1, T, F]
                view_times_list.append(clean_t0_times)
        
        # 3. Add AUGMENTED t0 views for SSL invariance learning
        #    Apply transforms individually per view to ensure each gets a unique
        #    random augmentation (batch-applying would apply the same random choice
        #    to all views, reducing view diversity).
        if self.repeat_factor > 0:
            t0_augmented = torch.stack([
                self.transform(window.clone().unsqueeze(0)).squeeze(0)
                for _ in range(self.repeat_factor)
            ])  # [R, C, T]
            views_list.append(t0_augmented)
            
            if time_features is not None:
                # Time features are static across augmented views of same window
                t0_times = time_features.unsqueeze(0).repeat(self.repeat_factor, 1, 1)  # [R, T, F]
                view_times_list.append(t0_times)

        # Concatenate all views
        views = torch.cat(views_list, dim=0)
        
        if not view_times_list:
            final_times = None
        else:
            final_times = torch.cat(view_times_list, dim=0)

        return views, final_times


__all__ = [
    "AugmentationViewBuilder",
]
