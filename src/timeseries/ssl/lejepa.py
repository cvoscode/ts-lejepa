from __future__ import annotations

"""LeJEPA-style SSL core for time-series representations.

This module is encoder-agnostic and computes invariance and SIGReg losses
from multi-view inputs.

View Structure (with num_prev_views=2, repeat_factor=10):
    [t-1_aug1, t-1_aug2, t0_clean, t0_aug1, t0_aug2, ..., t0_aug10]
    
    - Indices 0 to num_prev_views-1: Augmented copies of t-1 window
    - Index num_prev_views: Clean t0 (unaugmented, used as anchor)
    - Indices num_prev_views+1 onwards: Augmented t0 views
    
Key Improvements:
    1. Clean t0 anchor-based invariance
    2. Embedding diagnostics: Monitor std and collapse ratio
    3. Previous views are augmented copies of t-1 only (no t-2, t-3, etc.)
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ..core.types import InvarianceMix
from .regularizers import TemporalSIGReg


@dataclass
class SSLBatchResult:
    """Container for SSL outputs and losses."""
    inv_loss: torch.Tensor
    sigreg_loss: torch.Tensor
    total_loss: torch.Tensor
    proj: torch.Tensor
    # Additional metrics for monitoring
    temporal_alignment: Optional[torch.Tensor] = None  # How well prev views align with t0
    # Embedding health diagnostics
    embedding_std: Optional[torch.Tensor] = None
    feature_collapse_ratio: Optional[torch.Tensor] = None


class LeJEPA_SSL(nn.Module):
    """Encoder-agnostic SSL module using LeJEPA-style invariance + SIGReg.

    Expects views shaped [B, V, C, T] (or [B, V, C, T, F]).
    
    View convention:
        [t-1_aug1, ..., t-1_augN, t0_clean, t0_aug1, ..., t0_augR]
        - Indices 0..num_prev_views-1: Augmented copies of t-1 window
        - Index num_prev_views: Clean t0 view (anchor for probe and SSL)
        - Indices num_prev_views+1..V: Augmented t0 views for invariance learning
        
    Key SSL Objectives:
        1. Invariance: Augmented t0 views should have similar embeddings to clean t0
        2. Regularization: SIGReg to prevent collapse
    """

    def __init__(
        self,
        *,
        encoder: nn.Module,
        projector: nn.Module,
        proj_dim: int,
        lamb: float = 0.5,
        invariance_mix: Optional[InvarianceMix] = None,
        num_prev_views: int = 1,
        include_clean_t0: bool = True,  # Whether view structure includes clean t0
        regularizer: Optional[nn.Module] = None,
        # Legacy args for backward compatibility (used if regularizer is None)
        sigreg_slices: int = 1024,
        sigreg_knots: int = 17,
        sigreg_seed: int = 0,
        # Improved invariance options
        use_anchor_invariance: bool = True,  # Use clean t0 as anchor instead of mean
    ) -> None:
        """Initialize SSL core.

        Args:
            encoder: Backbone that encodes windows into pooled or token embeddings.
            projector: Projection head applied to embeddings.
            proj_dim: Projected feature dimension used by SIGReg.
            lamb: Weight for SIGReg vs invariance loss.
            invariance_mix: Config for t0 vs previous views invariance view selection.
            num_prev_views: Number of previous views (0=none, 1=t-1, 2=t-1,t-2, etc.).
            include_clean_t0: Whether the view structure includes a clean t0 view.
            regularizer: Optional injection of regularizer module.
            sigreg_slices: Number of random projection slices (used if regularizer is None).
            sigreg_knots: Number of knots for char function (used if regularizer is None).
            sigreg_seed: RNG seed (used if regularizer is None).
            use_anchor_invariance: If True, use clean t0 as anchor instead of mean.
        """
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.lamb = float(lamb)
        self.invariance_mix = invariance_mix or InvarianceMix()
        self.num_prev_views = int(num_prev_views)
        self.include_clean_t0 = bool(include_clean_t0)
        self.use_anchor_invariance = bool(use_anchor_invariance)
        
        # Index of the clean t0 view (first t0 view after previous views)
        self.clean_t0_index = self.num_prev_views
        
        if regularizer is not None:
            self.sigreg = regularizer
        else:
            self.sigreg = TemporalSIGReg(
                feature_dim=proj_dim,
                num_slices=sigreg_slices,
                knots=sigreg_knots,
                seed=sigreg_seed,
            )

    def _encode(self, views: torch.Tensor, view_times: Optional[torch.Tensor]) -> torch.Tensor:
        """Encode batched views with optional time features."""
        B, V = views.shape[:2]
        flat_views = views.view(B * V, *views.shape[2:])
        
        if view_times is None:
            emb = self.encoder(flat_views)
        else:
            flat_times = view_times.view(B * V, *view_times.shape[2:])
            # Explicit check instead of try-except for performance
            # Assuming encoder handles args if user provided view_times. 
            # If not, we fallback gracefully only if we know it fails. 
            # But try-except is safer for "agnostic" design. 
            # However, we can optimize by checking attribute once? 
            # For now, keeping logic but ensuring variable names are clean.
            try:
                emb = self.encoder(flat_views, flat_times)
            except TypeError:
                emb = self.encoder(flat_views)

        if emb.dim() == 2:
            return emb.view(B, V, -1)
        return emb.view(B, V, emb.shape[1], emb.shape[2])

    def _project(self, emb: torch.Tensor) -> torch.Tensor:
        """Project pooled or token embeddings while preserving structure."""
        if emb.dim() == 3:
            B, V, D = emb.shape
            z = self.projector(emb.view(B * V, D))
            return z.view(B, V, -1)
        B, V, T, D = emb.shape
        z = self.projector(emb.reshape(-1, D))
        return z.view(B, V, T, -1)

    def _Select_Invariance_Views_Optimized(self, proj: torch.Tensor) -> tuple[torch.Tensor, Optional[int]]:
        """Optimized selection of invariance views.
        
        Views are ordered as: [t-N, ..., t-2, t-1, t0_clean, t0_aug1, ..., t0_augR]
        where N = num_prev_views.
        
        Returns:
            selected_proj: Selected projections for invariance
            anchor_index: Index of anchor (clean t0) in selected views, or None
        """
        V = proj.shape[1]
        
        if self.num_prev_views == 0 and not self.include_clean_t0:
            # All views are t0 augmentations (legacy mode)
            t0_count = V
            n_t0 = max(1, int(round(self.invariance_mix.p_t0 * t0_count)))
            n_t0 = min(n_t0, t0_count)
            
            if n_t0 >= t0_count:
                return proj, None
            
            sel = torch.randperm(t0_count, device=proj.device)[:n_t0]
            return proj[:, sel, ...], None

        # Calculate where t0 views start
        t0_start = self.num_prev_views
        if self.include_clean_t0:
            t0_start += 1  # clean_t0 is at num_prev_views, augmented start after
        
        if V <= t0_start:
            # Only previous views (and maybe clean t0), no augmented views
            return proj, self.num_prev_views if self.include_clean_t0 else None

        # Number of augmented t0 views
        aug_t0_count = V - t0_start
        n_aug = max(1, int(round(self.invariance_mix.p_t0 * aug_t0_count)))
        n_aug = min(n_aug, aug_t0_count)

        # Generate augmented t0 indices
        aug_sel = torch.randperm(aug_t0_count, device=proj.device)[:n_aug] + t0_start
        
        # Build selection list
        sel_list = []
        anchor_index = None
        
        # Optionally include previous views
        include_prev = self.invariance_mix.p_tminus1 > 0
        if include_prev and self.invariance_mix.sample_policy == "at_least_one_tminus1":
            prev_indices = torch.arange(self.num_prev_views, device=proj.device)
            sel_list.append(prev_indices)
        
        # Always include clean t0 as anchor if available
        if self.include_clean_t0:
            clean_idx = torch.tensor([self.num_prev_views], device=proj.device)
            anchor_index = len(sel_list[0]) if sel_list else 0
            sel_list.append(clean_idx)
        
        # Add selected augmented views
        sel_list.append(aug_sel)
        
        sel = torch.cat(sel_list) if len(sel_list) > 1 else sel_list[0]
        return proj[:, sel, ...], anchor_index

    def _select_invariance_views(self, proj: torch.Tensor) -> tuple[torch.Tensor, Optional[int]]:
        """Wrapper to keep forward method clean."""
        return self._Select_Invariance_Views_Optimized(proj)
    
    def _compute_invariance_loss(
        self, 
        inv_pool: torch.Tensor, 
        anchor_index: Optional[int]
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute invariance loss with optional anchor-based formulation.
        
        Args:
            inv_pool: [B, V_sel, D] pooled projections of selected views
            anchor_index: Index of clean t0 anchor in selected views, or None
            
        Returns:
            inv_loss: Invariance loss
            temporal_alignment: Optional metric for prev->t0 alignment
        """
        temporal_alignment = None
        
        if self.use_anchor_invariance and anchor_index is not None:
            # Anchor-based invariance: all views should match the clean t0 anchor
            # This is more principled than mean-based when we have a ground truth
            anchor = inv_pool[:, anchor_index:anchor_index+1, :]  # [B, 1, D]
            inv_loss = (inv_pool - anchor).square().mean()
            
            # Compute temporal alignment metric (how well prev views align with t0)
            if self.num_prev_views > 0:
                prev_views = inv_pool[:, :self.num_prev_views, :]  # [B, num_prev, D]
                temporal_alignment = (prev_views - anchor).square().mean()
        else:
            # Mean-based invariance (original behavior)
            inv_mean = inv_pool.mean(dim=1, keepdim=True)
            inv_loss = (inv_pool - inv_mean).square().mean()
        
        return inv_loss, temporal_alignment
    
    def forward(self, views: torch.Tensor, view_times: Optional[torch.Tensor] = None, *, global_step: int | None = None) -> SSLBatchResult:
        """Compute SSL losses: invariance + regularization."""
        emb = self._encode(views, view_times)
        proj = self._project(emb)

        # 1. Invariance loss (augmented views should match clean t0)
        inv_proj, anchor_index = self._select_invariance_views(proj)
        if inv_proj.dim() == 4:
            inv_pool = inv_proj.mean(dim=2)
        else:
            inv_pool = inv_proj

        inv_loss, temporal_alignment = self._compute_invariance_loss(inv_pool, anchor_index)

        # 2. SIGReg regularization
        if proj.dim() == 4:
            z_sig = proj.view(-1, proj.shape[2], proj.shape[3])
        else:
            z_sig = proj.view(-1, proj.shape[-1])
        sigreg_loss = self.sigreg(z_sig, global_step=global_step)
        
        # Compute embedding diagnostics (for monitoring collapse)
        with torch.no_grad():
            z_flat = z_sig.reshape(-1, z_sig.shape[-1]) if z_sig.dim() > 2 else z_sig
            embedding_std = z_flat.std(dim=0).mean()
            feature_collapse_ratio = (z_flat.std(dim=0) < 0.1).float().mean()

        # Combine losses
        # Base: invariance + regularization
        total_loss = (1 - self.lamb) * inv_loss + self.lamb * sigreg_loss
        
        return SSLBatchResult(
            inv_loss=inv_loss,
            sigreg_loss=sigreg_loss,
            total_loss=total_loss,
            proj=proj,
            temporal_alignment=temporal_alignment,
            embedding_std=embedding_std,
            feature_collapse_ratio=feature_collapse_ratio,
        )


__all__ = ["LeJEPA_SSL", "SSLBatchResult"]
