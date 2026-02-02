from __future__ import annotations

"""LeJEPA-style SSL core for time-series representations.

This module is encoder-agnostic and computes invariance and SIGReg losses
from multi-view inputs.
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


class LeJEPA_SSL(nn.Module):
    """Encoder-agnostic SSL module using LeJEPA-style invariance + SIGReg.

    Expects views shaped [B, V, C, T] (or [B, V, C, T, F]).
    View convention when t-1 is included: index 0 is t-1, the rest are t0 views.
    """

    def __init__(
        self,
        *,
        encoder: nn.Module,
        projector: nn.Module,
        proj_dim: int,
        lamb: float = 0.5,
        invariance_mix: Optional[InvarianceMix] = None,
        has_prev_view: bool = True,
        regularizer: Optional[nn.Module] = None,
        # Legacy args for backward compatibility (used if regularizer is None)
        sigreg_slices: int = 1024,
        sigreg_knots: int = 17,
        sigreg_seed: int = 0,
    ) -> None:
        """Initialize SSL core.

        Args:
            encoder: Backbone that encodes windows into pooled or token embeddings.
            projector: Projection head applied to embeddings.
            proj_dim: Projected feature dimension used by SIGReg.
            lamb: Weight for SIGReg vs invariance loss.
            invariance_mix: Config for t0 vs t-1 invariance view selection.
            has_prev_view: Whether the first view (index 0) is a t-1 view.
            regularizer: Optional injection of regularizer module.
            sigreg_slices: Number of random projection slices (used if regularizer is None).
            sigreg_knots: Number of knots for char function (used if regularizer is None).
            sigreg_seed: RNG seed (used if regularizer is None).
        """
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.lamb = float(lamb)
        self.invariance_mix = invariance_mix or InvarianceMix()
        self.has_prev_view = has_prev_view

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

    def _Select_Invariance_Views_Optimized(self, proj: torch.Tensor) -> torch.Tensor:
        """Optimized selection of invariance views."""
        V = proj.shape[1]
        
        if not self.has_prev_view:
             t0_count = V
             n_t0 = max(1, int(round(self.invariance_mix.p_t0 * t0_count)))
             n_t0 = min(n_t0, t0_count)
             
             if n_t0 >= t0_count:
                 return proj
             
             # Random selection without replacement
             # To improve speed on repeated calls, we could use a fixed permutation if deterministic behavior is desired per-step, 
             # but random is needed for SSL.
             sel = torch.randperm(t0_count, device=proj.device)[:n_t0]
             return proj[:, sel, ...]

        if V <= 1:
             return proj

        t0_count = V - 1
        n_t0 = max(1, int(round(self.invariance_mix.p_t0 * t0_count)))
        n_t0 = min(n_t0, t0_count)

        # Generate t0 indices (shifting by 1 because index 0 is t-1)
        t0_sel = torch.randperm(t0_count, device=proj.device)[:n_t0] + 1
        
        include_tminus1 = self.invariance_mix.p_tminus1 > 0
        if include_tminus1 and self.invariance_mix.sample_policy == "at_least_one_tminus1":
            # Prepend 0
            sel = torch.cat([torch.tensor([0], device=proj.device), t0_sel])
        else:
            sel = t0_sel

        return proj[:, sel, ...]

    def _select_invariance_views(self, proj: torch.Tensor) -> torch.Tensor:
        # Wrapper to keep forward method clean
        return self._Select_Invariance_Views_Optimized(proj)

    def forward(self, views: torch.Tensor, view_times: Optional[torch.Tensor] = None, *, global_step: int | None = None) -> SSLBatchResult:
        """Compute invariance + SIGReg losses and return projections."""
        emb = self._encode(views, view_times)
        proj = self._project(emb)

        inv_proj = self._select_invariance_views(proj)
        if inv_proj.dim() == 4:
            inv_pool = inv_proj.mean(dim=2)
        else:
            inv_pool = inv_proj

        inv_mean = inv_pool.mean(dim=1, keepdim=True)
        inv_loss = (inv_pool - inv_mean).square().mean()

        if proj.dim() == 4:
            z_sig = proj.view(-1, proj.shape[2], proj.shape[3])
        else:
            z_sig = proj.view(-1, proj.shape[-1])
        sigreg_loss = self.sigreg(z_sig, global_step=global_step)

        total_loss = (1 - self.lamb) * inv_loss + self.lamb * sigreg_loss
        return SSLBatchResult(
            inv_loss=inv_loss,
            sigreg_loss=sigreg_loss,
            total_loss=total_loss,
            proj=proj,
        )


__all__ = ["LeJEPA_SSL", "SSLBatchResult"]
