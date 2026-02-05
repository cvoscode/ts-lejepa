from __future__ import annotations

"""Regularizers for SSL embeddings."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RegularizerResult:
    """Container for regularizer outputs and diagnostics."""
    loss: torch.Tensor
    # Diagnostic metrics (for monitoring, not gradient)
    embedding_std: Optional[torch.Tensor] = None  # Mean std across features
    embedding_cov_off_diag: Optional[torch.Tensor] = None  # Off-diagonal covariance magnitude
    feature_collapse_ratio: Optional[torch.Tensor] = None  # Ratio of near-zero variance features


class CovarianceRegularizer(nn.Module):
    """VICReg-style covariance regularizer to decorrelate features.
    
    Encourages the off-diagonal elements of the covariance matrix to be zero,
    preventing feature collapse where all features become correlated.
    """
    
    def __init__(self, feature_dim: int, eps: float = 1e-4):
        super().__init__()
        self.feature_dim = feature_dim
        self.eps = eps
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute covariance regularization loss.
        
        Args:
            z: [N, D] or [B, T, D] embeddings
            
        Returns:
            Scalar covariance loss
        """
        if z.dim() == 3:
            z = z.reshape(-1, z.shape[-1])
        
        N, D = z.shape
        if N < 2:
            return torch.tensor(0.0, device=z.device, dtype=z.dtype)
        
        # Center the embeddings
        z_centered = z - z.mean(dim=0, keepdim=True)
        
        # Compute covariance matrix
        cov = (z_centered.T @ z_centered) / (N - 1)
        
        # Off-diagonal elements should be zero
        off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
        off_diag = off_diag / D  # Normalize by feature dim
        
        return off_diag


class VarianceRegularizer(nn.Module):
    """VICReg-style variance regularizer to prevent collapse.
    
    Encourages the variance of each feature to be above a threshold,
    preventing dimensional collapse.
    """
    
    def __init__(self, feature_dim: int, target_std: float = 1.0, eps: float = 1e-4):
        super().__init__()
        self.feature_dim = feature_dim
        self.target_std = target_std
        self.eps = eps
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute variance regularization loss.
        
        Args:
            z: [N, D] or [B, T, D] embeddings
            
        Returns:
            Scalar variance loss
        """
        if z.dim() == 3:
            z = z.reshape(-1, z.shape[-1])
        
        # Compute per-feature std
        std = z.std(dim=0)
        
        # Hinge loss: penalize when std < target
        var_loss = F.relu(self.target_std - std).mean()
        
        return var_loss


class CombinedRegularizer(nn.Module):
    """Combined SIGReg + Covariance + Variance regularization.
    
    Provides multiple complementary regularization signals:
    1. SIGReg: Encourages isotropic Gaussian marginals
    2. Covariance: Decorrelates features  
    3. Variance: Prevents dimensional collapse
    
    Also computes diagnostic metrics for monitoring.
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_slices: int = 1024,
        knots: int = 17,
        seed: int = 0,
        # Weights for combined loss
        sigreg_weight: float = 1.0,
        cov_weight: float = 0.04,  # VICReg default
        var_weight: float = 0.0,  # Often covered by SIGReg
        target_std: float = 1.0,
    ):
        super().__init__()
        self.sigreg = TemporalSIGReg(
            feature_dim=feature_dim,
            num_slices=num_slices,
            knots=knots,
            seed=seed,
        )
        self.cov_reg = CovarianceRegularizer(feature_dim)
        self.var_reg = VarianceRegularizer(feature_dim, target_std=target_std)
        
        self.sigreg_weight = sigreg_weight
        self.cov_weight = cov_weight
        self.var_weight = var_weight
    
    def forward(
        self, z: torch.Tensor, global_step: int | None = None
    ) -> RegularizerResult:
        """Compute combined regularization with diagnostics."""
        if z.dim() == 3:
            z_flat = z.reshape(-1, z.shape[-1])
        else:
            z_flat = z
        
        # Compute component losses
        sigreg_loss = self.sigreg(z, global_step=global_step)
        cov_loss = self.cov_reg(z_flat)
        var_loss = self.var_reg(z_flat) if self.var_weight > 0 else torch.tensor(0.0, device=z.device)
        
        total_loss = (
            self.sigreg_weight * sigreg_loss +
            self.cov_weight * cov_loss +
            self.var_weight * var_loss
        )
        
        # Compute diagnostics (detached, for monitoring only)
        with torch.no_grad():
            std = z_flat.std(dim=0)
            embedding_std = std.mean()
            
            # Covariance off-diagonal magnitude
            z_centered = z_flat - z_flat.mean(dim=0, keepdim=True)
            cov = (z_centered.T @ z_centered) / (z_flat.shape[0] - 1 + 1e-8)
            off_diag_cov = (cov.pow(2).sum() - cov.diagonal().pow(2).sum()).sqrt() / z_flat.shape[-1]
            
            # Feature collapse ratio (features with std < 0.1)
            collapse_ratio = (std < 0.1).float().mean()
        
        return RegularizerResult(
            loss=total_loss,
            embedding_std=embedding_std,
            embedding_cov_off_diag=off_diag_cov,
            feature_collapse_ratio=collapse_ratio,
        )


class TemporalSIGReg(nn.Module):
    """Vectorized SIGReg adapted for temporal embeddings.

    - input z: [N_samples, D] or [B, T, D]
    - the test projects onto `num_slices` random directions (1D slices)
    - returns scalar loss (mean across slices)
    """

    def __init__(self, feature_dim: int, num_slices: int = 1024, knots: int = 17, A_dim: int = 256, seed: int = 0):
        """Initialize SIGReg.

        Args:
            feature_dim: Embedding dimension.
            num_slices: Number of random projection directions.
            knots: Number of time knots for characteristic function integration.
            A_dim: Reserved for compatibility; not used directly.
            seed: RNG seed for projection sampling.
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_slices = num_slices
        self.knots = knots
        self.A_dim = A_dim
        self.seed = seed

        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def _sample_A(self, *, device: torch.device, dtype: torch.dtype, global_step: int | None):
        """Sample normalized random projection directions."""
        rng = torch.Generator(device=device)
        if global_step is None:
            rng.manual_seed(self.seed)
        else:
            rng.manual_seed(int(self.seed) + int(global_step))

        A = torch.randn(self.feature_dim, self.num_slices, generator=rng, device=device, dtype=dtype)
        A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-12)
        return A

    def forward(self, z: torch.Tensor, global_step: int | None = None) -> torch.Tensor:
        """Compute SIGReg loss for pooled or token embeddings."""
        if z.dim() == 3:
            B, T, D = z.shape
            z_flat = z.reshape(-1, D)
        else:
            z_flat = z

        A = self._sample_A(device=z_flat.device, dtype=z_flat.dtype, global_step=global_step)
        proj = z_flat @ A

        x_t = proj.unsqueeze(-1) * self.t.view(1, 1, -1)
        cos_mean = x_t.cos().mean(dim=0)
        sin_mean = x_t.sin().mean(dim=0)
        err = (cos_mean - self.phi.view(1, -1)).square() + sin_mean.square()
        per_slice = err @ self.weights
        return per_slice.mean()


__all__ = [
    "TemporalSIGReg",
    "CovarianceRegularizer",
    "VarianceRegularizer", 
    "CombinedRegularizer",
    "RegularizerResult",
]
