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

        # Off-diagonal elements should be zero. We divide by D*(D-1)
        # (the number of off-diagonal entries) instead of D: the raw sum
        # of squared off-diagonal covariances scales as D^2, so /D would
        # grow ~D and make this term width-dependent across proj_dim sweeps.
        # /D*(D-1) is O(1) and width-invariant, matching LpWM's _covariance_loss.
        off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
        denom = max(D * (D - 1), 1)
        off_diag = off_diag / denom

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
    "Link",
    "reprelu",
    "RDMReg",
]


# ---------------------------------------------------------------------------
# Sparse-representation building blocks (transferred from LpWM, MIT).
# All signatures and shapes stay identical to the existing module surface;
# these are additive.
# ---------------------------------------------------------------------------


def reprelu(x: torch.Tensor) -> torch.Tensor:
    """Straight-through ``RepReLU``: ReLU on the forward, GELU gradient on backward.

    Forward value equals ``max(x, 0)`` exactly (the detached ReLU/GELU terms
    cancel in value), so hard zeros are preserved. The backward gradient
    uses ``GELU'(x)`` which is nonzero for ``x < 0`` -- a thresholded / zeroed
    coordinate keeps receiving gradient instead of becoming a dead zero
    (as with plain ReLU). Drop-in replacement for ``max(., 0)`` whose zero
    region would otherwise stop learning.
    """
    return torch.relu(x).detach() + F.gelu(x) - F.gelu(x).detach()


class Link(nn.Module):
    """Architectural link function ``h(.)`` selecting the representation space.

    - ``identity``: dense, pairs with the standard Gaussian target (SIGReg / RDMReg p=2).
    - ``relu``: rectified, pairs with the rectified-GG target.
    - ``reprelu``: rectified with GELU-grad backward, pairs with rectified-GG
      target while keeping dead-zero coordinates trainable.

    Identical to LpWM's ``Link`` module. Used by ``LeJEPA_SSL`` to transform
    the projector output before the regularizer matches its distribution.
    """

    def __init__(self, kind: str = "identity") -> None:
        super().__init__()
        if kind not in ("identity", "relu", "reprelu"):
            raise ValueError(f"link {kind!r} not supported")
        self.kind = kind

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "identity":
            return x
        if self.kind == "relu":
            return F.relu(x)
        return reprelu(x)


def _gng_unit_sigma(p: float) -> float:
    """Auto sigma so that zero-mean GN_p has unit variance.

    ``p=2`` -> standard normal (sigma=1); ``p=1`` -> Laplace scale 1/sqrt(2).
    """
    import math
    return math.sqrt(math.gamma(1.0 / p) / math.gamma(3.0 / p)) / (p ** (1.0 / p))


def sample_generalized_gaussian(
    shape: tuple[int, ...],
    p: float = 2.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample zero-mean, unit-variance Generalized Gaussian GN_p.

    - ``p=2`` -> standard normal.
    - ``p=1`` -> Laplace (auto-scaled to unit variance).
    - other ``p>0`` -> sign * (p * Gamma(1/p))^{1/p}, then sigma-scaled.

    Auto sigma chosen so the output has unit variance, matching LpWM's
    ``gng_unit_sigma`` so the link's rectified target has the right scale.
    """
    if p <= 0:
        raise ValueError(f"target_p must be > 0, got {p}")
    sigma = _gng_unit_sigma(p)

    if p == 1.0:
        # Laplace(scale=1) has variance 2; we want var=1, so scale = 1/sqrt(2).
        # Then sigma = gng_unit_sigma(1) = 1/sqrt(2). Net effect: scale = sigma.
        lap = torch.distributions.Laplace(
            torch.tensor(0.0, device=device, dtype=dtype),
            torch.tensor(sigma, device=device, dtype=dtype),
        )
        return lap.sample(shape)

    if p == 2.0:
        return sigma * torch.randn(shape, device=device, dtype=dtype)

    sign = torch.empty(shape, device=device, dtype=dtype).bernoulli_(0.5) * 2 - 1
    g = torch.distributions.Gamma(1.0 / p, 1.0).sample(shape).to(device=device, dtype=dtype)
    gn = sign * (p * g).pow(1.0 / p)
    return sigma * gn


def _sliced_wasserstein_2(
    z: torch.Tensor,
    target: torch.Tensor,
    num_projections: int = 1024,
) -> torch.Tensor:
    """Sliced-Wasserstein-2 distance between ``z`` and ``target``.

    Both inputs share the LAST dim as the support axis. Any leading axes
    are treated as additional batch dims (an independent SWD per slice,
    averaged). Project onto ``num_projections`` random unit directions,
    sort along the sample dim (axis 0), and compute MSE of the resulting
    1-D empirical CDFs. Standard SWD-2 estimator.
    """
    n = z.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)
    d = z.shape[-1]
    proj = torch.randn(num_projections, d, device=z.device, dtype=z.dtype)
    proj = proj / proj.norm(dim=1, keepdim=True).clamp_min(1e-12)
    pz = torch.sort(z @ proj.T, dim=0).values
    pt = torch.sort(target @ proj.T, dim=0).values
    return ((pz - pt) ** 2).mean()


class RDMReg(nn.Module):
    """Rectified Distribution-Matching Regularizer (LpWM-style).

    Matches per-feature distributions to a rectified Generalized Gaussian
    target via sliced-Wasserstein-2 on random 1-D projections. For
    ``target_p=2`` + ``link=identity`` this reduces to a Wasserstein-2
    distance to a standard Gaussian target -- a strict alternative to the
    Epps-Pulley statistic used by ``TemporalSIGReg`` for the same target,
    but with a closed-form-free objective (no characteristic-function
    kernel, no knot resolution).

    For ``p<1`` + ``link in {relu, reprelu}`` this is the LpWM "sparse"
    recipe: GN_p has heavier-than-Gaussian peaks near zero, and the
    rectified link matches that to a sparse, non-negative code. The
    ``link`` argument is supplied externally (typically the model's link
    ``h(.)``), so target = ``h(GN_p + mu)`` is self-consistent with the
    representation space.

    Parameters
    ----------
    feature_dim : int
        Embedding dimension (informational; SWD operates on per-sample vectors).
    target_p : float
        Shape parameter of GN_p. ``p=2`` -> Gaussian, ``p=1`` -> Laplace.
    num_projections : int
        Number of random unit directions used in the SWD-2 estimator.
        SWD-2 is O(1) per ``num_projections``, so a large value (~1024+) is
        cheap and recommended. ``reg_weight`` for RDMReg is roughly 10-100x
        SIGReg's since the SWD-2 magnitude is O(1).
    mu : float
        Target-mean shift; useful with the relu/reprelu link to control
        sparsity (negative mu -> sparser rectified code).
    """

    def __init__(
        self,
        feature_dim: int,
        target_p: float = 1.0,
        num_projections: int = 1024,
        mu: float = 0.0,
    ) -> None:
        super().__init__()
        if target_p <= 0:
            raise ValueError(f"target_p must be > 0, got {target_p}")
        if num_projections < 1:
            raise ValueError(f"num_projections must be >= 1, got {num_projections}")
        self.feature_dim = feature_dim
        self.target_p = float(target_p)
        self.num_projections = int(num_projections)
        self.mu = float(mu)

    def forward(
        self,
        z: torch.Tensor,
        link: nn.Module | None = None,
    ) -> torch.Tensor:
        """Compute RDMReg loss.

        Args:
            z: embeddings of shape ``[N, D]`` or ``[B, T, D]``. Treated as a
                flat sample set (axis 0 = sample axis) for the SWD estimator.
            link: optional ``Link`` module ``h(.)``. Target is computed as
                ``link(GN_p + mu)`` so it lives in the same representation
                space as ``link(z)``. Pass ``None`` for the identity link.

        Returns:
            Scalar SWD-2 loss.
        """
        if z.dim() == 3:
            z = z.reshape(-1, z.shape[-1])

        sample_shape = z.shape  # (N, D)
        base = sample_generalized_gaussian(
            sample_shape, p=self.target_p, device=z.device, dtype=z.dtype
        )
        if self.mu != 0.0:
            base = base + self.mu
        target = base.detach()
        if link is not None:
            target = link(target)

        return _sliced_wasserstein_2(z, target, num_projections=self.num_projections)
