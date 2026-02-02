from __future__ import annotations

"""Regularizers for SSL embeddings."""

import torch
import torch.nn as nn


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


__all__ = ["TemporalSIGReg"]
