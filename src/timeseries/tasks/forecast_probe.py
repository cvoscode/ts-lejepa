"""Forecasting probe head for SSL evaluation (no encoder gradients)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..encoders.covariate import FutureCovariateEncoder


class ForecastProbe(nn.Module):
    """Probe head that maps pooled embeddings to a forecast horizon.

    The probe is intended for evaluation only. Call with detached embeddings
    or wrap usage in `torch.no_grad()`.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        horizon: int,
        output_channels: int,
        use_covariates: bool = False,
        num_time_features: int = 6,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.output_channels = int(output_channels)
        self.use_covariates = bool(use_covariates)
        self.num_time_features = int(num_time_features)

        if self.use_covariates:
            self.future_cov_encoder = FutureCovariateEncoder(
                input_features=self.num_time_features,
                hidden_dim=32,
                output_dim=64,
            )
            forecast_input_dim = input_dim + self.future_cov_encoder.output_dim
        else:
            self.future_cov_encoder = None
            forecast_input_dim = input_dim

        self.head = nn.Sequential(
            nn.LayerNorm(forecast_input_dim),
            nn.Linear(forecast_input_dim, self.output_channels * self.horizon),
        )

    def forward(self, emb: torch.Tensor, future_times: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return forecast tensor shaped [B, C, H]."""
        if self.use_covariates and future_times is not None:
            cov_emb = self.future_cov_encoder(future_times)
            emb = torch.cat([emb, cov_emb], dim=-1)

        out = self.head(emb)
        return out.view(emb.shape[0], self.output_channels, self.horizon)


__all__ = ["ForecastProbe"]
