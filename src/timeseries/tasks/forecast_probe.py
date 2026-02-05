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
    
    Improved design for reduced overfitting:
    - Dropout for regularization
    - Optional multi-layer MLP for higher capacity
    - Multi-level feature fusion (for pyramid encoders)
    - Configurable architecture
    """

    def __init__(
        self,
        *,
        input_dim: int,
        horizon: int,
        output_channels: int,
        use_covariates: bool = False,
        num_time_features: int = 6,
        hidden_dim: Optional[int] = None,  # If None, direct projection
        num_hidden_layers: int = 2,
        dropout: float = 0.3,  # Regularization
        use_residual: bool = False,  # Residual connection if hidden_dim matches
        level_fusion: str = "attn",  # How to fuse multi-level features
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.output_channels = int(output_channels)
        self.use_covariates = bool(use_covariates)
        self.num_time_features = int(num_time_features)
        self.dropout_rate = float(dropout)
        self.use_residual = bool(use_residual)
        self.level_fusion = str(level_fusion)

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

        self.input_norm = nn.LayerNorm(forecast_input_dim)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.level_attn = None
        if self.level_fusion == "attn":
            hidden_attn = max(8, forecast_input_dim // 4)
            self.level_attn = nn.Sequential(
                nn.Linear(forecast_input_dim, hidden_attn),
                nn.GELU(),
                nn.Linear(hidden_attn, 1),
            )
        
        if hidden_dim is not None:
            layers: list[nn.Module] = []
            in_dim = forecast_input_dim
            for _ in range(max(1, int(num_hidden_layers))):
                layers.extend(
                    [
                        nn.Linear(in_dim, hidden_dim),
                        nn.GELU(),
                        nn.LayerNorm(hidden_dim),
                        nn.Dropout(self.dropout_rate),
                    ]
                )
                in_dim = hidden_dim
            self.hidden = nn.Sequential(*layers)
            self.head = nn.Linear(hidden_dim, self.output_channels * self.horizon)
            self._hidden_dim = hidden_dim
        else:
            # Direct projection (simpler, but add dropout)
            self.hidden = None
            self.head = nn.Linear(forecast_input_dim, self.output_channels * self.horizon)
            self._hidden_dim = None

    def _fuse_levels(self, emb_list: list[torch.Tensor]) -> torch.Tensor:
        pooled: list[torch.Tensor] = []
        for emb in emb_list:
            if emb.dim() == 3:
                emb = emb.mean(dim=1)
            pooled.append(emb)
        stacked = torch.stack(pooled, dim=1)  # [B, L, D]

        if self.level_attn is not None:
            weights = self.level_attn(stacked).squeeze(-1)  # [B, L]
            weights = torch.softmax(weights, dim=1).unsqueeze(-1)
            return (stacked * weights).sum(dim=1)

        return stacked.mean(dim=1)

    def forward(self, emb: torch.Tensor | list[torch.Tensor], future_times: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return forecast tensor shaped [B, C, H]."""
        if isinstance(emb, (list, tuple)):
            emb = self._fuse_levels(list(emb))

        if self.use_covariates and future_times is not None:
            cov_emb = self.future_cov_encoder(future_times)
            emb = torch.cat([emb, cov_emb], dim=-1)

        x = self.input_norm(emb)
        x = self.dropout(x)
        
        if self.hidden is not None:
            x = self.hidden(x)
        
        out = self.head(x)
        return out.view(emb.shape[0], self.output_channels, self.horizon)


__all__ = ["ForecastProbe"]
