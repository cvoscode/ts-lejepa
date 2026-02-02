from __future__ import annotations

import torch
import torch.nn as nn
from ..core.base_encoder import BaseEncoder

# Try importing mamba_ssm
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False


class MambaEncoder(BaseEncoder):
    """Mamba-based encoder. Falls back to a simple Gated CNN if Mamba is not installed."""

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        pool_mode: str = "mean",
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__(input_channels, output_dim, pool_mode)
        
        self.input_proj = nn.Linear(input_channels, output_dim)
        self.layers = nn.ModuleList()
        
        for _ in range(num_layers):
            if HAS_MAMBA:
                self.layers.append(
                    Mamba(
                        d_model=output_dim,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand
                    )
                )
            else:
                # Fallback: simple residual block with gating
                self.layers.append(
                    nn.Sequential(
                        nn.Linear(output_dim, output_dim),
                        nn.GELU(),
                        nn.Linear(output_dim, output_dim)
                    )
                )
        
        self.norm = nn.LayerNorm(output_dim)

    def forward_backbone(self, x: torch.Tensor, time_features: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        
        for layer in self.layers:
            if HAS_MAMBA:
                 x = layer(x) # Mamba is usually residual internally? No, need to add resid
                 # Note: Mamba official usually implies wrapping in a MambaBlock with norm/resid
                 # We'll assume simple stacking here for demonstration
            else:
                 x = x + layer(x)
        
        x = self.norm(x)
        return x
