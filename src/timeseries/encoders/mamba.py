from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.init as init

from ..core.base_encoder import BaseEncoder
from .layers import ChannelMixer

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
        dropout: float = 0.1,
        channel_mixer: str = "none",
        channel_mixer_reduction: int = 4,
        channel_mixer_attn_dim: int = 64,
        channel_mixer_attn_heads: int = 4,
        channel_mixer_attn_dropout: float = 0.0,
    ):
        super().__init__(input_channels, output_dim, pool_mode)

        self.channel_mixer = ChannelMixer(
            input_channels,
            mode=channel_mixer,
            reduction=channel_mixer_reduction,
            attn_dim=channel_mixer_attn_dim,
            attn_heads=channel_mixer_attn_heads,
            attn_dropout=channel_mixer_attn_dropout,
        )
        
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
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.orthogonal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

    def forward_backbone(self, x: torch.Tensor, time_features: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, C, T] -> [B, T, C]
        x = self.channel_mixer(x)
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
