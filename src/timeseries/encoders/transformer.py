from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.init as init

from ..core.base_encoder import BaseEncoder
from ..preprocessing.time_encoding import NUM_TIME_FEATURES
from .layers import ChannelMixer, TimeFeatureProjector


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:T, :].unsqueeze(0)


class TransformerEncoder(BaseEncoder):
    """Standard Transformer Encoder for time series."""

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        pool_mode: str = "mean",
        n_heads: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        channel_mixer: str = "none",
        channel_mixer_reduction: int = 4,
        channel_mixer_attn_dim: int = 64,
        channel_mixer_attn_heads: int = 4,
        channel_mixer_attn_dropout: float = 0.0,
        num_time_features: int = NUM_TIME_FEATURES,
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
        self.time_feature_proj = (
            TimeFeatureProjector(int(num_time_features), output_dim)
            if int(num_time_features) > 0
            else None
        )
        self.pos_encoder = PositionalEncoding(output_dim)
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=output_dim,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        ])
        self.n_layers = n_layers
        
        # Per-layer projection heads for forward_multilevel
        self.level_heads = nn.ModuleList([
            nn.Linear(output_dim, output_dim) for _ in range(n_layers)
        ])
        
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
        x = self.input_proj(x) # [B, T, D]

        x = self._apply_time_features(x, time_features)

        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x)
        return x

    def forward_multilevel(self, x: torch.Tensor, time_features: torch.Tensor | None = None) -> list[torch.Tensor]:
        """Return pooled per-layer embeddings for probe fusion.

        Returns list of [B, output_dim] tensors, one per transformer layer.
        """
        x = self.channel_mixer(x)
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self._apply_time_features(x, time_features)
        x = self.pos_encoder(x)
        levels: list[torch.Tensor] = []
        for layer, head in zip(self.layers, self.level_heads):
            x = layer(x)  # [B, T, D]
            pooled = x.mean(dim=1)  # [B, D]
            levels.append(head(pooled))  # [B, D]
        return levels

    def _apply_time_features(
        self, x: torch.Tensor, time_features: torch.Tensor | None
    ) -> torch.Tensor:
        if time_features is None or self.time_feature_proj is None:
            return x
        return x + self.time_feature_proj(time_features)
