from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.init as init

from ..core.base_encoder import BaseEncoder
from .layers import ChannelMixer


class ResidualBlock1d(nn.Module):
    """
    Ein Residual-Block mit 1D-Faltungen, Batch-Norm und ReLU.
    Verhindert das Verschwinden von Gradienten bei tieferen Netzen.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.LayerNorm(out_channels)
        
        # Shortcut anpassen, falls sich Dimensionen ändern (z.B. durch Stride oder Kanaländerung)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.LayerNorm(out_channels)
            )

    def forward(self, x):
        residual = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        return out

class MultiScalePool(nn.Module):
    """Combines average and max pooling with learned fusion."""
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Linear(channels * 2, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        """x: [B, C, T] -> [B, C]"""
        avg = self.avg_pool(x).squeeze(-1)  # [B, C]
        max_p = self.max_pool(x).squeeze(-1)  # [B, C]
        combined = torch.cat([avg, max_p], dim=1)  # [B, 2C]
        out = self.linear(combined)  # [B, C]
        out = self.norm(out)  # [B, C]
        return out

class TCNEncoder(BaseEncoder):
    """Temporal CNN encoder for multivariate time series.
    
    Uses strided convolutions to progressively downsample the temporal dimension,
    then returns per-timestep embeddings.
    """

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        pool_mode: str = "mean",
        hidden_channels: int = 64,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.2,
        channel_mixer: str = "none",
        channel_mixer_reduction: int = 4,
        channel_mixer_attn_dim: int = 64,
        channel_mixer_attn_heads: int = 4,
        channel_mixer_attn_dropout: float = 0.0,
    ):
        super().__init__(input_channels, output_dim, pool_mode)
        
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.channel_mixer = ChannelMixer(
            input_channels,
            mode=channel_mixer,
            reduction=channel_mixer_reduction,
            attn_dim=channel_mixer_attn_dim,
            attn_heads=channel_mixer_attn_heads,
            attn_dropout=channel_mixer_attn_dropout,
        )
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.LayerNorm(hidden_channels),
            nn.ReLU()
        )
        
        # Residual blocks with stride=1 to preserve temporal dimension
        layers = []
        in_ch = hidden_channels
        out_ch = hidden_channels * 2
        for i in range(num_layers):
            layers.append(ResidualBlock1d(in_ch, out_ch, kernel_size=kernel_size, stride=1, dropout=dropout))
            in_ch = out_ch
        self.backbone = nn.Sequential(*layers)
        
        # Project to output dimension per timestep
        self.encoder_head = nn.Sequential(
            nn.Linear(in_ch, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.orthogonal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

    def forward_backbone(self, x: torch.Tensor, time_features: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: [B, C, L]
        Returns:
            emb: [B, L, D]
        """
        # x: [B, C, L]
        x = self.channel_mixer(x)
        x = self.stem(x)      # [B, hidden_channels, L]
        x = self.backbone(x)  # [B, hidden_channels*2^num_layers, L]
        
        # Transpose to [B, L, C] for linear projection
        x = x.transpose(1, 2)  # [B, L, C]
        emb = self.encoder_head(x)  # [B, L, D]
        
        return emb