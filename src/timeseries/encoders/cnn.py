from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.init as init

from ..core.base_encoder import BaseEncoder
from ..preprocessing.time_encoding import NUM_TIME_FEATURES
from .layers import MultiScalePool, ChannelMixer, TimeFeatureProjector


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channel dimension for [B, C, T] tensors."""
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] -> [B, T, C], normalize last dim, -> [B, C, T]
        x_t = x.transpose(1, 2)
        x_n = self.norm(x_t)
        return x_n.transpose(1, 2)


class DilatedResidualBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve length")

        padding = (kernel_size - 1) // 2 * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = ChannelLayerNorm(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = ChannelLayerNorm(out_channels)

        self.downsample = nn.Identity()
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                ChannelLayerNorm(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        out = self.act(out)
        return out


class CNNEncoder(BaseEncoder):
    """Temporal CNN encoder for multivariate time series.

    Expected input: x [B, C, L]
    """

    def __init__(
        self,
        input_channels: int,
        output_dim: int = 512,
        pool_mode: str = "mean",
        *,
        stem_channels: int = 128,
        kernel_size: int = 5,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
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

        self.time_feature_proj = (
            TimeFeatureProjector(int(num_time_features), input_channels)
            if int(num_time_features) > 0
            else None
        )

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        stem_pad = (kernel_size - 1) // 2
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, stem_channels, kernel_size=kernel_size, padding=stem_pad, bias=False),
            ChannelLayerNorm(stem_channels),
            nn.GELU(),
        )

        blocks: list[nn.Module] = []
        self._level_channels: list[int] = []
        in_ch = stem_channels
        for i, d in enumerate(dilations):
            out_ch = stem_channels if i < len(dilations) - 1 else stem_channels * 2
            blocks.append(
                DilatedResidualBlock1d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=3,
                    dilation=d,
                    dropout=dropout,
                )
            )
            self._level_channels.append(out_ch)
            in_ch = out_ch
        self.backbone = nn.ModuleList(blocks)

        # Per-level projection heads for forward_multilevel
        self.level_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(ch, output_dim), nn.LayerNorm(output_dim))
            for ch in self._level_channels
        ])

        # Final head (same as last level head) for forward_backbone
        self.head = nn.Sequential(
            nn.Linear(in_ch, output_dim),
            nn.LayerNorm(output_dim),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            init.orthogonal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.LayerNorm):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

    def forward_multilevel(self, x: torch.Tensor, time_features: torch.Tensor | None = None) -> list[torch.Tensor]:
        """Return pooled per-level embeddings for probe fusion.

        Returns list of [B, output_dim] tensors, one per dilated block.
        """
        x = self.channel_mixer(x)
        x = self._apply_time_features(x, time_features)
        x = self.stem(x)
        levels: list[torch.Tensor] = []
        for block, head in zip(self.backbone, self.level_heads):
            x = block(x)  # [B, C_i, T]
            pooled = x.mean(dim=-1)  # [B, C_i]
            levels.append(head(pooled))  # [B, output_dim]
        return levels

    def forward_backbone(self, x: torch.Tensor, time_features: torch.Tensor | None = None) -> torch.Tensor:
        # x shape: [Batch, Channels, Time]
        x = self.channel_mixer(x)
        x = self._apply_time_features(x, time_features)
        x = self.stem(x)
        for block in self.backbone:
            x = block(x)
        
        # Project channel dim to output_dim
        # [B, C, T] -> [B, T, C]
        x_t = x.transpose(1, 2)
        x_proj = self.head(x_t) # [B, T, D]
        return x_proj

    def _apply_time_features(
        self, x: torch.Tensor, time_features: torch.Tensor | None
    ) -> torch.Tensor:
        if time_features is None or self.time_feature_proj is None:
            return x
        time_emb = self.time_feature_proj(time_features).transpose(1, 2)
        return x + time_emb
