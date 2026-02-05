from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from ..core.base_encoder import BaseEncoder
from .layers import ChannelMixer


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channel dimension for [B, C, T] tensors."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        x_n = self.norm(x_t)
        return x_n.transpose(1, 2)


class ConvBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = ChannelLayerNorm(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm2 = ChannelLayerNorm(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                ChannelLayerNorm(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + residual
        out = self.act(out)
        return out


class PyramidPooling1d(nn.Module):
    """Pyramid Pooling Module for 1D signals."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_scales: tuple[int, ...] = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        self.pool_scales = pool_scales
        self.proj = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            ChannelLayerNorm(out_channels),
            nn.GELU(),
        )

        stages = []
        for scale in pool_scales:
            stages.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool1d(scale),
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                    ChannelLayerNorm(out_channels),
                    nn.GELU(),
                )
            )
        self.stages = nn.ModuleList(stages)

        concat_channels = out_channels * (len(pool_scales) + 1)
        self.bottleneck = nn.Sequential(
            nn.Conv1d(concat_channels, out_channels, kernel_size=1, bias=False),
            ChannelLayerNorm(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.proj(x)
        pooled = [base]
        for stage in self.stages:
            out = stage(x)
            out = F.interpolate(out, size=x.shape[-1], mode="linear", align_corners=False)
            pooled.append(out)
        x_cat = torch.cat(pooled, dim=1)
        return self.bottleneck(x_cat)


class UPerNetEncoder(BaseEncoder):
    """UPerNet-style 1D encoder with PPM + FPN for multi-level features."""

    def __init__(
        self,
        input_channels: int,
        output_dim: int = 512,
        pool_mode: str = "mean",
        *,
        stem_channels: int = 64,
        fpn_dim: int | None = None,
        dropout: float = 0.1,
        pool_scales: tuple[int, ...] = (1, 2, 4, 8),
        channel_mixer: str = "none",
        channel_mixer_reduction: int = 4,
        channel_mixer_attn_dim: int = 64,
        channel_mixer_attn_heads: int = 4,
        channel_mixer_attn_dropout: float = 0.0,
    ) -> None:
        super().__init__(input_channels, output_dim, pool_mode)
        fpn_dim = output_dim if fpn_dim is None else fpn_dim

        self.channel_mixer = ChannelMixer(
            input_channels,
            mode=channel_mixer,
            reduction=channel_mixer_reduction,
            attn_dim=channel_mixer_attn_dim,
            attn_heads=channel_mixer_attn_heads,
            attn_dropout=channel_mixer_attn_dropout,
        )

        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, stem_channels, kernel_size=3, padding=1, bias=False),
            ChannelLayerNorm(stem_channels),
            nn.GELU(),
        )

        self.stage1 = ConvBlock1d(stem_channels, stem_channels, stride=1, dropout=dropout)
        self.stage2 = ConvBlock1d(stem_channels, stem_channels * 2, stride=2, dropout=dropout)
        self.stage3 = ConvBlock1d(stem_channels * 2, stem_channels * 4, stride=2, dropout=dropout)
        self.stage4 = ConvBlock1d(stem_channels * 4, stem_channels * 8, stride=2, dropout=dropout)

        self.ppm = PyramidPooling1d(stem_channels * 8, fpn_dim, pool_scales=pool_scales)

        self.lateral1 = nn.Conv1d(stem_channels, fpn_dim, kernel_size=1, bias=False)
        self.lateral2 = nn.Conv1d(stem_channels * 2, fpn_dim, kernel_size=1, bias=False)
        self.lateral3 = nn.Conv1d(stem_channels * 4, fpn_dim, kernel_size=1, bias=False)
        self.lateral4 = nn.Conv1d(stem_channels * 8, fpn_dim, kernel_size=1, bias=False)

        self.fpn_conv1 = nn.Conv1d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False)
        self.fpn_conv2 = nn.Conv1d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False)
        self.fpn_conv3 = nn.Conv1d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False)
        self.fpn_conv4 = nn.Conv1d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False)

        self.fpn_norms = nn.ModuleList(
            [
                ChannelLayerNorm(fpn_dim),
                ChannelLayerNorm(fpn_dim),
                ChannelLayerNorm(fpn_dim),
                ChannelLayerNorm(fpn_dim),
            ]
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

    def _upsample(self, x: torch.Tensor, size: int) -> torch.Tensor:
        return F.interpolate(x, size=size, mode="linear", align_corners=False)

    def _pool_level(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=-1)

    def forward_multilevel(self, x: torch.Tensor, time_features: torch.Tensor | None = None) -> list[torch.Tensor]:
        """Return pooled multi-level embeddings for probe fusion."""
        x = self.channel_mixer(x)
        x = self.stem(x)
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)

        p4 = self.ppm(c4)
        p3 = self.lateral3(c3) + self._upsample(p4, c3.shape[-1])
        p2 = self.lateral2(c2) + self._upsample(p3, c2.shape[-1])
        p1 = self.lateral1(c1) + self._upsample(p2, c1.shape[-1])

        p1 = self.fpn_norms[0](self.fpn_conv1(p1))
        p2 = self.fpn_norms[1](self.fpn_conv2(p2))
        p3 = self.fpn_norms[2](self.fpn_conv3(p3))
        p4 = self.fpn_norms[3](self.fpn_conv4(p4))

        return [self._pool_level(p1), self._pool_level(p2), self._pool_level(p3), self._pool_level(p4)]

    def forward_backbone(self, x: torch.Tensor, time_features: torch.Tensor | None = None) -> torch.Tensor:
        x = self.channel_mixer(x)
        x = self.stem(x)
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)

        p4 = self.ppm(c4)
        p3 = self.lateral3(c3) + self._upsample(p4, c3.shape[-1])
        p2 = self.lateral2(c2) + self._upsample(p3, c2.shape[-1])
        p1 = self.lateral1(c1) + self._upsample(p2, c1.shape[-1])

        p1 = self.fpn_norms[0](self.fpn_conv1(p1))
        return p1.transpose(1, 2)


__all__ = ["UPerNetEncoder"]
