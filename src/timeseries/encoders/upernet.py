from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from ..core.base_encoder import BaseEncoder
from ..preprocessing.time_encoding import NUM_TIME_FEATURES
from .layers import ChannelMixer, TimeFeatureProjector

# ---------------------------------------------------------------------------
# Norm helpers
# ---------------------------------------------------------------------------

NormType = Literal["layer", "batch", "group"]


def make_norm(channels: int, norm_type: NormType = "batch", num_groups: int = 8) -> nn.Module:
    """Create a normalisation layer that operates on [B, C, T] tensors.

    * ``"batch"`` → ``BatchNorm1d`` (fastest on GPU, no transpose needed).
    * ``"group"`` → ``GroupNorm`` (good for small batches, no transpose needed).
    * ``"layer"`` → ``ChannelLayerNorm`` (original behaviour, 2× transpose).
    """
    if norm_type == "batch":
        return nn.BatchNorm1d(channels)
    if norm_type == "group":
        groups = min(num_groups, channels)
        # channels must be divisible by groups
        while channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    # "layer" — wrap LayerNorm with transposes (backward-compat)
    return _ChannelLayerNorm(channels)


class _ChannelLayerNorm(nn.Module):
    """LayerNorm over channel dimension for [B, C, T] tensors."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


# Keep a public alias so existing imports still work.
ChannelLayerNorm = _ChannelLayerNorm

# ---------------------------------------------------------------------------
# Depthwise-separable convolution
# ---------------------------------------------------------------------------


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise-separable 1-D convolution (depth-wise 3×1 + point-wise 1×1).

    ~``kernel_size`` × fewer FLOPs than a full Conv1d at the same channel width.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


# ---------------------------------------------------------------------------
# Residual conv block
# ---------------------------------------------------------------------------


def _make_conv(
    in_ch: int,
    out_ch: int,
    kernel_size: int,
    stride: int,
    padding: int,
    depthwise_sep: bool,
    bias: bool = False,
) -> nn.Module:
    if depthwise_sep and in_ch == out_ch and stride == 1:
        return DepthwiseSeparableConv1d(in_ch, out_ch, kernel_size, stride, padding, bias)
    return nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias)


class ConvBlock1d(nn.Module):
    """Residual conv block with configurable norm & optional depthwise-separable convs."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        dropout: float = 0.0,
        norm_type: NormType = "batch",
        depthwise_sep: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.norm1 = make_norm(out_channels, norm_type)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Second conv can be depthwise-separable (same in/out channels, stride 1)
        self.conv2 = _make_conv(out_channels, out_channels, 3, 1, 1, depthwise_sep)
        self.norm2 = make_norm(out_channels, norm_type)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                make_norm(out_channels, norm_type),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)

        out = self.act(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.norm2(self.conv2(out))

        return self.act(out + residual)


# ---------------------------------------------------------------------------
# Pyramid Pooling Module
# ---------------------------------------------------------------------------


class PyramidPooling1d(nn.Module):
    """Pyramid Pooling Module for 1D signals with configurable norm & upsample."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_scales: tuple[int, ...] = (1, 2, 4, 8),
        norm_type: NormType = "batch",
        upsample_mode: str = "nearest",
    ) -> None:
        super().__init__()
        self.pool_scales = pool_scales
        self.upsample_mode = upsample_mode
        self.align_corners = None if upsample_mode == "nearest" else False

        self.proj = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            make_norm(out_channels, norm_type),
            nn.GELU(),
        )

        stages: list[nn.Module] = []
        for scale in pool_scales:
            stages.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool1d(scale),
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                    make_norm(out_channels, norm_type),
                    nn.GELU(),
                )
            )
        self.stages = nn.ModuleList(stages)

        concat_channels = out_channels * (len(pool_scales) + 1)
        self.bottleneck = nn.Sequential(
            nn.Conv1d(concat_channels, out_channels, kernel_size=1, bias=False),
            make_norm(out_channels, norm_type),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[-1]
        base = self.proj(x)
        pooled = [base]
        for stage in self.stages:
            out = stage(x)
            out = F.interpolate(out, size=T, mode=self.upsample_mode, align_corners=self.align_corners)
            pooled.append(out)
        return self.bottleneck(torch.cat(pooled, dim=1))


# ---------------------------------------------------------------------------
# UPerNet Encoder
# ---------------------------------------------------------------------------


class UPerNetEncoder(BaseEncoder):
    """UPerNet-style 1D encoder with PPM + FPN for multi-level features.

    Speed-relevant parameters (new):
        norm_type: ``"batch"`` (fastest), ``"group"``, or ``"layer"`` (original).
        depthwise_sep: Use depthwise-separable convolutions in backbone & FPN.
        num_stages: Number of hierarchical stages (2–4). Fewer = faster.
        channel_mult: Width multiplier per stage (default 2.0 = double each stage).
        upsample_mode: ``"nearest"`` (fast) or ``"linear"`` (original).

    All previous parameters remain supported for backward compatibility.
    """

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
        # --- Speed knobs (new) ---
        norm_type: NormType = "batch",
        depthwise_sep: bool = True,
        num_stages: int = 4,
        channel_mult: float = 2.0,
        upsample_mode: str = "nearest",
        # --- Channel mixer (unchanged) ---
        channel_mixer: str = "none",
        channel_mixer_reduction: int = 4,
        channel_mixer_attn_dim: int = 64,
        channel_mixer_attn_heads: int = 4,
        channel_mixer_attn_dropout: float = 0.0,
        num_time_features: int = NUM_TIME_FEATURES,
    ) -> None:
        super().__init__(input_channels, output_dim, pool_mode)

        if not 2 <= num_stages <= 4:
            raise ValueError(f"num_stages must be 2–4, got {num_stages}")

        fpn_dim = output_dim if fpn_dim is None else fpn_dim
        self._num_stages = num_stages
        self._upsample_mode = upsample_mode
        self._align_corners = None if upsample_mode == "nearest" else False

        # --- Channel mixer ---
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

        # --- Stem ---
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, stem_channels, kernel_size=3, padding=1, bias=False),
            make_norm(stem_channels, norm_type),
            nn.GELU(),
        )

        # --- Backbone stages (variable count) ---
        stage_channels: list[int] = [stem_channels]
        for i in range(num_stages):
            ch = stem_channels if i == 0 else int(stage_channels[-1] * channel_mult)
            stage_channels.append(ch)

        self.stages = nn.ModuleList()
        for i in range(num_stages):
            in_ch = stage_channels[i]
            out_ch = stage_channels[i + 1]
            stride = 1 if i == 0 else 2
            self.stages.append(
                ConvBlock1d(
                    in_ch, out_ch, stride=stride, dropout=dropout,
                    norm_type=norm_type, depthwise_sep=depthwise_sep,
                )
            )

        # Keep channel list for lateral connections (skip stem entry)
        self._stage_channels = stage_channels[1:]  # length == num_stages

        # --- PPM on deepest stage ---
        self.ppm = PyramidPooling1d(
            self._stage_channels[-1], fpn_dim,
            pool_scales=pool_scales,
            norm_type=norm_type,
            upsample_mode=upsample_mode,
        )

        # --- FPN lateral + smooth convs ---
        self.laterals = nn.ModuleList([
            nn.Conv1d(ch, fpn_dim, kernel_size=1, bias=False)
            for ch in self._stage_channels
        ])

        self.fpn_convs = nn.ModuleList()
        self.fpn_norms = nn.ModuleList()
        for _ in range(num_stages):
            self.fpn_convs.append(
                _make_conv(fpn_dim, fpn_dim, 3, 1, 1, depthwise_sep)
            )
            self.fpn_norms.append(make_norm(fpn_dim, norm_type))

        self.apply(self._init_weights)

    # ---- helpers ----

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            init.orthogonal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm)):
            if m.weight is not None:
                init.constant_(m.weight, 1)
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def _upsample(self, x: torch.Tensor, size: int) -> torch.Tensor:
        return F.interpolate(x, size=size, mode=self._upsample_mode, align_corners=self._align_corners)

    @staticmethod
    def _pool_level(x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=-1)

    # ---- shared backbone pass ----

    def _backbone_and_fpn(
        self, x: torch.Tensor, time_features: torch.Tensor | None = None
    ) -> list[torch.Tensor]:
        """Run backbone stages → PPM → top-down FPN. Returns FPN feature list."""
        x = self.channel_mixer(x)
        x = self._apply_time_features(x, time_features)
        x = self.stem(x)

        # Collect per-stage features
        feats: list[torch.Tensor] = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)

        # PPM on deepest feature
        p = self.ppm(feats[-1])

        # Top-down FPN pathway
        fpn_feats: list[torch.Tensor | None] = [None] * self._num_stages
        fpn_feats[-1] = p
        for i in range(self._num_stages - 2, -1, -1):
            lat = self.laterals[i](feats[i])
            fpn_feats[i] = lat + self._upsample(fpn_feats[i + 1], feats[i].shape[-1])  # type: ignore[arg-type]

        # Smooth
        for i in range(self._num_stages):
            fpn_feats[i] = self.fpn_norms[i](self.fpn_convs[i](fpn_feats[i]))  # type: ignore[arg-type]

        return fpn_feats  # type: ignore[return-value]

    # ---- public API ----

    def forward_multilevel(self, x: torch.Tensor, time_features: torch.Tensor | None = None) -> list[torch.Tensor]:
        """Return pooled multi-level embeddings for probe fusion."""
        return [self._pool_level(p) for p in self._backbone_and_fpn(x, time_features)]

    def forward_backbone(self, x: torch.Tensor, time_features: torch.Tensor | None = None) -> torch.Tensor:
        """Return finest-level FPN features as [B, T, D]."""
        fpn_feats = self._backbone_and_fpn(x, time_features)
        return fpn_feats[0].transpose(1, 2)

    def _apply_time_features(
        self, x: torch.Tensor, time_features: torch.Tensor | None
    ) -> torch.Tensor:
        if time_features is None or self.time_feature_proj is None:
            return x
        time_emb = self.time_feature_proj(time_features).transpose(1, 2)
        return x + time_emb


__all__ = ["UPerNetEncoder"]
