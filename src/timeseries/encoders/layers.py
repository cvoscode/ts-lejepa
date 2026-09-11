import torch
import torch.nn as nn


class SqueezeExcite1d(nn.Module):
    """Channel-wise squeeze & excitation for [B, C, T]."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        pooled = self.avg_pool(x).squeeze(-1)  # [B, C]
        gates = self.mlp(pooled).unsqueeze(-1)  # [B, C, 1]
        return x * gates


class ChannelSelfAttention(nn.Module):
    """Channel-wise self-attention using time-averaged tokens."""

    def __init__(self, channels: int, attn_dim: int = 64, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.proj_in = nn.Linear(1, attn_dim)
        self.attn = nn.MultiheadAttention(attn_dim, heads, dropout=dropout, batch_first=True)
        self.proj_out = nn.Linear(attn_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        x_mean = x.mean(dim=2, keepdim=True)  # [B, C, 1]
        tokens = self.proj_in(x_mean)  # [B, C, attn_dim]
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        gates = torch.sigmoid(self.proj_out(attn_out)).squeeze(-1)  # [B, C]
        return x * gates.unsqueeze(-1)


class ChannelMixer(nn.Module):
    """Optional channel mixing for [B, C, T] tensors.

    Modes:
        - "none": identity
        - "linear": 1x1 convolution across channels
        - "se": squeeze & excitation
        - "attn": channel-wise self-attention
    """

    def __init__(
        self,
        channels: int,
        mode: str = "none",
        *,
        reduction: int = 4,
        attn_dim: int = 64,
        attn_heads: int = 4,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.mode = mode

        if mode == "none":
            self.mixer = nn.Identity()
        elif mode == "linear":
            self.mixer = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=1, bias=False),
                nn.GELU(),
            )
        elif mode == "se":
            self.mixer = SqueezeExcite1d(channels, reduction=reduction)
        elif mode == "attn":
            self.mixer = ChannelSelfAttention(
                channels, attn_dim=attn_dim, heads=attn_heads, dropout=attn_dropout
            )
        else:
            raise ValueError(f"Unknown channel_mixer mode: {mode!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mixer(x)


class TimeFeatureProjector(nn.Module):
    """Project time features [B, T, F] into a target feature dimension.

    Two flavours, controlled by ``mode``:
      - ``"additive"`` (default): Linear -> LayerNorm, returns ``[B, T, output_dim]``
        and is added to the encoder input.
      - ``"film"``: returns ``(scale, shift)`` of shape ``[B, T, output_dim]``
        each so the encoder can do FiLM-style modulation
        ``y = scale * x + shift``.
    """

    def __init__(
        self,
        input_features: int,
        output_dim: int,
        mode: str = "additive",
    ) -> None:
        super().__init__()
        if mode not in {"additive", "film"}:
            raise ValueError(f"Unknown TimeFeatureProjector mode: {mode!r}")
        self.mode = mode
        self.output_dim = int(output_dim)
        out_features = self.output_dim * 2 if mode == "film" else self.output_dim
        self.proj = nn.Linear(int(input_features), out_features)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, time_features: torch.Tensor):
        """Return either ``[B, T, output_dim]`` or a ``(scale, shift)`` pair."""
        out = self.norm(self.proj(time_features))
        if self.mode == "film":
            scale, shift = out.chunk(2, dim=-1)
            # Initialise scale ~ 1 so FiLM starts close to identity.
            scale = 1.0 + scale
            return scale, shift
        return out


class TimeFeatureFiLM(nn.Module):
    """Apply FiLM modulation ``y = scale * x + shift`` with per-time-step gates.

    Used as the consumer of :class:`TimeFeatureProjector` (``mode="film"``)
    outputs. ``scale`` and ``shift`` are both ``[B, T, D]``; ``x`` is either
    ``[B, D, T]`` or ``[B, T, D]`` and the right layout is detected
    automatically.
    """

    def forward(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        # Detect layout.
        if x.dim() == 3 and scale.dim() == 3:
            if x.shape[1] == scale.shape[2]:
                # x: [B, C, T] with C == scale.shape[-1]; treat scale as [B, T, C]
                scale = scale.transpose(1, 2)
                shift = shift.transpose(1, 2)
        return x * scale + shift


class MultiScalePool(nn.Module):
    """Combines average and max pooling with learned fusion."""
    def __init__(self, channels: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fuse = nn.Linear(channels * 2, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T] -> [B, C]"""
        # [B, C, T] -> [B, C, 1] -> [B, C]
        avg = self.avg_pool(x).squeeze(-1)
        max_p = self.max_pool(x).squeeze(-1)
        out = torch.cat([avg, max_p], dim=1)
        out = self.fuse(out)
        out = self.norm(out)
        return out
