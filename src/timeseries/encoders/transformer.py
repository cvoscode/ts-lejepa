from __future__ import annotations

import math
from typing import Literal, Optional

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


def build_time_attention_mask(
    T: int,
    mode: Literal["full", "causal", "block_causal"],
    block_size: int = 16,
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    """Build a [T, T] attention mask for the time axis.

    Mirrors the block-causal convention used by LeVJEPA, adapted to 1-D time:
      - "full": no mask (bidirectional across all time steps, baseline).
      - "causal": strictly lower-triangular -- token t sees only t' <= t.
        Streaming-friendly; gives the encoder a temporally ordered state
        for free.
      - "block_causal": bidirectional within a chunk of `block_size`
        consecutive time steps, causal across chunks. Matches the
        locality of physical-world time series and is much cheaper at
        long windows than full attention.

    PyTorch convention (nn.MultiheadAttention / nn.TransformerEncoderLayer
    with `batch_first=True`): positions where the mask is `True` are
    *masked out* (treated as -inf in the additive attention mask). We
    therefore return `True` for positions that must NOT attend.

    For ``"causal"``, callers should pair this mask with
    ``is_causal=True`` inside the attention call: passing only an
    explicit mask (as a bool tensor) routes SDPA into fused kernels
    that are NOT numerically robust to the single-allowed-logit row at
    time-step 0, which produces NaNs after a few hundred training steps
    when query/key norms grow large. The fused causal kernel used when
    ``is_causal=True`` is set handles this boundary row correctly.
    """
    if mode == "full":
        return None
    if T <= 0:
        return None

    if mode == "causal":
        # True on and above the diagonal: token t cannot see t' >= t.
        return torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0
        )

    if mode == "block_causal":
        if block_size <= 0:
            raise ValueError(f"attn_block_size must be > 0, got {block_size}")
        idx = torch.arange(T, device=device)
        same_block = (idx.unsqueeze(0) // block_size) == (
            idx.unsqueeze(1) // block_size
        )
        past_or_same = idx.unsqueeze(0) >= idx.unsqueeze(1)
        allow = same_block | past_or_same
        return ~allow

    raise ValueError(f"Unknown attn_mode: {mode!r}")


class _CausalTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """nn.TransformerEncoderLayer patched to use is_causal=True SDPA.

    nn.TransformerEncoderLayer.forward does not expose is_causal to its
    inner self_attn call. Passing only a bool causal mask routes SDPA
    into a fused kernel path that NaNs on the single-allowed-logit row
    at time-step 0 once query/key norms grow during training. The fix
    (mirroring LeVJEPA's own block_causal attention path) is to also
    pass is_causal=True so the fused causal kernel is selected.
    """

    def __init__(self, *args, use_is_causal: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_is_causal = bool(use_is_causal)

    def forward(
        self,
        src,
        src_mask=None,
        src_key_padding_mask=None,
        is_causal=None,
    ):
        if not self._use_is_causal:
            return super().forward(
                src,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
            )

        # Convert bool mask to additive float -inf, then call self_attn
        # with is_causal=True (the fused causal kernel).
        if src_mask is not None and src_mask.dtype == torch.bool:
            add_mask = torch.zeros_like(
                src_mask, dtype=src.dtype, device=src_mask.device
            ).masked_fill(src_mask, float("-inf"))
        else:
            add_mask = src_mask

        attn_output, _ = self.self_attn(
            src, src, src,
            attn_mask=add_mask,
            need_weights=False,
            is_causal=True,
        )
        attn_output = self.dropout1(attn_output)

        if self.norm_first:
            src = src + attn_output
            src = src + self._ff_block(self.norm2(src))
        else:
            src = self.norm1(src + attn_output)
            src = self.norm2(src + self._ff_block(src))
        return src


class TransformerEncoder(BaseEncoder):
    """Standard Transformer Encoder for time series.

    Time-axis attention modes (see `build_time_attention_mask`):
      - "full" (default): bidirectional attention across all time steps.
      - "causal": strictly causal along T (streaming-friendly).
      - "block_causal": bidirectional within `attn_block_size`-sized
        chunks, causal across chunks. Inductive bias matches the
        locality of physical-world time series.
    """

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
        attn_mode: Literal["full", "causal", "block_causal"] = "full",
        attn_block_size: int = 16,
    ):
        super().__init__(input_channels, output_dim, pool_mode)

        self.attn_mode = attn_mode
        self.attn_block_size = int(attn_block_size)
        # Cached [T, T] mask buffer, lazily built on first forward.
        # We store an empty tensor for "full" mode and rebuild only when
        # T or block_size changes.
        self.register_buffer(
            "_time_mask_cache",
            torch.empty(0, dtype=torch.bool),
            persistent=False,
        )
        self._time_mask_T: int = -1

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
        
        # For strict-causal we need is_causal=True on the SDPA call;
        # the patched subclass above lets us do that. block_causal and
        # full don't need the patch and use the stock layer.
        layer_cls = (
            _CausalTransformerEncoderLayer
            if attn_mode == "causal"
            else nn.TransformerEncoderLayer
        )
        layer_kwargs = dict(
            d_model=output_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        if attn_mode == "causal":
            layer_kwargs["use_is_causal"] = True

        self.layers = nn.ModuleList([
            layer_cls(**layer_kwargs) for _ in range(n_layers)
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

    def _get_time_mask(self, T: int) -> Optional[torch.Tensor]:
        """Return a cached [T, T] bool mask for the current attention
        mode, rebuilding only if the time length changed."""
        if self.attn_mode == "full":
            return None
        if self._time_mask_T != T or self._time_mask_cache.numel() == 0:
            mask = build_time_attention_mask(
                T,
                mode=self.attn_mode,
                block_size=self.attn_block_size,
                device=self.pos_encoder.pe.device,
            )
            self._time_mask_cache = (
                mask
                if mask is not None
                else torch.empty(0, dtype=torch.bool, device=self.pos_encoder.pe.device)
            )
            self._time_mask_T = T
        return (
            self._time_mask_cache
            if self._time_mask_cache.numel() > 0
            else None
        )

    def forward_backbone(self, x: torch.Tensor, time_features: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, C, T] -> [B, T, C]
        x = self.channel_mixer(x)
        x = x.transpose(1, 2)
        x = self.input_proj(x) # [B, T, D]

        x = self._apply_time_features(x, time_features)

        x = self.pos_encoder(x)

        time_mask = self._get_time_mask(x.size(1))
        for layer in self.layers:
            x = layer(x, src_mask=time_mask)
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

        time_mask = self._get_time_mask(x.size(1))
        levels: list[torch.Tensor] = []
        for layer, head in zip(self.layers, self.level_heads):
            x = layer(x, src_mask=time_mask)  # [B, T, D]
            pooled = x.mean(dim=1)  # [B, D]
            levels.append(head(pooled))  # [B, D]
        return levels

    def _apply_time_features(
        self, x: torch.Tensor, time_features: torch.Tensor | None
    ) -> torch.Tensor:
        if time_features is None or self.time_feature_proj is None:
            return x
        return x + self.time_feature_proj(time_features)
