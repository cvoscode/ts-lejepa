from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.init as init

from ..core.base_encoder import BaseEncoder
from ..preprocessing.time_encoding import NUM_TIME_FEATURES
from .layers import ChannelMixer, TimeFeatureProjector


class LSTMEncoder(BaseEncoder):
    """
    LSTM-based encoder for multivariate time series.
    """
    def __init__(
        self, 
        input_channels: int, 
        output_dim: int, 
        pool_mode: str = "mean",
        hidden_channels: int = 64, 
        num_layers: int = 2,
        dropout: float = 0.2, 
        bidirectional: bool = True,
        channel_mixer: str = "none",
        channel_mixer_reduction: int = 4,
        channel_mixer_attn_dim: int = 64,
        channel_mixer_attn_heads: int = 4,
        channel_mixer_attn_dropout: float = 0.0,
        num_time_features: int = NUM_TIME_FEATURES,
    ):
        super().__init__(input_channels, output_dim, pool_mode)
        
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.bidirectional = bidirectional

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
        
        lstm_out_channels = hidden_channels * (2 if bidirectional else 1)

        # Build per-layer LSTMs for multilevel access
        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            inp_sz = input_channels if i == 0 else lstm_out_channels
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=inp_sz,
                    hidden_size=hidden_channels,
                    num_layers=1,
                    dropout=0.0,
                    bidirectional=bidirectional,
                    batch_first=True,
                )
            )
        self.lstm_dropouts = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(max(0, num_layers - 1))
        ])

        # Per-layer projection heads for forward_multilevel
        self.level_heads = nn.ModuleList([
            nn.Linear(lstm_out_channels, output_dim) for _ in range(num_layers)
        ])

        # Final projection (same dim) for forward_backbone
        self.encoder_head = nn.Linear(lstm_out_channels, output_dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.orthogonal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

    def _run_layers(self, x_seq: torch.Tensor) -> list[torch.Tensor]:
        """Run per-layer LSTMs and return list of per-layer outputs."""
        h = x_seq
        layer_outputs: list[torch.Tensor] = []
        for i, lstm_layer in enumerate(self.lstm_layers):
            h, _ = lstm_layer(h)  # [B, L, hidden*dirs]
            if i < len(self.lstm_dropouts):
                h = self.lstm_dropouts[i](h)
            layer_outputs.append(h)
        return layer_outputs

    def forward_multilevel(self, x: torch.Tensor, time_features: torch.Tensor | None = None) -> list[torch.Tensor]:
        """Return pooled per-layer embeddings for probe fusion.

        Returns list of [B, output_dim] tensors, one per LSTM layer.
        """
        x = self.channel_mixer(x)
        x = self._apply_time_features(x, time_features)
        x_seq = x.transpose(1, 2)
        layer_outputs = self._run_layers(x_seq)
        return [head(out.mean(dim=1)) for head, out in zip(self.level_heads, layer_outputs)]

    def forward_backbone(self, x: torch.Tensor, time_features: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: [B, C, L]
        Returns:
            emb: [B, L, D]
        """
        # LSTM expects [B, L, C]
        x = self.channel_mixer(x)
        x = self._apply_time_features(x, time_features)
        x_seq = x.transpose(1, 2)
        
        # Run through all layers, take final output
        layer_outputs = self._run_layers(x_seq)
        lstm_out = layer_outputs[-1]  # [B, L, hidden*2]
        
        # [B, L, D]
        emb = self.encoder_head(lstm_out)
        
        # BaseEncoder expects [B, T, D] (which is [B, L, D])
        return emb

    def _apply_time_features(
        self, x: torch.Tensor, time_features: torch.Tensor | None
    ) -> torch.Tensor:
        if time_features is None or self.time_feature_proj is None:
            return x
        time_emb = self.time_feature_proj(time_features).transpose(1, 2)
        return x + time_emb


class GRUEncoder(BaseEncoder):
    """
    GRU-based encoder for multivariate time series.
    """
    def __init__(
        self, 
        input_channels: int, 
        output_dim: int, 
        pool_mode: str = "mean",
        hidden_channels: int = 64, 
        num_layers: int = 2,
        dropout: float = 0.2, 
        bidirectional: bool = True,
        channel_mixer: str = "none",
        channel_mixer_reduction: int = 4,
        channel_mixer_attn_dim: int = 64,
        channel_mixer_attn_heads: int = 4,
        channel_mixer_attn_dropout: float = 0.0,
        num_time_features: int = NUM_TIME_FEATURES,
    ):
        super().__init__(input_channels, output_dim, pool_mode)
        
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.bidirectional = bidirectional

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
        
        gru_out_channels = hidden_channels * (2 if bidirectional else 1)

        # Build per-layer GRUs for multilevel access
        self.gru_layers = nn.ModuleList()
        for i in range(num_layers):
            inp_sz = input_channels if i == 0 else gru_out_channels
            self.gru_layers.append(
                nn.GRU(
                    input_size=inp_sz,
                    hidden_size=hidden_channels,
                    num_layers=1,
                    dropout=0.0,
                    bidirectional=bidirectional,
                    batch_first=True,
                )
            )
        self.gru_dropouts = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(max(0, num_layers - 1))
        ])

        # Per-layer projection heads for forward_multilevel
        self.level_heads = nn.ModuleList([
            nn.Linear(gru_out_channels, output_dim) for _ in range(num_layers)
        ])

        self.encoder_head = nn.Linear(gru_out_channels, output_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.orthogonal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

    def _run_layers(self, x_seq: torch.Tensor) -> list[torch.Tensor]:
        """Run per-layer GRUs and return list of per-layer outputs."""
        h = x_seq
        layer_outputs: list[torch.Tensor] = []
        for i, gru_layer in enumerate(self.gru_layers):
            h, _ = gru_layer(h)
            if i < len(self.gru_dropouts):
                h = self.gru_dropouts[i](h)
            layer_outputs.append(h)
        return layer_outputs

    def forward_multilevel(self, x: torch.Tensor, time_features: torch.Tensor | None = None) -> list[torch.Tensor]:
        """Return pooled per-layer embeddings for probe fusion.

        Returns list of [B, output_dim] tensors, one per GRU layer.
        """
        x = self.channel_mixer(x)
        x = self._apply_time_features(x, time_features)
        x_seq = x.transpose(1, 2)
        layer_outputs = self._run_layers(x_seq)
        return [head(out.mean(dim=1)) for head, out in zip(self.level_heads, layer_outputs)]

    def forward_backbone(self, x: torch.Tensor, time_features: torch.Tensor | None = None) -> torch.Tensor:
        x = self.channel_mixer(x)
        x = self._apply_time_features(x, time_features)
        x_seq = x.transpose(1, 2)
        layer_outputs = self._run_layers(x_seq)
        gru_out = layer_outputs[-1]
        emb = self.encoder_head(gru_out)
        return emb

    def _apply_time_features(
        self, x: torch.Tensor, time_features: torch.Tensor | None
    ) -> torch.Tensor:
        if time_features is None or self.time_feature_proj is None:
            return x
        time_emb = self.time_feature_proj(time_features).transpose(1, 2)
        return x + time_emb
