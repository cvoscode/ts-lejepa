from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.init as init

from ..core.base_encoder import BaseEncoder


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
        bidirectional: bool = True
    ):
        super().__init__(input_channels, output_dim, pool_mode)
        
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_channels,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        lstm_out_channels = hidden_channels * (2 if bidirectional else 1)
        
        # Project LSTM output to target dimension
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

    def forward_backbone(self, x: torch.Tensor, time_features: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: [B, C, L]
        Returns:
            emb: [B, L, D]
        """
        # LSTM expects [B, L, C]
        x_seq = x.transpose(1, 2)
        
        # [B, L, hidden*2]
        lstm_out, _ = self.lstm(x_seq)
        
        # [B, L, D]
        emb = self.encoder_head(lstm_out)
        
        # BaseEncoder expects [B, T, D] (which is [B, L, D])
        return emb


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
        bidirectional: bool = True
    ):
        super().__init__(input_channels, output_dim, pool_mode)
        
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.gru = nn.GRU(
            input_size=input_channels,
            hidden_size=hidden_channels,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        gru_out_channels = hidden_channels * (2 if bidirectional else 1)
        
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

    def forward_backbone(self, x: torch.Tensor, time_features: torch.Tensor | None = None) -> torch.Tensor:
        x_seq = x.transpose(1, 2)
        gru_out, _ = self.gru(x_seq)
        emb = self.encoder_head(gru_out)
        return emb

# Alias for backward compatibility if needed, but discouraged
LSTMEncoder
GRUEncoder
