from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.init as init
from torch_geometric_temporal.nn.recurrent import GConvLSTM

from ..core.base_encoder import BaseEncoder
from .layers import ChannelMixer

# ============================================================
# Graph-LSTM encoder using BaseEncoder pattern
# ============================================================


class GraphLSTMEncoder(BaseEncoder):
    """
    Graph-LSTM encoder for spatiotemporal time series.
    
    Processes spatial graphs evolving over time using Graph Convolutional LSTM.
    Expects edge_index to be provided at initialization or via additional kwargs.
    
    Input:
        x: [B, C, T] where C is number of nodes
    
    Output:
        [B, T, D] temporal embeddings
    """
    def __init__(self,
                 input_channels: int,  # Number of nodes
                 output_dim: int,
                 pool_mode: str = "mean",
                 gnn_hidden_channels: int = 64,
                 K: int = 2,
                 dropout: float = 0.0,
                 edge_index: torch.Tensor | None = None,
                 edge_weight: torch.Tensor | None = None,
                 channel_mixer: str = "none",
                 channel_mixer_reduction: int = 4,
                 channel_mixer_attn_dim: int = 64,
                 channel_mixer_attn_heads: int = 4,
                 channel_mixer_attn_dropout: float = 0.0):
        super().__init__(input_channels, output_dim, pool_mode)

        self.gnn_hidden_channels = gnn_hidden_channels
        self.K = K

        self.channel_mixer = ChannelMixer(
            input_channels,
            mode=channel_mixer,
            reduction=channel_mixer_reduction,
            attn_dim=channel_mixer_attn_dim,
            attn_heads=channel_mixer_attn_heads,
            attn_dropout=channel_mixer_attn_dropout,
        )

        # Store graph structure on the module
        if edge_index is not None:
            self.register_buffer("edge_index", edge_index)
        else:
            self.edge_index = None
        if edge_weight is not None:
            self.register_buffer("edge_weight", edge_weight)
        else:
            self.edge_weight = None

        # Graph convolutional LSTM
        self.gconv_lstm = GConvLSTM(
            in_channels=1,  # Scalar features per node
            out_channels=gnn_hidden_channels,
            K=K
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Project from graph embedding to output dimension
        self.encoder_head = nn.Sequential(
            nn.Linear(gnn_hidden_channels, output_dim),
            nn.LayerNorm(output_dim)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.orthogonal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

    def forward_backbone(self, x: torch.Tensor, time_features: torch.Tensor | None = None, 
                        edge_index: torch.Tensor | None = None,
                        edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] where C is number of nodes
            time_features: Optional time features (unused)
            edge_index: Graph edge indices (optional if set at init)
            edge_weight: Graph edge weights (optional if set at init)
        Returns:
            emb: [B, T, D] temporal embeddings per timestep
        """
        if edge_index is None:
            edge_index = getattr(self, "edge_index", None)
        if edge_weight is None:
            edge_weight = getattr(self, "edge_weight", None)
        if edge_index is None:
            raise ValueError("GraphLSTMEncoder requires edge_index (pass at init or forward(..., edge_index=...)).")

        # x: [B, C, T]
        x = self.channel_mixer(x)
        B, C, T = x.shape
        
        # Reformat to [B, T, C, 1] for graph processing
        x = x.permute(0, 2, 1).unsqueeze(-1)  # [B, T, C, 1]
        
        # Process each timestep through graph LSTM
        h, c = None, None
        outputs = []
        
        for t in range(T):
            x_t = x[:, t]  # [B, C, 1]
            x_t = x_t.reshape(B * C, 1)  # [B*C, 1]
            
            if h is not None:
                h = h.view(B * C, self.gnn_hidden_channels)
                c = c.view(B * C, self.gnn_hidden_channels)
            
            h, c = self.gconv_lstm(x_t, edge_index, edge_weight, h, c)
            h = self.dropout(h)
            
            # Average over nodes for this timestep
            h_t = h.view(B, C, self.gnn_hidden_channels).mean(dim=1)  # [B, gnn_hidden]
            outputs.append(h_t)
        
        # Stack temporal outputs: [B, T, gnn_hidden]
        output = torch.stack(outputs, dim=1)
        
        # Project to output dimension
        emb = self.encoder_head(output)  # [B, T, D]
        
        return emb

