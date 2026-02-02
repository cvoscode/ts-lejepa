from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Literal

import torch
import torch.nn as nn


class BaseEncoder(nn.Module, ABC):
    """Abstract base class for all time-series encoders.

    Ensures consistent input/output shapes for SSL and forecasting tasks.
    """

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        pool_mode: Literal["mean", "max", "last", "none"] = "mean",
    ):
        """
        Args:
            input_channels: Number of input variables/sensors.
            output_dim: Dimension of the output embedding.
            pool_mode: How to pool the temporal dimension.
                       'none' returns [B, T, D].
                       'mean', 'max', 'last' return [B, D].
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.pool_mode = pool_mode

    @abstractmethod
    def forward_backbone(self, x: torch.Tensor, time_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the backbone layers.

        Args:
            x: Input tensor [B, C, T].
            time_features: Optional time covariates [B, T, F_time].

        Returns:
            Representation [B, D_model, T] or [B, T, D_model] depending on architecture.
            We recommend returning strictly [B, T, D_model] for consistency before pooling.
        """
        pass

    def forward(self, x: torch.Tensor, time_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Standard forward pass with pooling options.

        Args:
            x: [B, C, T]
            time_features: [B, T, F_time]

        Returns:
            Encoded representation.
            If pool_mode == 'none': [B, T, output_dim]
            Else: [B, output_dim]
        """
        # [B, T, D_model]
        seq_emb = self.forward_backbone(x, time_features)
        
        # Ensure consistent shape [B, T, D] if backbone returns [B, D, T]? 
        # We rely on subclasses to return [B, T, D] or we enforce it here if identifiable.
        # For now, we assume subclasses return [B, T, D].
        
        if self.pool_mode == "none":
            return seq_emb

        if self.pool_mode == "mean":
            return seq_emb.mean(dim=1)
        
        if self.pool_mode == "max":
            return seq_emb.max(dim=1)[0]
            
        if self.pool_mode == "last":
            return seq_emb[:, -1, :]
            
        return seq_emb
