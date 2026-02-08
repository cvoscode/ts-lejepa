"""Virtual sensing probe head for SSL evaluation.

Maps pooled embeddings to reconstruct masked channel values.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SensorProbe(nn.Module):
    """Probe that reconstructs masked sensor channels from embeddings.

    Given an embedding from the encoder (with some channels masked at input),
    this probe predicts the values of the masked channels over the window.

    Input:  [B, D] embedding (from encoder with masked channels)
    Output: [B, C_masked, T] reconstructed sensor values

    Usage:
        >>> probe = SensorProbe(input_dim=256, num_masked_channels=10, window_size=96)
        >>> emb = encoder(masked_input)  # [B, D]
        >>> reconstructed = probe(emb)   # [B, 10, 96]
    """

    def __init__(
        self,
        *,
        input_dim: int,
        num_masked_channels: int,
        window_size: int,
        hidden_dim: Optional[int] = None,
        num_hidden_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_masked_channels = int(num_masked_channels)
        self.window_size = int(window_size)
        self.output_size = self.num_masked_channels * self.window_size

        self.input_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

        if hidden_dim is not None:
            layers: list[nn.Module] = []
            in_dim = input_dim
            for _ in range(max(1, int(num_hidden_layers))):
                layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout),
                ])
                in_dim = hidden_dim
            self.hidden = nn.Sequential(*layers)
            self.head = nn.Linear(hidden_dim, self.output_size)
        else:
            self.hidden = None
            self.head = nn.Linear(input_dim, self.output_size)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """Return reconstructed masked channels shaped [B, C_masked, T].

        Args:
            emb: Pooled encoder embedding [B, D].

        Returns:
            Reconstructed values [B, C_masked, T].
        """
        if isinstance(emb, (list, tuple)):
            # Multi-level fusion: simple mean
            pooled = []
            for e in emb:
                if e.dim() == 3:
                    e = e.mean(dim=1)
                pooled.append(e)
            emb = torch.stack(pooled, dim=1).mean(dim=1)

        if emb.dim() == 3:
            emb = emb.mean(dim=1)

        x = self.input_norm(emb)
        x = self.dropout(x)
        if self.hidden is not None:
            x = self.hidden(x)
        out = self.head(x)
        return out.view(emb.shape[0], self.num_masked_channels, self.window_size)


__all__ = ["SensorProbe"]
