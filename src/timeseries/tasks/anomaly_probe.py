"""Anomaly detection probe head for SSL evaluation.

Maps pooled embeddings to an anomaly score.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class AnomalyProbe(nn.Module):
    """Probe that maps embeddings to an anomaly score.

    Input:  [B, D] embedding
    Output: [B, 1] anomaly score (higher = more anomalous)

    The score is unbounded; use sigmoid for [0, 1] interpretation if needed.

    Usage:
        >>> probe = AnomalyProbe(input_dim=256)
        >>> emb = encoder(window)     # [B, D]
        >>> scores = probe(emb)       # [B, 1]
        >>> probs = scores.sigmoid()  # [B, 1] in [0, 1]
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        num_hidden_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

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
            self.head = nn.Linear(hidden_dim, 1)
        else:
            self.hidden = None
            self.head = nn.Linear(input_dim, 1)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """Return anomaly scores shaped [B, 1].

        Args:
            emb: Pooled encoder embedding [B, D] or multi-level list.

        Returns:
            Anomaly scores [B, 1].
        """
        if isinstance(emb, (list, tuple)):
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
        return self.head(x)


__all__ = ["AnomalyProbe"]
