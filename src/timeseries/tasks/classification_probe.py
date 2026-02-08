"""Classification probe head for SSL evaluation.

Maps pooled embeddings to class logits.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ClassificationProbe(nn.Module):
    """Probe that maps embeddings to class predictions.

    Input:  [B, D] embedding
    Output: [B, num_classes] logits

    Usage:
        >>> probe = ClassificationProbe(input_dim=256, num_classes=5)
        >>> emb = encoder(window)  # [B, D]
        >>> logits = probe(emb)    # [B, 5]
    """

    def __init__(
        self,
        *,
        input_dim: int,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        num_hidden_layers: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)

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
            self.head = nn.Linear(hidden_dim, self.num_classes)
        else:
            self.hidden = None
            self.head = nn.Linear(input_dim, self.num_classes)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """Return class logits shaped [B, num_classes].

        Args:
            emb: Pooled encoder embedding [B, D] or multi-level list.

        Returns:
            Logits [B, num_classes]. Apply softmax/argmax for predictions.
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


__all__ = ["ClassificationProbe"]
