import torch
import torch.nn as nn

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
