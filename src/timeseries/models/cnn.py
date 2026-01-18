import torch
import torch.nn as nn
import torch.nn.init as init


class DilatedResidualBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve length")

        padding = (kernel_size - 1) // 2 * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.LayerNorm(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.LayerNorm(out_channels)

        self.downsample = nn.Identity()
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.LayerNorm(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        out = self.act(out)
        return out


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
        avg = self.avg_pool(x).squeeze(-1)
        max_p = self.max_pool(x).squeeze(-1)
        out = torch.cat([avg, max_p], dim=1)
        out = self.fuse(out)
        out = self.norm(out)
        return out


class TimeSeriesEncoder(nn.Module):
    """Temporal CNN encoder for multivariate time series.

    Expected input: x [B, C, L]
    Output: embedding [B, output_dim]

    This version uses dilated residual blocks to capture multi-scale patterns
    (e.g., daily/weekly periodicity + sharp spikes) common in traffic data.
    """

    def __init__(
        self,
        input_channels: int,
        output_dim: int = 512,
        *,
        stem_channels: int = 128,
        kernel_size: int = 5,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        stem_pad = (kernel_size - 1) // 2
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, stem_channels, kernel_size=kernel_size, padding=stem_pad, bias=False),
            nn.LayerNorm(stem_channels),
            nn.GELU(),
        )

        blocks: list[nn.Module] = []
        in_ch = stem_channels
        for i, d in enumerate(dilations):
            out_ch = stem_channels if i < len(dilations) - 1 else stem_channels * 2
            blocks.append(
                DilatedResidualBlock1d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=3,
                    dilation=d,
                    dropout=dropout,
                )
            )
            in_ch = out_ch
        self.backbone = nn.Sequential(*blocks)

        self.pool = MultiScalePool(in_ch)
        self.head = nn.Sequential(
            nn.Linear(in_ch, output_dim),
            nn.LayerNorm(output_dim),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            init.orthogonal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.LayerNorm):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, Channels, Time]
        x = self.stem(x)
        x = self.backbone(x)
        x = self.pool(x)
        return self.head(x)