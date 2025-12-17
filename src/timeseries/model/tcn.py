import torch
import torch.nn as nn
import torch.nn.init as init

class ResidualBlock1d(nn.Module):
    """
    Ein Residual-Block mit 1D-Faltungen, Batch-Norm und ReLU.
    Verhindert das Verschwinden von Gradienten bei tieferen Netzen.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut anpassen, falls sich Dimensionen ändern (z.B. durch Stride oder Kanaländerung)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        return out

class MultiScalePool(nn.Module):
    """Combines average and max pooling with learned fusion."""
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Linear(channels * 2, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        """x: [B, C, T] -> [B, C]"""
        avg = self.avg_pool(x).squeeze(-1)  # [B, C]
        max_p = self.max_pool(x).squeeze(-1)  # [B, C]
        combined = torch.cat([avg, max_p], dim=1)  # [B, 2C]
        out = self.linear(combined)  # [B, C]
        out = self.norm(out)  # [B, C]
        return out

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        layers = []
        in_dim = input_dim
        
        # Hidden Layers
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(norm_layer(h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
            
        # --- FIX 1: Letzte Schicht separat behandeln ---
        self.net = nn.Sequential(*layers)
        
        # Die letzte Projektion auf die Dimension, in der der Loss berechnet wird
        self.last_layer = nn.Linear(in_dim, output_dim)
        self.last_norm = nn.LayerNorm(output_dim)
        #1 RMSNorm ok
        #1 LayerNorm elementwise false.. lower but instable
        #2 LayerNorm elementwise True.. similar to rms
        #3 LayerNorm similar to rms, layer


    def forward(self, x):
        x = self.net(x)
        x = self.last_layer(x)
        x= self.last_norm(x)
        return x

class TimeSeriesEncoder(nn.Module):
    def __init__(self, input_channels: int = 170, encoder_output_dim: int = 12, 
                 proj_dim: int = 128):
        super().__init__()
        
      
        
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.layer1 = ResidualBlock1d(64, 128, stride=2)
        self.pooling = MultiScalePool(128)
        self.encoder_head = nn.Linear(128, encoder_output_dim)

        self.proj = MLP(
            input_dim=encoder_output_dim, 
            hidden_dims=[proj_dim, proj_dim], # Letzter dim ist output via last_layer
            output_dim=proj_dim,
            norm_layer=nn.LayerNorm
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Orthogonale Initialisierung hilft bei Dekorrelation
            init.orthogonal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            # Kaiming für Convs (Best Practice für ReLU)
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

    def _backbone_forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        x = self.stem(x_flat)
        x = self.layer1(x)
        x = self.pooling(x)  # MultiScalePool returns [B, C] directly
        emb = self.encoder_head(x)
        return emb

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        N, V, C, L = x.shape
        x_flat = x.view(N * V, C, L)
        
        emb = self._backbone_forward(x_flat)
        
        # Projektor liefert jetzt bereits normalisierte Werte
        proj = self.proj(emb) 
        
        return emb, proj