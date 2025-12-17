import torch
import torch.nn as nn
import torch.nn.init as init


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


class JEPAProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # Standard SSL design: 3 layers, Batch/Layer Norm, ReLU
        # Hidden dimension is typically wider than input (e.g., 4x)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(), # GELU is often preferred over ReLU in modern architectures (BERT, ViT)
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            
            nn.Linear(hidden_dim, output_dim),
            # In JEPA/SimCLR, the final layer often has no non-linearity 
            # to allow the full vector space to be utilized.
            nn.LayerNorm(output_dim) 
        )

    def forward(self, x):
        return self.net(x)


class TimeSeriesLSTMEncoder(nn.Module):
    """
    LSTM-based encoder for multivariate time series.
    
    Processes sensor data through stacked LSTM layers,
    then aggregates temporal information via pooling.
    """
    def __init__(self, input_channels: int = 170, encoder_output_dim: int = 12, 
                 proj_dim: int = 128, hidden_channels: int = 64, num_layers: int = 2,
                 dropout: float = 0.2, bidirectional: bool = True):
        super().__init__()
        
        self.input_channels = input_channels
        self.encoder_output_dim = encoder_output_dim
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM processes time series: input [B, L, C]
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_channels,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Determine LSTM output size
        lstm_out_channels = hidden_channels * (2 if bidirectional else 1)
        
        # Temporal pooling
        self.pooling = MultiScalePool(lstm_out_channels)
        
        # Encoder head: reduce to embedding dimension
        self.encoder_head = nn.Linear(lstm_out_channels, encoder_output_dim)
        
        # Projector
        self.proj = JEPAProjector(
            input_dim=encoder_output_dim, 
            hidden_dim=4*encoder_output_dim,
            output_dim=proj_dim,
          
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

    def _backbone_forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM.
        
        Args:
            x_flat: [B*V, C, L] - flattened batch of time series
            
        Returns:
            emb: [B*V, encoder_output_dim]
        """
        B, C, L = x_flat.shape
        
        # Transpose for LSTM: [B*V, C, L] -> [B*V, L, C]
        x_seq = x_flat.transpose(1, 2)
        
        # LSTM forward: [B*V, L, C] -> [B*V, L, hidden*2] (if bidirectional)
        lstm_out, (h_n, c_n) = self.lstm(x_seq)
        
        # Pool temporal information: reshape lstm_out to [B*V, hidden, L] for pooling
        lstm_out_channels = lstm_out.shape[-1]
        x_pooled = lstm_out.transpose(1, 2)  # [B*V, hidden, L]
        x_pooled = self.pooling(x_pooled)  # [B*V, hidden]
        
        # Project to embedding dimension
        emb_out = self.encoder_head(x_pooled)  # [B*V, encoder_output_dim]
        
        return emb_out

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, V, C, L] - batch of multiple views of time series
            
        Returns:
            emb: [B*V, encoder_output_dim]
            proj: [B*V, proj_dim]
        """
        N, V, C, L = x.shape
        x_flat = x.view(N * V, C, L)
        
        # Forward through backbone
        emb = self._backbone_forward(x_flat)
        
        # Project embeddings
        proj = self.proj(emb)
        
        return emb, proj


class TimeSeriesGRUEncoder(nn.Module):
    """
    GRU-based encoder for multivariate time series.
    
    Similar to LSTM but with GRU (fewer parameters, faster training).
    Processes sensor data through stacked GRU layers,
    then aggregates temporal information via pooling.
    """
    def __init__(self, input_channels: int = 170, encoder_output_dim: int = 12, 
                 proj_dim: int = 128, hidden_channels: int = 64, num_layers: int = 2,
                 dropout: float = 0.2, bidirectional: bool = True):
        super().__init__()
        
        self.input_channels = input_channels
        self.encoder_output_dim = encoder_output_dim
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # GRU processes time series: input [B, L, C]
        self.gru = nn.GRU(
            input_size=input_channels,
            hidden_size=hidden_channels,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Determine GRU output size
        gru_out_channels = hidden_channels * (2 if bidirectional else 1)
        
        # Temporal pooling
        self.pooling = MultiScalePool(gru_out_channels)
        
        # Encoder head: reduce to embedding dimension
        self.encoder_head = nn.Linear(gru_out_channels, encoder_output_dim)
        
        # Projector
        self.proj = JEPAProjector(
            input_dim=encoder_output_dim, 
            hidden_dim=4*encoder_output_dim,
            output_dim=proj_dim,
          
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

    def _backbone_forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GRU.
        
        Args:
            x_flat: [B*V, C, L] - flattened batch of time series
            
        Returns:
            emb: [B*V, encoder_output_dim]
        """
        B, C, L = x_flat.shape
        
        # Transpose for GRU: [B*V, C, L] -> [B*V, L, C]
        x_seq = x_flat.transpose(1, 2)
        
        # GRU forward: [B*V, L, C] -> [B*V, L, hidden*2] (if bidirectional)
        gru_out, h_n = self.gru(x_seq)
        
        # Pool temporal information: reshape gru_out to [B*V, hidden, L] for pooling
        gru_out_channels = gru_out.shape[-1]
        x_pooled = gru_out.transpose(1, 2)  # [B*V, hidden, L]
        x_pooled = self.pooling(x_pooled)  # [B*V, hidden]
        
        # Project to embedding dimension
        emb_out = self.encoder_head(x_pooled)  # [B*V, encoder_output_dim]
        
        return emb_out

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, V, C, L] - batch of multiple views of time series
            
        Returns:
            emb: [B*V, encoder_output_dim]
            proj: [B*V, proj_dim]
        """
        N, V, C, L = x.shape
        x_flat = x.view(N * V, C, L)
        
        # Forward through backbone
        emb = self._backbone_forward(x_flat)
        
        # Project embeddings
        proj = self.proj(emb)
        
        return emb, proj


class TimeSeriesAttentionRNNEncoder(nn.Module):
    """
    LSTM encoder with temporal attention pooling.
    
    Uses attention mechanism to weight importance of different time steps
    instead of simple avg/max pooling.
    """
    def __init__(self, input_channels: int = 170, encoder_output_dim: int = 12, 
                 proj_dim: int = 128, hidden_channels: int = 64, num_layers: int = 2,
                 dropout: float = 0.2, bidirectional: bool = True):
        super().__init__()
        # never use instance norm!
        self.input_channels = input_channels
        self.encoder_output_dim = encoder_output_dim
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM processes time series
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_channels,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Determine LSTM output size
        lstm_out_channels = hidden_channels * (2 if bidirectional else 1)
        
        # Attention mechanism for temporal weighting
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_channels, lstm_out_channels // 4),
            nn.ReLU(),
            nn.Linear(lstm_out_channels // 4, 1)
        )
        
        # Encoder head: reduce to embedding dimension
        self.encoder_head = nn.Linear(lstm_out_channels, encoder_output_dim)
        
        # Projector
        self.proj = JEPAProjector(
            input_dim=encoder_output_dim, 
            hidden_dim=4*encoder_output_dim,
            output_dim=proj_dim,
          
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

    def _backbone_forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM with attention pooling.
        
        Args:
            x_flat: [B*V, C, L] - flattened batch of time series
            
        Returns:
            emb: [B*V, encoder_output_dim]
        """
        B, C, L = x_flat.shape
        # Transpose for LSTM: [B*V, C, L] -> [B*V, L, C]
        x_seq = x_flat.transpose(1, 2)
        
        # LSTM forward: [B*V, L, C] -> [B*V, L, hidden*2] (if bidirectional)
        lstm_out, (h_n, c_n) = self.lstm(x_seq)
        
        # Attention pooling: weight each timestep by importance
        # Compute attention weights: [B*V, L, 1]
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)  # [B*V, L, 1]
        
        # Apply attention: [B*V, L, hidden] * [B*V, L, 1] -> [B*V, hidden]
        x_pooled = (lstm_out * attn_weights).sum(dim=1)
        
        # Project to embedding dimension
        emb_out = self.encoder_head(x_pooled)  # [B*V, encoder_output_dim]
        
        return emb_out

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, V, C, L] - batch of multiple views of time series
            
        Returns:
            emb: [B*V, encoder_output_dim]
            proj: [B*V, proj_dim]
        """
        N, V, C, L = x.shape
        x_flat = x.view(N * V, C, L)
        
        # Forward through backbone
        emb = self._backbone_forward(x_flat)
        
        # Project embeddings
        proj = self.proj(emb)
        
        return emb, proj
