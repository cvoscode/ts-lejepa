"""
Covariate encoder and fusion modules for time series forecasting.

CovariateEncoder: Encodes time covariates [B, L, 6] -> [B, D]
FusedEncoder: Combines sensor backbone + covariate encoder outputs
"""

import torch
import torch.nn as nn
import torch.nn.init as init

from ..core.base_encoder import BaseEncoder
from .layers import ChannelMixer


class CovariateEncoder(nn.Module):
    """
    Encodes temporal covariates (cyclical time features) into a fixed-dim embedding.
    
    Input: [B, L, num_features] - time features over sequence length
    Output: [B, output_dim] - pooled temporal embedding
    """
    
    def __init__(
        self,
        input_features: int = 6,
        hidden_dim: int = 64,
        output_dim: int = 64,
        channel_mixer: str = "none",
        channel_mixer_reduction: int = 4,
        channel_mixer_attn_dim: int = 64,
        channel_mixer_attn_heads: int = 4,
        channel_mixer_attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.output_dim = output_dim

        self.channel_mixer = ChannelMixer(
            input_features,
            mode=channel_mixer,
            reduction=channel_mixer_reduction,
            attn_dim=channel_mixer_attn_dim,
            attn_heads=channel_mixer_attn_heads,
            attn_dropout=channel_mixer_attn_dropout,
        )
        
        # Process time features with 1D convolutions over time
        self.net = nn.Sequential(
            # [B, 6, L] -> [B, hidden_dim, L]
            nn.Conv1d(input_features, hidden_dim, kernel_size=3, padding=1),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),  # [B, hidden_dim, 1]
            nn.Flatten(),             # [B, hidden_dim]
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Time features [B, L, num_features]
        Returns:
            Temporal embedding [B, output_dim]
        """
        # Transpose to [B, num_features, L] for Conv1d
        x = x.transpose(1, 2)
        x = self.channel_mixer(x)
        return self.net(x)
    

class FusedEncoder(BaseEncoder):
    """
    Fuses sensor encoder output with covariate encoder output.
    
    Both encoders should be BaseEncoder instances.
    Concatenates embeddings at each timestep and projects to unified dimension.
    """
    
    def __init__(self, sensor_encoder: BaseEncoder, covariate_encoder: BaseEncoder, 
                 output_dim: int = None, pool_mode: str = "mean"):
        # Use sensor encoder's input channels as our input channels
        input_channels = sensor_encoder.input_channels
        
        # Default output_dim to sensor encoder's output_dim
        if output_dim is None:
            output_dim = sensor_encoder.output_dim
        
        super().__init__(input_channels, output_dim, pool_mode)
        
        # Set both encoders to 'none' pooling so we get temporal outputs
        self.sensor_encoder = sensor_encoder
        self.sensor_encoder.pool_mode = "none"
        
        self.covariate_encoder = covariate_encoder
        self.covariate_encoder.pool_mode = "none"
        
        # Fusion projection
        fusion_input_dim = sensor_encoder.output_dim + covariate_encoder.output_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
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
    
    def forward_backbone(self, x: torch.Tensor, time_features: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: Sensor input [B, C, L]
            time_features: Time covariates [B, L, F]
        Returns:
            Fused embedding [B, L, D]
        """
        if time_features is None:
            raise ValueError("FusedEncoder requires time_features")
        
        # Both return [B, L, D_sensor] and [B, L, D_cov]
        sensor_emb = self.sensor_encoder.forward_backbone(x, time_features)
        cov_emb = self.covariate_encoder.forward_backbone(x, time_features)
        
        # Concatenate along feature dimension
        fused = torch.cat([sensor_emb, cov_emb], dim=-1)  # [B, L, D_sensor + D_cov]
        return self.fusion(fused)  # [B, L, D]


class FutureCovariateEncoder(BaseEncoder):
    """
    Encodes known future covariates for the forecast horizon.
    
    Input: time_features [B, H, F] - time features for each forecast step
    Output: [B, H, D] or pooled [B, D] depending on pool_mode
    """
    
    def __init__(
        self,
        input_channels: int = 6,
        output_dim: int = 64,
        pool_mode: str = "mean",
        hidden_dim: int = 32,
        channel_mixer: str = "none",
        channel_mixer_reduction: int = 4,
        channel_mixer_attn_dim: int = 64,
        channel_mixer_attn_heads: int = 4,
        channel_mixer_attn_dropout: float = 0.0,
    ):
        super().__init__(input_channels, output_dim, pool_mode)
        self.hidden_dim = hidden_dim

        self.channel_mixer = ChannelMixer(
            input_channels,
            mode=channel_mixer,
            reduction=channel_mixer_reduction,
            attn_dim=channel_mixer_attn_dim,
            attn_heads=channel_mixer_attn_heads,
            attn_dropout=channel_mixer_attn_dropout,
        )
        
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.orthogonal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
    
    def forward_backbone(self, x: torch.Tensor, time_features: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: Ignored (can be dummy input)
            time_features: Future time features [B, H, F]
        Returns:
            Future context embedding [B, H, D]
        """
        if time_features is None:
            raise ValueError("FutureCovariateEncoder requires time_features")
        
        # time_features: [B, H, F] -> [B, F, H]
        x = time_features.transpose(1, 2)
        x = self.channel_mixer(x)
        x = self.net(x)  # [B, hidden_dim, H]
        x = x.transpose(1, 2)  # [B, H, hidden_dim]
        x = self.proj(x)  # [B, H, D]
        return x
