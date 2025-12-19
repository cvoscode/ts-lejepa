"""
Covariate encoder and fusion modules for time series forecasting.

CovariateEncoder: Encodes time covariates [B, L, 6] -> [B, D]
FusedEncoder: Combines sensor backbone + covariate encoder outputs
"""

import torch
import torch.nn as nn


class CovariateEncoder(nn.Module):
    """
    Encodes temporal covariates (cyclical time features) into a fixed-dim embedding.
    
    Input: [B, L, num_features] - time features over sequence length
    Output: [B, output_dim] - pooled temporal embedding
    """
    
    def __init__(self, input_features: int = 6, hidden_dim: int = 64, output_dim: int = 64):
        super().__init__()
        self.output_dim = output_dim
        
        # Process time features with 1D convolutions over time
        self.net = nn.Sequential(
            # [B, 6, L] -> [B, hidden_dim, L]
            nn.Conv1d(input_features, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
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
        return self.net(x)
    

class FusedEncoder(nn.Module):
    """
    Fuses sensor encoder output with covariate encoder output.
    
    Concatenates both embeddings and projects to a unified dimension.
    """
    
    def __init__(self, sensor_encoder: nn.Module, covariate_encoder: nn.Module, 
                 fusion_dim: int = None):
        super().__init__()
        self.sensor_encoder = sensor_encoder
        self.covariate_encoder = covariate_encoder
        
        sensor_out = getattr(sensor_encoder, "output_dim", None)
        cov_out = getattr(covariate_encoder, "output_dim", None)
        
        if sensor_out is None or cov_out is None:
            raise ValueError("Both encoders must expose .output_dim attribute")
        
        self.sensor_out = sensor_out
        self.cov_out = cov_out
        
        # Output dimension defaults to sensor encoder's output
        self.output_dim = fusion_dim if fusion_dim is not None else sensor_out
        
        # Fusion projection
        self.fusion = nn.Sequential(
            nn.Linear(sensor_out + cov_out, self.output_dim),
            nn.GELU(),
        )
    
    def forward(self, sensor_data: torch.Tensor, time_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sensor_data: Sensor input [B, C, L]
            time_features: Time covariates [B, L, 6]
        Returns:
            Fused embedding [B, output_dim]
        """
        sensor_emb = self.sensor_encoder(sensor_data)   # [B, sensor_out]
        cov_emb = self.covariate_encoder(time_features) # [B, cov_out]
        
        fused = torch.cat([sensor_emb, cov_emb], dim=-1)  # [B, sensor_out + cov_out]
        return self.fusion(fused)  # [B, output_dim]


class FutureCovariateEncoder(nn.Module):
    """
    Encodes known future covariates for the forecast horizon.
    
    Input: [B, H, 6] - time features for each forecast step
    Output: [B, output_dim] - aggregated future context
    """
    
    def __init__(self, input_features: int = 6, hidden_dim: int = 32, output_dim: int = 64):
        super().__init__()
        self.output_dim = output_dim
        
        self.net = nn.Sequential(
            nn.Conv1d(input_features, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Future time features [B, H, num_features]
        Returns:
            Future context embedding [B, output_dim]
        """
        x = x.transpose(1, 2)  # [B, num_features, H]
        return self.net(x)
