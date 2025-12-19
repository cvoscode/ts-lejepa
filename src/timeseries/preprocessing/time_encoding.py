"""
Cyclical time encoding utilities for time series forecasting.

Provides sin/cos encoding for cyclical time features:
- Day of year (1-365/366)
- Day of week (0-6)  
- Hour of day (0-23)

This gives 6 features total, appropriate for 5-minute granularity data.
"""

import numpy as np
import pandas as pd
import torch


def encode_timestamps(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """
    Generate cyclical time features from timestamps.
    
    Args:
        timestamps: DatetimeIndex of timestamps
        
    Returns:
        np.ndarray of shape [T, 6] with columns:
        [day_sin, day_cos, weekday_sin, weekday_cos, hour_sin, hour_cos]
    """
    # Day of year: [1, 366] -> [0, 2*pi]
    day_of_year = timestamps.dayofyear
    day_sin = np.sin(2 * np.pi * day_of_year / 365.25)
    day_cos = np.cos(2 * np.pi * day_of_year / 365.25)
    
    # Day of week: [0, 6] -> [0, 2*pi]
    weekday = timestamps.dayofweek
    weekday_sin = np.sin(2 * np.pi * weekday / 7)
    weekday_cos = np.cos(2 * np.pi * weekday / 7)
    
    # Hour of day: [0, 23] -> [0, 2*pi]
    hour = timestamps.hour + timestamps.minute / 60.0  # Include fractional hour
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    
    # Stack into [T, 6] array
    features = np.stack([
        day_sin, day_cos,
        weekday_sin, weekday_cos,
        hour_sin, hour_cos
    ], axis=1).astype(np.float32)
    
    return features


def encode_timestamps_torch(timestamps: pd.DatetimeIndex) -> torch.Tensor:
    """
    Generate cyclical time features as a PyTorch tensor.
    
    Args:
        timestamps: DatetimeIndex of timestamps
        
    Returns:
        torch.Tensor of shape [T, 6]
    """
    features = encode_timestamps(timestamps)
    return torch.from_numpy(features)


# Number of time features produced
NUM_TIME_FEATURES = 6
