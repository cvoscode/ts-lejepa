from __future__ import annotations
import torch
from ..base import Transform

class Masking(Transform):
    """Randomly masks a portion of the time series with zeros."""
    def __init__(self, mask_ratio: float = 0.1):
        self.mask_ratio = mask_ratio

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        B, C, T = x.shape
        mask = torch.rand(B, 1, T, device=x.device) > self.mask_ratio
        return x * mask.float()


class TimeWarping(Transform):
    """Simulates time warping via interpolation.
    
    Simplified implementation: Downsample and Upsample.
    """
    def __init__(self, warp_factor: float = 0.2):
        self.warp_factor = warp_factor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        B, C, T = x.shape
        # Only warp if batch > 0
        
        # Select random new length
        # warp between [1-factor, 1+factor]
        # Since we must return same length T, we just resample.
        # Ideally, we warp segments.
        # Simple approach: Resample to T' then interpolate back to T? 
        # That's just scaling in Time.
        
        # Better: clamp a random speed change.
        # For simplicity in this gap filling, let's implement simple time stretching 
        # (resample random subsegment to full length?) 
        # No, that changes content.
        
        # Let's implement "Resizing" style warping
        # Interpolate a random window size to T
        
        # Random window size within range around T
        # size = int(T * (1 + random(-warp, warp)))
        # F.interpolate(x, size) ... then crop or pad?
        
        # Correct TimeWarp is usually `F.grid_sample`. 
        # Let's stick to a simpler "Crop and Resize" which is common in Images.
        # But for TimeSeries, simply "Speed change" is good.
        
        speed = 1.0 + (torch.rand(1).item() * 2 - 1) * self.warp_factor
        new_T = int(T * speed)
        
        if new_T == T:
            return x
        
        out = torch.nn.functional.interpolate(x, size=new_T, mode='linear', align_corners=False)
        
        if new_T > T:
            # Crop center
            start = (new_T - T) // 2
            return out[:, :, start:start+T]
        else:
            # Pad ends -> replicating last value or zero? 
            # Replicate last value
            padding = T - new_T
            last_val = out[:, :, -1:]
            
            # pad_right
            # [B, C, T-new_T]
            pad = last_val.repeat(1, 1, padding)
            return torch.cat([out, pad], dim=2)

