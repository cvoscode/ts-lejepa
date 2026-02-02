"""Basic augmentation ops for time-series SSL."""

from __future__ import annotations
import random
import torch
import torch.nn.functional as F

from ..base import Transform


class Scaling(Transform):
    """Randomly scale each channel."""
    def __init__(self, scale_range: tuple[float, float] = (0.8, 1.2)):
        self.scale_range = scale_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        B, C, L = x.shape
        scales = torch.empty(B, C, 1, device=x.device).uniform_(*self.scale_range)
        out = x * scales
        return out.squeeze(0) if squeeze else out


class Drift(Transform):
    """Add a linear drift per channel."""
    def __init__(self, slope_range: tuple[float, float] = (-0.1, 0.1)):
        self.slope_range = slope_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        B, C, L = x.size()
        t = torch.linspace(0, 1, steps=L, device=x.device).view(1, 1, L)
        slopes = torch.empty(B, C, 1, device=x.device).uniform_(*self.slope_range)
        out = x + t * slopes
        return out.squeeze(0) if squeeze else out


class FeatureJitter(Transform):
    """Add Gaussian noise to each sample."""
    def __init__(self, jitter_std: float = 0.05):
        self.jitter_std = float(jitter_std)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.jitter_std


class AddGaussianNoise(Transform):
    """Add scaled Gaussian noise with probability p."""
    def __init__(self, scale: float = 0.05, p: float = 0.3):
        self.scale = float(scale)
        self.p = float(p)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        stds = x.std(dim=-1, keepdim=True) * self.scale
        return x + torch.randn_like(x) * stds


class FrequencyMask(Transform):
    """Mask random frequency bands in the FFT domain."""
    def __init__(self, p: float = 0.3, max_freq_ratio: float = 0.1):
        self.p = float(p)
        self.max_freq_ratio = float(max_freq_ratio)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        B, C, L = x.size()
        rfft_data = torch.fft.rfft(x, dim=-1)
        rlen = rfft_data.size(-1)
        if rlen <= 2:
            return x.squeeze(0) if squeeze else x
        max_mask = max(1, int(rlen * self.max_freq_ratio))
        if max_mask >= (rlen - 1):
            return x.squeeze(0) if squeeze else x
        mask_starts = torch.randint(1, rlen - max_mask, (B, C), device=x.device)
        mask_sizes = torch.randint(1, max_mask + 1, (B, C), device=x.device)
        indices = torch.arange(rlen, device=x.device).view(1, 1, rlen)
        mask = (indices >= mask_starts.unsqueeze(-1)) & (indices < (mask_starts + mask_sizes).unsqueeze(-1))
        rfft_data[mask] = 0.0
        out = torch.fft.irfft(rfft_data, n=L, dim=-1)
        return out.squeeze(0) if squeeze else out


class MagnitudeWarp(Transform):
    """Apply a smooth multiplicative warp curve."""
    def __init__(self, p: float = 0.3, num_knots: int = 4, noise_scale: float = 0.1):
        self.p = float(p)
        self.num_knots = int(num_knots)
        self.noise_scale = float(noise_scale)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        B, C, L = x.size()
        knots = torch.randn(B, C, self.num_knots, device=x.device) * self.noise_scale + 1.0
        warp_curves = F.interpolate(knots, size=L, mode="linear", align_corners=False)
        out = x * warp_curves
        return out.squeeze(0) if squeeze else out


class TemporalCrop(Transform):
    """Crop then resize to output length."""
    def __init__(self, output_length: int, crop_ratio_range: tuple[float, float] = (0.8, 1.0)):
        self.output_length = int(output_length)
        self.crop_ratio_range = crop_ratio_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        B, C, L_in = x.size()
        crop_ratio = random.uniform(*self.crop_ratio_range)
        crop_L = int(L_in * crop_ratio)
        crop_L = max(1, min(crop_L, L_in))
        start = random.randint(0, L_in - crop_L)
        cropped = x[:, :, start : start + crop_L]
        out = F.interpolate(cropped, size=self.output_length, mode="linear", align_corners=False)
        return out.squeeze(0) if squeeze else out


class TemporalBlockMask(Transform):
    """Mask a contiguous temporal block per channel."""
    def __init__(self, p: float = 0.5, block_size_ratio: float = 0.1):
        self.p = float(p)
        self.block_size_ratio = float(block_size_ratio)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        B, C, L = x.shape
        block_size = int(L * self.block_size_ratio)
        if block_size < 1:
            return x.squeeze(0) if squeeze else x
        max_start = L - block_size
        start_indices = torch.randint(0, max_start + 1, (B, C), device=x.device)
        indices = torch.arange(L, device=x.device).view(1, 1, L)
        mask = (indices >= start_indices.unsqueeze(-1)) & (indices < (start_indices + block_size).unsqueeze(-1))
        r = random.random()
        if r < 0.5:
            fill_values = x.mean(dim=-1, keepdim=True)
        else:
            fill_values = torch.zeros_like(x[:, :, :1])
        out = torch.where(mask, fill_values, x)
        return out.squeeze(0) if squeeze else out


__all__ = [
    "Scaling",
    "Drift",
    "FeatureJitter",
    "AddGaussianNoise",
    "FrequencyMask",
    "MagnitudeWarp",
    "TemporalCrop",
    "TemporalBlockMask",
]
