import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

class TimeEncoding(nn.Module):
    """
    Generates sinusoidal time encodings (positional encodings) for time series.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, L)
        Returns:
            (B, C + d_model, L)
        """
        B, C, L = x.shape
        pe = self.pe[:L, :].transpose(0, 1).unsqueeze(0).expand(B, -1, -1)
        return torch.cat([x, pe], dim=1)

class TemporalBlockMasking(nn.Module):
    def __init__(self, p: float = 0.5, block_size_ratio: float = 0.1):
        super().__init__()
        self.p = p 
        self.block_size_ratio = block_size_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        # Apply with probability p during train, never during eval.
        if (not self.training) or (random.random() >= self.p):
            return x

        B, C, L = x.shape
        block_size = int(L * self.block_size_ratio)
        if block_size < 1:
            return x
        
        # Generate start indices for all batches and channels: (B, C)
        max_start = L - block_size
        start_indices = torch.randint(0, max_start + 1, (B, C), device=x.device)
        
        # Create a mask of shape (B, C, L)
        # Using broadcasting to create the mask efficiently
        indices = torch.arange(L, device=x.device).view(1, 1, L)
        mask = (indices >= start_indices.unsqueeze(-1)) & (indices < (start_indices + block_size).unsqueeze(-1))
        
        # Efficient replacement values
        # We'll use a mix of mean and zero for simplicity and performance in vectorization
        # Randomly choose replacement strategy per batch (simplified for vectorization)
        r = random.random()
        if r < 0.5:
            # Mean of each channel
            fill_values = x.mean(dim=-1, keepdim=True)
        else:
            # Zero
            fill_values = torch.zeros_like(x[:, :, :1])
            
        return torch.where(mask, fill_values, x)
    
class TimeSeriesTransform(nn.Module):
    """
    Applies vectorized channel-wise randomized augmentations to (B, C, L).
    """
    def __init__(
        self, output_length: int, 
        scale_range=(0.8, 1.2),
        crop_ratio_range=None,
        jitter_std=0.05, 
        p_noise=0.3, 
        p_freq_mask=0.3, 
        max_freq_ratio=0.1,
        p_temporal_mask=0.5,
        p_magnitude_warp=0.3,
        p_transform=0.8,
        use_time_encoding=False,
        d_time=4
    ):
        super().__init__()
        self.output_length = output_length
        self.scale_range = scale_range
        # Crop ratio is now independent from amplitude scaling.
        # Backward compatible default: if not specified, reuse scale_range.
        self.crop_ratio_range = crop_ratio_range if crop_ratio_range is not None else scale_range
        self.jitter_std = jitter_std
        self.p_noise = p_noise
        self.p_freq_mask = p_freq_mask
        self.max_freq_ratio = max_freq_ratio
        self.p_temporal_mask = p_temporal_mask
        self.p_magnitude_warp = p_magnitude_warp
        self.p_transform = p_transform
        
        self.use_time_encoding = use_time_encoding
        if use_time_encoding:
            self.time_encoding = TimeEncoding(d_model=d_time)

        self.temporal_block_masking = TemporalBlockMasking(
            p=self.p_temporal_mask,
            block_size_ratio=0.1
        )

    def _ensure_batch(self, x):
        if x.dim() == 2:
            return x.unsqueeze(0)
        return x

    def _remove_batch(self, x, orig_dim):
        if orig_dim == 2:
            return x.squeeze(0)
        return x

    def _scaling(self, x):
        B, C, L = x.size()
        scales = torch.empty(B, C, 1, device=x.device).uniform_(*self.scale_range)
        return x * scales

    def _drift(self, x):
        B, C, L = x.size()
        t = torch.linspace(0, 1, steps=L, device=x.device).view(1, 1, L)
        slopes = torch.empty(B, C, 1, device=x.device).uniform_(-0.1, 0.1) # Reduced drift range
        return x + t * slopes

    def _frequency_masking(self, x):
        if random.random() > self.p_freq_mask:
            return x
        B, C, L = x.size()
        rfft_data = torch.fft.rfft(x, dim=-1)
        rlen = rfft_data.size(-1)
        
        # Guard against edge cases that would make randint bounds invalid.
        if rlen <= 2:
            return x

        max_mask = max(1, int(rlen * self.max_freq_ratio))
        # Ensure at least one valid start index in [1, rlen-max_mask).
        if max_mask >= (rlen - 1):
            return x

        mask_starts = torch.randint(1, rlen - max_mask, (B, C), device=x.device)
        mask_sizes = torch.randint(1, max_mask + 1, (B, C), device=x.device)
        
        indices = torch.arange(rlen, device=x.device).view(1, 1, rlen)
        mask = (indices >= mask_starts.unsqueeze(-1)) & (indices < (mask_starts + mask_sizes).unsqueeze(-1))
        
        rfft_data[mask] = 0.0
        return torch.fft.irfft(rfft_data, n=L, dim=-1)

    def _feature_jitter(self, x):
        B, C, L = x.size()
        # Fast vectorized jitter
        noise = torch.randn_like(x) * self.jitter_std
        return x + noise

    def _add_gaussian_noise(self, x):
        if random.random() > self.p_noise:
            return x
        # Use a single global noise scale or per channel
        stds = x.std(dim=-1, keepdim=True) * 0.05
        return x + torch.randn_like(x) * stds

    def _magnitude_warping(self, x):
        if random.random() > self.p_magnitude_warp:
            return x
        B, C, L = x.size()
        # Create a smooth warping curve using low frequency noise
        num_knots = 4
        knots = torch.randn(B, C, num_knots, device=x.device) * 0.1 + 1.0
        # Interpolate to signal length
        warp_curves = F.interpolate(knots, size=L, mode='linear', align_corners=False)
        return x * warp_curves

    def _temporal_crop(self, x):
        # x: (B, C, L)
        B, C, L_in = x.size()
        # Random crop for the whole batch for performance, or different per sample
        # Let's do different per sample but vectorized using grid_sample or just interpolate
        # Simple version: Fixed size crop from random start
        crop_ratio = random.uniform(*self.crop_ratio_range)
        crop_L = int(L_in * crop_ratio)
        crop_L = max(1, min(crop_L, L_in))
        
        # To vectorize different starts, we can use a trick with unfold or just do a shared start for the batch
        # Shared start is much faster.
        start = random.randint(0, L_in - crop_L)
        cropped = x[:, :, start:start + crop_L]
        
        return F.interpolate(cropped, size=self.output_length, mode="linear", align_corners=False)

    def forward(self, x):
        orig_dim = x.dim()
        x = self._ensure_batch(x) # (B, C, L)
        
        if self.training:
            if random.random() < self.p_transform:
                # Apply sequence of augmentations
                # We can chain them or pick one for speed
                r = random.random()
                if r < 0.2: x = self._scaling(x)
                elif r < 0.4: x = self._drift(x)
                elif r < 0.6: x = self._magnitude_warping(x)
                elif r < 0.8: x = self._frequency_masking(x)
                else: x = self._add_gaussian_noise(x)
                
                x = self._temporal_crop(x)
                x = self.temporal_block_masking(x)
            else:
                # Always ensure output_length
                if x.size(-1) != self.output_length:
                    x = F.interpolate(x, size=self.output_length, mode="linear", align_corners=False)
        else:
            # Eval mode: just resize if needed
            if x.size(-1) != self.output_length:
                x = F.interpolate(x, size=self.output_length, mode="linear", align_corners=False)

        if self.use_time_encoding:
            x = self.time_encoding(x)
            
        return self._remove_batch(x, orig_dim)
