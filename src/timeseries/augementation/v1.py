import torch
import torch.nn as nn
import random

class TemporalBlockMasking(nn.Module):
    def __init__(self, p: float = 0.5, block_size_ratio: float = 0.3):
        super().__init__()
        self.p = p  # Wahrscheinlichkeit, dass die Maskierung ÜBERHAUPT angewendet wird
        self.block_size_ratio = block_size_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return x

        C, L = x.shape
        block_size = int(L * self.block_size_ratio)
        max_start_idx = L - block_size

        if block_size < 1 or max_start_idx < 0:
            return x
        
        # Erstelle eine Kopie des Tensors, um die Änderungen zu speichern
        x_masked = x.clone()

        # Iteriere über jeden Kanal (C)
        for c in range(C):
            # 1. Zufällige Auswahl des Block-Startindexes pro Kanal
            start_idx = random.randint(0, max_start_idx)
            end_idx = start_idx + block_size
            
            # 2. Anwendung der Maskierung auf den aktuellen Kanal
            block = x[c, start_idx:end_idx]
            
            block_min = block.min()
            block_max = block.max()
            block_mean = block.mean()
            block_median = block.median() 

            r = random.random()
            
            if r < 0.33:
                # Uniform random between min and max
                rand = torch.rand(1).item()
                new_vals = block_min + rand * (block_max - block_min)
            elif r > 0.33 and r < 0.66:
                # Mean
                new_vals = block_mean
            else:
                # Median
                new_vals = block_median

            # Zuweisung der neuen Werte in die Kopie
            x_masked[c, start_idx:end_idx] = new_vals

        return x_masked
    
class TimeSeriesTransform(nn.Module):
    """
    Applies channel-wise randomized augmentations to (C, L) time series or batches (B, C, L).
    Each channel receives independently sampled transform parameters.
    Temporal cropping remains sample-level (shared across channels).
    """
    def __init__(
        self, output_length: int, scale_range=(.7, 1.2),
            jitter_std=0.3,  p_noise=0.3, p_freq_mask=0.5, max_freq_ratio=0.2,p_temporal_mask=0.8,p_transform=0.8
    ):
        super().__init__()
        self.output_length = output_length
        self.scale_range = scale_range
        self.jitter_std = jitter_std
        self.p_noise = p_noise

        self.p_freq_mask = p_freq_mask
        self.max_freq_ratio = max_freq_ratio

        self.p_temporal_mask = p_temporal_mask
        self.p_transform = p_transform

        # channel masking (whole channels)
        self.temporal_block_masking = TemporalBlockMasking(
            p=self.p_temporal_mask,
            block_size_ratio=0.1
        )

    def _ensure_batch(self, x):
        # Accepts (C, L) or (B, C, L), returns (B, C, L)
        if x.dim() == 2:
            return x.unsqueeze(0)
        return x

    def _remove_batch(self, x, orig_dim):
        # If input was (C, L), return (C, L), else (B, C, L)
        if orig_dim == 2:
            return x.squeeze(0)
        return x

    # -----------------------
    # Channel-wise Scaling
    # -----------------------
    def _scaling(self, x):
        # x: (B, C, L)
        B, C, L = x.size()
        scales = torch.empty(B, C, 1, device=x.device).uniform_(
            self.scale_range[0], self.scale_range[1]
        )
        return x * scales

    # -----------------------
    # Channel-wise Drift
    # -----------------------
    def _drift(self, x):
        B, C, L = x.size()
        t = torch.linspace(0, 1, steps=L, device=x.device).view(1, 1, L).expand(B, C, L)
        slopes = torch.empty(B, C, 1, device=x.device).uniform_(
            self.scale_range[0], self.scale_range[1]
        )
        return x + t * slopes

    # -----------------------
    # Channel-wise Frequency Masking
    # -----------------------
    def _frequency_masking(self, x):
        # x: (B, C, L)
        B, C, L = x.size()
        x_out = x.clone()
        for b in range(B):
            for c in range(C):
                if random.random() < self.p_freq_mask:
                    rfft_data = torch.fft.rfft(x[b, c], dim=-1)
                    rlen = rfft_data.size(-1)
                    maskable_len = rlen - 1
                    if maskable_len <= 0:
                        continue
                    max_mask = int(maskable_len * self.max_freq_ratio)
                    if max_mask < 1:
                        continue
                    mask_size = random.randint(1, max_mask)
                    start_limit = rlen - mask_size
                    if start_limit < 1:
                        continue
                    start = random.randint(1, start_limit)
                    end = start + mask_size
                    rfft_data[start:end] = 0.0
                    x_out[b, c] = torch.fft.irfft(rfft_data, n=L, dim=-1)
        return x_out

    # -----------------------
    # Channel-wise Feature Jitter (already channel-wise)
    # -----------------------
    def _feature_jitter(self, x):
        B, C, L = x.size()
        shift = torch.randn((B, C, 1), device=x.device) * self.jitter_std
        scale = torch.rand((B, C, 1), device=x.device) * self.jitter_std + 1.0
        return x * scale + shift

    # -----------------------
    # Channel-wise Gaussian Noise
    # -----------------------
    def _add_gaussian_noise(self, x):
        B, C, L = x.size()
        x_out = x.clone()
        for b in range(B):
            for c in range(C):
                if random.random() < self.p_noise:
                    noise_std = x[b, c].std() * 0.1
                    noise = torch.randn_like(x[b, c]) * noise_std
                    x_out[b, c] = x[b, c] + noise
        return x_out

    # -----------------------
    # Global Temporal Crop
    # -----------------------
    def _temporal_crop(self, x):
        # x: (B, C, L)
        B, C, L_in = x.size()
        x_out = torch.empty((B, C, self.output_length), device=x.device)
        for b in range(B):
            scale = random.uniform(*self.scale_range)
            crop_L = int(L_in * scale)
            crop_L = max(1, min(crop_L, L_in))
            start = random.randint(0, L_in - crop_L)
            cropped = x[b, :, start:start + crop_L]
            resized = F.interpolate(
                cropped.unsqueeze(0),
                size=self.output_length,
                mode="linear",
                align_corners=False
            ).squeeze(0)
            x_out[b] = resized
        return x_out

    # -----------------------
    # Forward
    # -----------------------
    def forward(self, x):
        orig_dim = x.dim()
        x = self._ensure_batch(x)  # (B, C, L)
        if random.random() < self.p_transform*0.1:
            return self._remove_batch(x, orig_dim)
        # if random.random() < self.p_transform:
        #     x = self._temporal_crop(x)
        # if random.random() < self.p_transform:
        #     x = self._scaling(x)
        # if random.random() < self.p_transform:
        #     x = self._drift(x)
        # if random.random() < self.p_transform:
        #     x = self._feature_jitter(x)
        # if random.random() < self.p_transform:
        #     x = self._frequency_masking(x)
        if random.random() < self.p_transform:
            x = self._add_gaussian_noise(x)
        if random.random() < self.p_transform:
            # TemporalBlockMasking expects (C, L), so apply per batch
            x = torch.stack([self.temporal_block_masking(x[b]) for b in range(x.size(0))], dim=0)
        return self._remove_batch(x, orig_dim)
