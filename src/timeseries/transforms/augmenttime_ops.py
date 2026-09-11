"""Time-series augmentation ops backed by `augmenttime`.

Each class here is a thin `Transform` adapter around an `augmenttime.Augmentation`
so it can drop into the existing Compose / RandomApply / OneOf pipeline used by
the SSL data modules. The wrappers always accept a 3D `(batch, channels, time)`
tensor (the contract used throughout the SSL view builder) and return the same
shape.

Probabilistic gating (`p=...`) from the previous local ops is delegated to
`RandomApply` at the call site, not embedded in the wrapper.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from augmenttime import (
    BiasAugmentation,
    FrequencyBandAugmentation,
    NoiseAugmentation,
    ScaleAugmentation,
    SmoothingAugmentation,
    TimeWarpAugmentation,
    WindowMaskAugmentation,
)
from augmenttime.augmentations.base import Augmentation as _ATAugmentation

from .base import Transform


class _AugmenttimeOp(Transform):
    """Base wrapper that adapts an augmenttime `Augmentation` to the `Transform` interface.

    Augmenttime's augmentations sample fresh parameters per call, so we delegate
    directly. The 3D shape is preserved; no squeeze/unsqueeze dance is needed
    because the SSL view builder feeds 3D inputs (`[1, C, T]`).
    """

    def __init__(self, inner: _ATAugmentation) -> None:
        self.inner = inner

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        out = self.inner(x)
        if squeeze:
            out = out.squeeze(0)
        return out


class Scaling(_AugmenttimeOp):
    """Per-channel multiplicative scaling backed by `ScaleAugmentation`.

    Args:
        scale_range: `(lo, hi)` uniform range. Mapped to a Gaussian
            `(mean=(lo+hi)/2, std=(hi-lo)/4)` so ~95% of samples fall inside
            the requested interval (2-sigma rule).
    """

    def __init__(self, scale_range: tuple[float, float] = (0.8, 1.2)) -> None:
        lo, hi = scale_range
        mean = (lo + hi) / 2.0
        std = max((hi - lo) / 4.0, 1e-6)
        super().__init__(ScaleAugmentation(mean=mean, std=std))


class Bias(_AugmenttimeOp):
    """Per-channel additive bias offset backed by `BiasAugmentation`.

    Args:
        bias_std: Std of the per-channel bias sampling distribution (mean=0).
    """

    def __init__(self, bias_std: float = 0.05) -> None:
        super().__init__(BiasAugmentation(mean=0.0, std=float(bias_std)))


class FeatureJitter(_AugmenttimeOp):
    """White Gaussian noise added to every sample, backed by `NoiseAugmentation`.

    Args:
        jitter_std: Amplitude (std) of the additive white noise.
    """

    def __init__(self, jitter_std: float = 0.05) -> None:
        super().__init__(NoiseAugmentation(amplitude=float(jitter_std), noise_type="white"))


class AddGaussianNoise(_AugmenttimeOp):
    """White-noise injection with probability `p`, backed by `NoiseAugmentation`.

    The amplitude is scaled by per-sample std to roughly match the previous
    local op's behaviour (`noise = std * scale * randn`).
    """

    def __init__(self, scale: float = 0.05, p: float = 0.3) -> None:
        super().__init__(NoiseAugmentation(amplitude=float(scale), noise_type="white"))
        self.p = float(p)
        self.scale = float(scale)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(0)
        stds = x.std(dim=-1, keepdim=True) * self.scale
        # The inner NoiseAugmentation already adds Gaussian noise of amplitude=self.scale,
        # so subtract its contribution and re-add with the std-scaled amplitude.
        base = self.inner(x)
        # Replace the noise component: amplitude * randn -> stds * randn.
        # We re-derive base noise-free signal by detaching: x - noise = x - (base - x) = 2x - base.
        no_noise = 2.0 * x - base
        out = no_noise + torch.randn_like(x) * stds
        return out.squeeze(0) if squeeze else out


class FrequencyMask(_AugmenttimeOp):
    """FFT band-stop filtering with probability `p`, backed by `FrequencyBandAugmentation`.

    Args:
        p: Probability of applying the mask.
        max_freq_ratio: Width of each band as a fraction of FFT bins.
    """

    def __init__(self, p: float = 0.3, max_freq_ratio: float = 0.1) -> None:
        super().__init__(
            FrequencyBandAugmentation(
                num_bands=(1, 1),
                band_width_ratio=float(max_freq_ratio),
            )
        )
        self.p = float(p)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        return super().__call__(x)


class MagnitudeWarp(_AugmenttimeOp):
    """Smooth non-linear magnitude warping with probability `p`,
    backed by `TimeWarpAugmentation`.

    Args:
        p: Probability of applying the warp.
        num_knots: Number of displacement knots sampled along the time axis.
        noise_scale: Std of the per-knot displacement as a fraction of time steps.
    """

    def __init__(self, p: float = 0.3, num_knots: int = 4, noise_scale: float = 0.1) -> None:
        super().__init__(
            TimeWarpAugmentation(
                warp_scale=float(noise_scale),
                num_knots=int(num_knots),
            )
        )
        self.p = float(p)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        return super().__call__(x)


class TemporalBlockMask(_AugmenttimeOp):
    """Contiguous per-channel time-block masking with probability `p`,
    backed by `WindowMaskAugmentation`.

    Args:
        p: Probability of applying the mask.
        block_size_ratio: Window size as a fraction of time steps.
    """

    def __init__(self, p: float = 0.5, block_size_ratio: float = 0.1) -> None:
        self.p = float(p)
        self.block_size_ratio = float(block_size_ratio)

    def _window_range(self, L: int) -> tuple[int, int]:
        size = max(1, int(L * self.block_size_ratio))
        return (size, size)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(0)
        B, C, L = x.shape
        inner = WindowMaskAugmentation(window_range=self._window_range(L))
        out = inner(x)
        return out.squeeze(0) if squeeze else out


class Smoothing(_AugmenttimeOp):
    """Gaussian smoothing with per-channel kernel sampling,
    backed by `SmoothingAugmentation`.

    Args:
        kernel_range: `(min, max)` odd kernel sizes.
        sigma: Gaussian kernel sigma.
    """

    def __init__(
        self,
        kernel_range: tuple[int, int] = (3, 15),
        sigma: float = 2.0,
    ) -> None:
        super().__init__(SmoothingAugmentation(kernel_range=kernel_range, sigma=sigma))


# The following ops were previously defined locally; the SSL pipeline no longer
# relies on them (TemporalCrop was off by default in `train_ssl.py`; Drift was a
# linear trend we dropped in favor of augmenttime's BiasAugmentation). They are
# kept as no-op identity transforms so legacy imports still resolve and older
# configs degrade gracefully.

class Drift(Transform):
    """Deprecated: linear drift was removed when switching to augmenttime.
    Kept as an identity transform for import compatibility.
    """

    def __init__(self, slope_range: tuple[float, float] = (-0.1, 0.1)) -> None:  # noqa: ARG002
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x


class TemporalCrop(Transform):
    """Random temporal crop + right-pad back to the original length.

    This is a view-level op (different crop per view) and is the strongest
    single TS-SSL augmentation: it forces the encoder to be invariant to
    time alignment, which is exactly what forecasting requires.

    Args:
        output_length: Target length. The crop is drawn from
            ``[output_length, output_length * crop_ratio_range[1]]`` along
            time. The output is right-padded with the last value to keep the
            shape ``[C, output_length]`` (or ``[B, C, output_length]``).
        crop_ratio_range: ``(min_ratio, max_ratio)``. The crop length is
            ``output_length * ratio`` where ``ratio`` is uniform in the range.
            A range of ``(1.0, 1.0)`` is a no-op.
    """

    def __init__(
        self,
        output_length: int,
        crop_ratio_range: tuple[float, float] = (0.85, 1.0),
    ) -> None:
        self.output_length = int(output_length)
        lo, hi = crop_ratio_range
        if not (0.0 < lo <= hi <= 1.0):
            raise ValueError(
                f"crop_ratio_range must satisfy 0 < lo <= hi <= 1, got {crop_ratio_range}"
            )
        self.crop_ratio_range = (float(lo), float(hi))

    def _sample_ratio(self) -> float:
        import random
        lo, hi = self.crop_ratio_range
        if hi <= lo:
            return lo
        return random.uniform(lo, hi)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError(f"TemporalCrop expects 2D or 3D input, got {x.dim()}D")

        B, C, T = x.shape
        if T != self.output_length:
            # Crop is anchored to the configured output length; if the incoming
            # tensor doesn't match (e.g. already-cropped view) we resize by
            # random slicing and skip the right-pad.
            ratio = self._sample_ratio()
            new_len = max(1, int(round(T * ratio)))
            start = int(torch.randint(0, T - new_len + 1, (1,)).item()) if T > new_len else 0
            return x[:, :, start : start + new_len].contiguous()

        ratio = self._sample_ratio()
        new_len = max(1, int(round(T * ratio)))
        start = int(torch.randint(0, T - new_len + 1, (1,)).item())
        cropped = x[:, :, start : start + new_len]
        # Right-pad with the last column to keep shape [B, C, output_length].
        pad_len = self.output_length - new_len
        if pad_len > 0:
            pad = cropped[:, :, -1:].expand(-1, -1, pad_len)
            cropped = torch.cat([cropped, pad], dim=-1)
        return cropped.squeeze(0) if squeeze else cropped


# Re-exported so ``from timeseries.transforms.ops import F`` keeps working.
F_module = F  # silence "imported but unused" if linters get strict.


__all__ = [
    "Scaling",
    "Bias",
    "FeatureJitter",
    "AddGaussianNoise",
    "FrequencyMask",
    "MagnitudeWarp",
    "TemporalBlockMask",
    "Smoothing",
    "Drift",
    "TemporalCrop",
]