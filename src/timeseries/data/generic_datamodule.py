from __future__ import annotations

"""Generic DataModule for time-series SSL that works with any [C, T] tensor."""

from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
import lightning as L
from omegaconf import DictConfig

from ..data.window_dataset import WindowDataset
from ..data.view_dataset import ViewDataset, collate_batches
from ..data.view_builders import AugmentationViewBuilder
from .pems import _resolve_temporal_crop_ratio
from ..transforms.base import Transform
from ..transforms.compose import Compose, RandomApply
from ..transforms.ops import (
    AddGaussianNoise,
    Bias,
    FeatureJitter,
    FrequencyMask,
    MagnitudeWarp,
    Scaling,
    TemporalBlockMask,
)
from ..preprocessing.scalers import TorchStandardScaler
from ..preprocessing.time_encoding import encode_timestamps_torch


class IdentityTransform(Transform):
    """No-op transform."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x


class SimpleScaler:
    """Lightweight scaler with inverse_transform for compatibility."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def to(self, device: torch.device) -> "SimpleScaler":
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


class TimeSeriesSSLDataModule(L.LightningDataModule):
    """Generic SSL DataModule accepting any [C, T] tensor + optional metadata.

    Supports loading from:
        - Direct tensor input via `data` parameter
        - .npz files (expects 'data' key with shape [T, C, F] or [T, C])
        - .csv files (all numeric columns treated as channels)

    Features:
        - Temporal or random-window splits
        - TorchStandardScaler or SimpleScaler
        - Compound augmentation pipeline (Phase 1.1 compatible)
        - ViewBuilder integration with per-view augmentation diversity
        - Optional time features via timestamps

    Example:
        >>> cfg = DictConfig({
        ...     'window_size': 96,
        ...     'target_window_size': 12,
        ...     'batch_size': 32,
        ...     'train_split': 0.8,
        ... })
        >>> # From tensor
        >>> dm = TimeSeriesSSLDataModule(cfg, data=my_tensor)
        >>> # From file
        >>> dm = TimeSeriesSSLDataModule(cfg, data_path='data/my_dataset.npz')
    """

    def __init__(
        self,
        cfg: DictConfig | dict,
        data: Optional[torch.Tensor] = None,
        timestamps: Optional[pd.DatetimeIndex] = None,
        data_path: Optional[str] = None,
    ):
        super().__init__()
        self.cfg = cfg if isinstance(cfg, dict) else dict(cfg)
        self._raw_data = data
        self._timestamps = timestamps
        self._data_path = data_path or self.cfg.get("data_path", None)
        self.save_hyperparameters(ignore=["data", "timestamps"])

        self.scaler = TorchStandardScaler()

        # Build augmentation transform
        self.transform = self._build_transform()

    def _build_transform(self) -> Transform:
        """Build SSL augmentation pipeline from config.

        All augmentations are backed by the ``augmenttime`` package.
        """
        cfg = self.cfg
        mode = str(cfg.get("augmentation_mode", "compound"))
        p_per_aug = float(cfg.get("p_per_aug", 0.4))

        ops: list[Transform] = [
            Scaling(scale_range=tuple(cfg.get("scale_range", (0.9, 1.1)))),
        ]

        bias_std = float(cfg.get("bias_std", 0.0))
        if bias_std > 0:
            ops.append(Bias(bias_std=bias_std))

        jitter_std = float(cfg.get("jitter_std", 0.05))
        if jitter_std > 0:
            ops.append(FeatureJitter(jitter_std=jitter_std))

        p_noise = float(cfg.get("p_noise", 0.3))
        if p_noise > 0:
            ops.append(AddGaussianNoise(p=p_noise))

        p_freq_mask = float(cfg.get("p_freq_mask", 0.0))
        if p_freq_mask > 0:
            ops.append(FrequencyMask(p=p_freq_mask, max_freq_ratio=float(cfg.get("max_freq_ratio", 0.1))))

        p_magnitude_warp = float(cfg.get("p_magnitude_warp", 0.3))
        if p_magnitude_warp > 0:
            ops.append(MagnitudeWarp(p=p_magnitude_warp))

        pipeline: list[Transform] = []
        if mode == "compound":
            for op in ops:
                pipeline.append(RandomApply(op, p=p_per_aug))
        else:
            from ..transforms.compose import OneOf
            p_transform = float(cfg.get("p_transform", 0.7))
            if ops and p_transform > 0:
                pipeline.append(RandomApply(OneOf(ops), p=p_transform))

        p_temporal_mask = float(cfg.get("p_temporal_mask", 0.3))
        if p_temporal_mask > 0:
            pipeline.append(TemporalBlockMask(p=p_temporal_mask))

        if not pipeline:
            return IdentityTransform()
        return Compose(pipeline)

    def _load_data(self) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Load data and optional time features.

        Returns:
            data_ct: Tensor of shape [C, T]
            time_features: Optional tensor of shape [T, F_time]
        """
        if self._raw_data is not None:
            data = self._raw_data
            if data.dim() == 2 and data.shape[0] > data.shape[1]:
                # Assume [T, C] format, transpose to [C, T]
                data = data.t()
            elif data.dim() == 2:
                # Already [C, T]
                pass
            else:
                raise ValueError(f"Expected 2D tensor, got shape {tuple(data.shape)}")

            time_features = None
            if self._timestamps is not None:
                time_features = encode_timestamps_torch(self._timestamps)
            return data, time_features

        if self._data_path is None:
            raise ValueError("Must provide either `data` tensor or `data_path`")

        if self._data_path.endswith(".npz"):
            fp = np.load(self._data_path)
            if "data" in fp:
                raw = fp["data"]
            else:
                raw = fp[list(fp.keys())[0]]
            fp.close()

            if raw.ndim == 3:
                raw = raw[..., 0]  # [T, C]
            data_tc = torch.tensor(raw, dtype=torch.float32)  # [T, C]

        elif self._data_path.endswith(".csv"):
            df = pd.read_csv(self._data_path)
            if "date" in df.columns:
                df = df.drop(columns=["date"])
            data_tc = torch.tensor(df.values, dtype=torch.float32)  # [T, C]

        else:
            raise ValueError(f"Unsupported file format: {self._data_path}")

        # [T, C] -> [C, T]
        data_ct = data_tc.t()
        time_features = None
        if self._timestamps is not None:
            time_features = encode_timestamps_torch(self._timestamps)
        return data_ct, time_features

    def _make_view_dataset(
        self, data_ct: torch.Tensor, time_features: Optional[torch.Tensor], *, repeat_factor: int
    ) -> ViewDataset:
        """Create a ViewDataset from [C, T] data."""
        cfg = self.cfg
        include_prev_raw = cfg.get("include_prev", 0)
        if isinstance(include_prev_raw, bool):
            include_prev = 1 if include_prev_raw else 0
        else:
            include_prev = int(include_prev_raw)

        prev_shift = int(cfg.get("prev_shift", cfg.get("temporal_shift", 10)))
        horizon = int(cfg.get("target_window_size", 0))
        window_ds_prev = min(1, include_prev)

        window_ds = WindowDataset(
            data_ct,
            time_features,
            window_size=int(cfg.get("window_size", 96)),
            stride=int(cfg.get("stride", 1)),
            include_prev=window_ds_prev,
            prev_shift=prev_shift,
            horizon=horizon,
        )

        include_clean_t0 = bool(cfg.get("include_clean_t0", True))

        # Per-view temporal crop: strongest single TS-SSL augmentation.
        # Disabled when the ratio range is (1.0, 1.0). We prefer the new
        # ``temporal_crop_ratio_range`` key to avoid colliding with the legacy
        # amplitude-style ``crop_ratio_range`` (e.g. ``(0.8, 1.2)``); we fall
        # back to ``crop_ratio_range`` only when it actually looks like a ratio.
        crop_ratio = _resolve_temporal_crop_ratio(cfg)
        temporal_crop = None
        if crop_ratio is not None and crop_ratio != (1.0, 1.0):
            from ..transforms.ops import TemporalCrop
            temporal_crop = TemporalCrop(
                output_length=int(cfg.get("window_size", 96)),
                crop_ratio_range=crop_ratio,
            )

        view_builder = AugmentationViewBuilder(
            transform=self.transform,
            repeat_factor=int(repeat_factor),
            num_prev=include_prev,
            include_clean_t0=include_clean_t0,
            temporal_crop=temporal_crop,
        )

        return ViewDataset(window_ds, view_builder)

    def setup(self, stage: Optional[str] = None) -> None:
        """Split data and create train/val datasets."""
        data_ct, time_features = self._load_data()

        split_mode = str(self.cfg.get("split_mode", "temporal"))
        split_frac = float(self.cfg.get("train_split", 0.8))
        split_seed = int(self.cfg.get("split_seed", 0))
        repeat_factor_train = int(self.cfg.get("repeat_factor", 2))
        repeat_factor_val = int(self.cfg.get("repeat_factor_val", repeat_factor_train))

        C, T = data_ct.shape

        if split_mode == "random_windows":
            # Fix: fit scaler on train-portion timesteps only to prevent leakage
            # data_ct is [C, T], scaler expects [T, C]
            n_scaler = int(T * split_frac)
            self.scaler.fit(data_ct[:, :n_scaler].t())
            full_scaled_3d = self.scaler.transform(data_ct.t())
            full_scaled_ct = full_scaled_3d.squeeze(0)  # [C, T]

            base_train = self._make_view_dataset(full_scaled_ct, time_features, repeat_factor=repeat_factor_train)
            base_val = self._make_view_dataset(full_scaled_ct, time_features, repeat_factor=repeat_factor_val)

            n_windows = len(base_train)
            n_train = int(n_windows * split_frac)
            g = torch.Generator().manual_seed(split_seed)
            perm = torch.randperm(n_windows, generator=g).tolist()
            train_idx = perm[:n_train]
            val_idx_candidates = perm[n_train:]

            # Fix: purge val windows that overlap with any train window
            stride = int(cfg.get("stride", 1))
            window_size = int(cfg.get("window_size", 96))
            prev_shift = int(cfg.get("prev_shift", cfg.get("temporal_shift", 10)))
            include_prev_raw = cfg.get("include_prev", 0)
            if isinstance(include_prev_raw, bool):
                num_prev = 1 if include_prev_raw else 0
            else:
                num_prev = min(1, int(include_prev_raw))
            base_offset = prev_shift * num_prev

            train_starts = {base_offset + i * stride for i in train_idx}
            val_idx = []
            for vi in val_idx_candidates:
                vs = base_offset + vi * stride
                if all(abs(vs - ts) >= window_size for ts in train_starts):
                    val_idx.append(vi)

            if not val_idx:
                import warnings
                warnings.warn(
                    f"random_windows: all val windows overlap with train windows "
                    f"(stride={stride}, window_size={window_size}). "
                    f"Consider using split_mode='temporal' or increasing stride.",
                    UserWarning,
                    stacklevel=2,
                )
                val_idx = val_idx_candidates

            self.train_ds = Subset(base_train, train_idx)
            self.val_ds = Subset(base_val, val_idx)
            return

        # Temporal split (default)
        n_train = int(T * split_frac)
        train_ct = data_ct[:, :n_train]
        val_ct = data_ct[:, n_train:]

        train_time = time_features[:n_train] if time_features is not None else None
        val_time = time_features[n_train:] if time_features is not None else None

        # Fit scaler on training data [T_train, C]
        train_scaled_3d = self.scaler.fit_transform(train_ct.t())
        val_scaled_3d = self.scaler.transform(val_ct.t())
        train_scaled_ct = train_scaled_3d.squeeze(0)
        val_scaled_ct = val_scaled_3d.squeeze(0)

        self.train_ds = self._make_view_dataset(train_scaled_ct, train_time, repeat_factor=repeat_factor_train)
        self.val_ds = self._make_view_dataset(val_scaled_ct, val_time, repeat_factor=repeat_factor_val)

    def _build_collate_fn(self):
        """Compose ``collate_batches`` with optional batch-level channel mixup.

        Channel mixup is applied after collation so it can swap channels
        across samples in the same batch. Gated by ``cfg['channel_mixup_p']``.
        Val is never mixed.

        The clean t0 index used as the partner pool is ``cfg['include_prev']``.
        """
        from .view_dataset import BatchChannelMixup
        p = float(self.cfg.get("channel_mixup_p", 0.0))
        if p <= 0.0:
            return collate_batches
        max_channels = self.cfg.get("channel_mixup_max_channels", None)
        try:
            max_channels = int(max_channels) if max_channels is not None else None
        except (TypeError, ValueError):
            max_channels = None
        clean_t0_idx = int(self.cfg.get("include_prev", 0) or 0)
        mixup = BatchChannelMixup(p=p, max_channels=max_channels, clean_t0_index=clean_t0_idx)

        def _collate(items):
            return mixup(collate_batches(items))

        return _collate

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=int(self.cfg.get("batch_size", 32)),
            shuffle=True,
            num_workers=int(self.cfg.get("num_workers", 0)),
            collate_fn=self._build_collate_fn(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=int(self.cfg.get("batch_size", 32)),
            shuffle=False,
            num_workers=int(self.cfg.get("num_workers", 0)),
            collate_fn=collate_batches,
        )


# Keep backward compatibility
GenericTimeSeriesDataModule = TimeSeriesSSLDataModule
