import os
import urllib.request
import zipfile
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from torch_geometric.utils import dense_to_sparse
import lightning as L
from typing import Optional, List
from omegaconf import DictConfig
from ..transforms.base import Transform
from ..transforms.compose import Compose, RandomApply, OneOf
from ..transforms.ops import (
    AddGaussianNoise,
    Bias,
    FeatureJitter,
    FrequencyMask,
    MagnitudeWarp,
    Scaling,
    TemporalBlockMask,
)
from .probe_dataset import ForecastProbeDataset
from .view_builders import AugmentationViewBuilder
from .view_dataset import ViewDataset, collate_batches
from .window_dataset import WindowDataset
from ..preprocessing.scalers import TorchStandardScaler
from ..preprocessing.time_encoding import encode_timestamps_torch, NUM_TIME_FEATURES
from dataclasses import dataclass


class IdentityTransform(Transform):
    """No-op transform for disabling augmentation."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _resolve_temporal_crop_ratio(cfg) -> tuple[float, float] | None:
    """Return the temporal-crop ratio range from ``cfg`` or ``None``.

    The legacy ``crop_ratio_range`` key was overloaded: in some configs it
    is a temporal-crop ratio like ``(0.85, 1.0)``, in others it is an
    amplitude range like ``(0.8, 1.2)`` used by the old augmenttime
    amplitude-scaling op. The two cannot both be supported under the same
    name, so we introduce a dedicated ``temporal_crop_ratio_range`` key and
    fall back to ``crop_ratio_range`` only when it actually looks like a
    valid ratio (both endpoints in ``(0, 1]``).

    Returns ``None`` when the crop should be disabled.
    """
    def _as_pair(x):
        try:
            lo, hi = float(x[0]), float(x[1])
        except (TypeError, ValueError, IndexError):
            return None
        return lo, hi

    explicit = getattr(cfg, "temporal_crop_ratio_range", None)
    if explicit is not None:
        pair = _as_pair(explicit)
        if pair is None:
            return None
        lo, hi = pair
        if not (0.0 < lo <= hi <= 1.0):
            return None  # invalid; fall back to legacy only if also invalid
        return pair

    legacy = getattr(cfg, "crop_ratio_range", None)
    if legacy is None:
        return None
    pair = _as_pair(legacy)
    if pair is None:
        return None
    lo, hi = pair
    # Only treat the legacy value as a crop ratio if it is bounded by 1.0.
    # An amplitude range like (0.8, 1.2) is rejected silently (returns None)
    # and the crop is disabled — better than crashing on a stale config.
    if not (0.0 < lo <= hi <= 1.0):
        return None
    return pair


def build_ssl_transform(
    *,
    output_length: int,
    scale_range: tuple[float, float] = (1.0, 1.0),
    crop_ratio_range: tuple[float, float] = (1.0, 1.0),
    jitter_std: float = 0.0,
    p_noise: float = 0.0,
    p_freq_mask: float = 0.0,
    max_freq_ratio: float = 0.0,
    p_temporal_mask: float = 0.0,
    p_magnitude_warp: float = 0.0,
    p_temporal_crop: float = 0.0,
    p_transform: float = 0.0,
    augmentation_mode: str = "compound",
    p_per_aug: float = 0.4,
    bias_std: float = 0.0,
) -> Transform:
    """Build a configurable SSL augmentation pipeline using the new transform stack.

    All augmentations are backed by the ``augmenttime`` package.

    Args:
        augmentation_mode: 'compound' (default) applies each augmentation independently
            with probability p_per_aug, allowing multiple transforms to compose.
            'oneof' (legacy) selects at most one augmentation per view.
        p_per_aug: Per-augmentation probability in compound mode (default 0.4).
        p_transform: Overall probability of applying the OneOf block in legacy mode.
        bias_std: Std of the per-channel bias augmentation (replaces the legacy
            Drift op). Off by default; set >0 to enable.
        output_length, crop_ratio_range, p_temporal_crop: Kept for config
            backwards compatibility; ignored since augmenttime does not provide
            a temporal crop primitive.
    """
    ops = [Scaling(scale_range=scale_range)]

    if bias_std and bias_std > 0:
        ops.append(Bias(bias_std=bias_std))
    if jitter_std and jitter_std > 0:
        ops.append(FeatureJitter(jitter_std=jitter_std))
    if p_noise and p_noise > 0:
        ops.append(AddGaussianNoise(p=p_noise))
    if p_freq_mask and p_freq_mask > 0:
        ops.append(FrequencyMask(p=p_freq_mask, max_freq_ratio=max_freq_ratio))
    if p_magnitude_warp and p_magnitude_warp > 0:
        ops.append(MagnitudeWarp(p=p_magnitude_warp))

    pipeline = []

    if augmentation_mode == "compound":
        # Phase 1.1: Each augmentation fires independently with probability p_per_aug,
        # allowing multiple transforms to compose. This is the biggest single
        # improvement to SSL quality (analogous to SimCLR/BYOL augmentation stacking).
        for op in ops:
            pipeline.append(RandomApply(op, p=p_per_aug))
    else:
        # Legacy 'oneof' mode: selects at most one augmentation per view
        if ops and p_transform and p_transform > 0:
            pipeline.append(RandomApply(OneOf(ops), p=p_transform))

    if p_temporal_mask and p_temporal_mask > 0:
        pipeline.append(TemporalBlockMask(p=p_temporal_mask))

    if not pipeline:
        return IdentityTransform()

    return Compose(pipeline)

def download_url(url: str, root_dir: str) -> str:
    os.makedirs(root_dir, exist_ok=True)
    filename = url.split('/')[-1]
    filepath = os.path.join(root_dir, filename)
    if os.path.exists(filepath):
        return filepath
    urllib.request.urlretrieve(url, filepath)
    return filepath

def extract_zip(zip_path: str, extract_dir: str) -> None:
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)

# @dataclass
# class batch:
    
class PeMS08:
    name = 'PeMS08'
    start_date = '07-01-2016 00:00'
    num_sensors = 170
    url = 'https://drive.switch.ch/index.php/s/X0nkiDgb8oacOD0/download'

    def __init__(self, root: str = "./data/pems08", similarity: str = "distance", threshold: float | None = None):
        self.root = root
        self.similarity = similarity
        self.threshold = threshold
        self.target = None
        self.dist = None
        self.edge_index = None
        self.edge_weight = None
        self._maybe_download()
        self._load_data()
        self._build_graph()

    def _maybe_download(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        npz_path = os.path.join(self.root, 'pems08.npz')
        dist_path = os.path.join(self.root, 'distance.csv')
        if os.path.exists(npz_path) and os.path.exists(dist_path):
            return
        zip_path = download_url(self.url, self.root)
        extract_zip(zip_path, self.root)
        os.remove(zip_path)

    def _load_data(self) -> None:
        npz_path = os.path.join(self.root, 'pems08.npz')
        dist_path = os.path.join(self.root, 'distance.csv')
        fp = np.load(npz_path)
        data = fp['data']
        fp.close()
        index = pd.date_range(start=self.start_date, periods=len(data), freq='5min')
        self.target = pd.DataFrame(data[..., 0], index=index, dtype='float32')
        self.timestamps = index  # Store for time encoding
        distances = pd.read_csv(dist_path)
        self.dist = np.full((self.num_sensors, self.num_sensors), np.inf, dtype=np.float32)
        for src, dst, d in distances.values:
            self.dist[int(src), int(dst)] = d

    def _build_graph(self):
        adj = self.compute_similarity(self.similarity)
        if self.threshold is not None:
            adj = np.where(adj >= self.threshold, adj, 0.0)
        adj = torch.tensor(adj, dtype=torch.float32)
        edge_index, edge_weight = dense_to_sparse(adj)
        self.edge_index = edge_index
        self.edge_weight = edge_weight

    def compute_similarity(self, method: str):
        if method == 'distance':
            finite = self.dist[np.isfinite(self.dist)]
            sigma = finite.std()
            # NOTE: If `sigma` is 0 (or extremely small), this can blow up / produce NaNs.
            # Probably rare for real distance matrices, but worth guarding if you swap datasets.
            return np.exp(-np.square(self.dist / sigma))
        return (~np.isinf(self.dist)).astype('float32')

    def get_graph(self, device=None):
        if device is None:
            return self.edge_index, self.edge_weight
        return self.edge_index.to(device), self.edge_weight.to(device)

class PeMS08SSLDataModule(L.LightningDataModule):
    """PeMS08 SSL DataModule using past-only windows and view builders.

    This module constructs t0 views plus optional t-1 view (no t+1).
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        self.scaler = TorchStandardScaler()

        self.transform = build_ssl_transform(
            output_length=self.cfg.window_size,
            scale_range=getattr(self.cfg, "scale_range", (1.0, 1.0)),
            crop_ratio_range=getattr(self.cfg, "crop_ratio_range", (1.0, 1.0)),
            jitter_std=getattr(self.cfg, "jitter_std", 0.1),
            p_noise=getattr(self.cfg, "p_noise", 0.3),
            p_freq_mask=getattr(self.cfg, "p_freq_mask", 0.0),
            max_freq_ratio=getattr(self.cfg, "max_freq_ratio", 0.0),
            p_temporal_mask=getattr(self.cfg, "p_temporal_mask", 0.8),
            p_magnitude_warp=getattr(self.cfg, "p_magnitude_warp", 0.3),
            p_temporal_crop=getattr(self.cfg, "p_temporal_crop", 0.0),
            p_transform=getattr(self.cfg, "p_transform", 0.7),
            augmentation_mode=getattr(self.cfg, "augmentation_mode", "compound"),
            p_per_aug=getattr(self.cfg, "p_per_aug", 0.4),
        )

    def prepare_data(self):
        PeMS08(root="./data/pems08")

    def _make_view_dataset(self, data: torch.Tensor, time_features: torch.Tensor, *, repeat_factor: int) -> ViewDataset:
        # Handle include_prev as bool or int
        include_prev_raw = getattr(self.cfg, "include_prev", True)
        if isinstance(include_prev_raw, bool):
            include_prev = 1 if include_prev_raw else 0
        else:
            include_prev = int(include_prev_raw)
        prev_shift = int(getattr(self.cfg, "prev_shift", getattr(self.cfg, "temporal_shift", 10)))
        horizon = int(getattr(self.cfg, "target_window_size", 0))

        # WindowDataset only needs 1 prev window (t-1) since the view builder
        # creates num_prev augmented copies of just the t-1 window.
        window_ds_prev = min(1, include_prev)

        window_ds = WindowDataset(
            data,
            time_features,
            window_size=int(self.cfg.window_size),
            stride=int(getattr(self.cfg, "stride", 1)),
            include_prev=window_ds_prev,
            prev_shift=prev_shift,
            horizon=horizon,
        )

        # Always include clean t0 for proper probe training
        include_clean_t0 = bool(getattr(self.cfg, "include_clean_t0", True))

        # Per-view temporal crop: strongest single TS-SSL augmentation.
        # Disabled when the ratio range is (1.0, 1.0). We prefer the new
        # ``temporal_crop_ratio_range`` key to avoid colliding with the legacy
        # amplitude-style ``crop_ratio_range`` (e.g. ``(0.8, 1.2)`` in older
        # notebooks); if that key is missing or invalid as a ratio, fall back
        # to ``crop_ratio_range`` only when it actually looks like a ratio.
        crop_ratio = _resolve_temporal_crop_ratio(self.cfg)
        temporal_crop = None
        if crop_ratio is not None and crop_ratio != (1.0, 1.0):
            from ..transforms.ops import TemporalCrop
            temporal_crop = TemporalCrop(
                output_length=int(self.cfg.window_size),
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

    def setup(self, stage=None):
        pems08 = PeMS08(root="./data/pems08")
        data = torch.tensor(pems08.target.values, dtype=torch.float32)  # [T, C]

        time_features = encode_timestamps_torch(pems08.timestamps)  # [T, 6]

        split_mode = getattr(self.cfg, "split_mode", None)
        if split_mode is None:
            split_mode = "random_windows" if bool(getattr(self.cfg, "random_split", False)) else "temporal"

        split_seed = int(getattr(self.cfg, "split_seed", 0))
        split_frac = float(getattr(self.cfg, "train_split", 0.8))

        if split_mode not in {"temporal", "random_windows", "ts_cv"}:
            raise ValueError(f"Unknown cfg.split_mode={split_mode!r}. Use 'temporal', 'random_windows', or 'ts_cv'.")

        if split_mode in {"temporal", "random_windows"} and not (0.0 < split_frac < 1.0):
            raise ValueError(f"cfg.train_split must be in (0, 1), got {split_frac}")

        repeat_factor_train = int(getattr(self.cfg, "repeat_factor", 2))
        repeat_factor_val = int(getattr(self.cfg, "repeat_factor_val", repeat_factor_train))

        if split_mode == "random_windows":
            # Fix: fit scaler on train-portion timesteps only to prevent leakage
            n_scaler = int(len(data) * split_frac)
            self.scaler.fit(data[:n_scaler])
            full_scaled_3d = self.scaler.transform(data)
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
            # Window i starts at offset (prev_shift * num_prev) + i * stride
            stride = int(getattr(self.cfg, "stride", 1))
            window_size = int(self.cfg.window_size)
            prev_shift = int(getattr(self.cfg, "prev_shift", getattr(self.cfg, "temporal_shift", 10)))
            num_prev = min(1, int(getattr(self.cfg, "include_prev", 0)) if not isinstance(getattr(self.cfg, "include_prev", 0), bool) else (1 if getattr(self.cfg, "include_prev", 0) else 0))
            base_offset = prev_shift * num_prev

            train_starts = {base_offset + i * stride for i in train_idx}
            val_idx = []
            for vi in val_idx_candidates:
                vs = base_offset + vi * stride
                # Check if any train window start is within window_size of this val start
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
                val_idx = val_idx_candidates  # fall back to allow training to proceed

            self.train_ds = Subset(base_train, train_idx)
            self.val_ds = Subset(base_val, val_idx)
            return

        if split_mode == "ts_cv":
            n_splits = int(getattr(self.cfg, "cv_folds", 5))
            fold = int(getattr(self.cfg, "cv_fold", 0))
            gap = int(getattr(self.cfg, "cv_gap", self.cfg.window_size))

            if n_splits < 2:
                raise ValueError(f"cfg.cv_folds must be >= 2, got {n_splits}")
            if not (0 <= fold < n_splits):
                raise ValueError(f"cfg.cv_fold must be in [0, {n_splits - 1}], got {fold}")
            if gap < 0:
                raise ValueError(f"cfg.cv_gap must be >= 0, got {gap}")

            T = len(data)
            test_size = int(getattr(self.cfg, "cv_test_size", 0))
            if test_size <= 0:
                test_size = T // (n_splits + 1)

            train_end = (fold + 1) * test_size
            val_start = train_end + gap
            val_end = min(val_start + test_size, T)

            train_raw = data[:train_end]
            val_raw = data[val_start:val_end]
            train_time = time_features[:train_end]
            val_time = time_features[val_start:val_end]

            train_scaled_3d = self.scaler.fit_transform(train_raw)
            val_scaled_3d = self.scaler.transform(val_raw)
            train_scaled_ct = train_scaled_3d.squeeze(0)
            val_scaled_ct = val_scaled_3d.squeeze(0)

            self.train_ds = self._make_view_dataset(train_scaled_ct, train_time, repeat_factor=repeat_factor_train)
            self.val_ds = self._make_view_dataset(val_scaled_ct, val_time, repeat_factor=repeat_factor_val)
            return

        # Temporal split (default)
        n_train = int(len(data) * split_frac)
        train_raw = data[:n_train]
        val_raw = data[n_train:]
        train_time = time_features[:n_train]
        val_time = time_features[n_train:]

        train_scaled_3d = self.scaler.fit_transform(train_raw)
        val_scaled_3d = self.scaler.transform(val_raw)
        train_scaled_ct = train_scaled_3d.squeeze(0)
        val_scaled_ct = val_scaled_3d.squeeze(0)

        self.train_ds = self._make_view_dataset(train_scaled_ct, train_time, repeat_factor=repeat_factor_train)
        self.val_ds = self._make_view_dataset(val_scaled_ct, val_time, repeat_factor=repeat_factor_val)

    def _build_collate_fn(self):
        """Compose ``collate_batches`` with optional batch-level channel mixup.

        Channel mixup is applied after collation so it can swap channels
        across samples in the same batch. It is gated by ``cfg.channel_mixup_p``
        (set to 0.0 to disable). Val is never mixed.

        The clean t0 index used as the partner pool is ``cfg.include_prev``
        (the view builder places the clean t0 immediately after the
        ``include_prev`` augmented t-1 views).
        """
        from .view_dataset import BatchChannelMixup
        p = float(getattr(self.cfg, "channel_mixup_p", 0.0))
        if p <= 0.0:
            return collate_batches
        max_channels = getattr(self.cfg, "channel_mixup_max_channels", None)
        try:
            max_channels = int(max_channels) if max_channels is not None else None
        except (TypeError, ValueError):
            max_channels = None
        clean_t0_idx = int(getattr(self.cfg, "include_prev", 0) or 0)
        mixup = BatchChannelMixup(p=p, max_channels=max_channels, clean_t0_index=clean_t0_idx)

        def _collate(items):
            return mixup(collate_batches(items))

        return _collate

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=getattr(self.cfg, "num_workers", 0),
            collate_fn=self._build_collate_fn(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,  # Validation must not shuffle for reproducible metrics
            num_workers=getattr(self.cfg, "num_workers", 0),
            collate_fn=collate_batches,  # Val never gets channel mixup
        )


class PeMS08ProbeDataModule(L.LightningDataModule):
    """PeMS08 DataModule for forecasting probe evaluation."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.scaler = TorchStandardScaler()

    def prepare_data(self):
        PeMS08(root="./data/pems08")

    def setup(self, stage=None):
        pems08 = PeMS08(root="./data/pems08")
        data = torch.tensor(pems08.target.values, dtype=torch.float32)  # [T, C]
        time_features = encode_timestamps_torch(pems08.timestamps)  # [T, 6]

        split_frac = float(getattr(self.cfg, "train_split", 0.8))
        n_train = int(len(data) * split_frac)

        train_raw = data[:n_train]
        val_raw = data[n_train:]
        train_time = time_features[:n_train]
        val_time = time_features[n_train:]

        train_scaled_3d = self.scaler.fit_transform(train_raw)
        val_scaled_3d = self.scaler.transform(val_raw)
        train_scaled_ct = train_scaled_3d.squeeze(0)
        val_scaled_ct = val_scaled_3d.squeeze(0)

        window_size = int(self.cfg.window_size)
        horizon = int(self.cfg.target_window_size)
        stride = int(getattr(self.cfg, "stride", 1))

        self.train_ds = ForecastProbeDataset(
            train_scaled_ct,
            train_time,
            window_size=window_size,
            horizon=horizon,
            stride=stride,
        )
        self.val_ds = ForecastProbeDataset(
            val_scaled_ct,
            val_time,
            window_size=window_size,
            horizon=horizon,
            stride=stride,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=getattr(self.cfg, "num_workers", 0),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=getattr(self.cfg, "num_workers", 0),
        )

