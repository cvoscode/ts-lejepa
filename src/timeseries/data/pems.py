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
    Scaling,
    Drift,
    FeatureJitter,
    AddGaussianNoise,
    FrequencyMask,
    MagnitudeWarp,
    TemporalCrop,
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
) -> Transform:
    """Build a configurable SSL augmentation pipeline using the new transform stack.

    Args:
        augmentation_mode: 'compound' (default) applies each augmentation independently
            with probability p_per_aug, allowing multiple transforms to compose.
            'oneof' (legacy) selects at most one augmentation per view.
        p_per_aug: Per-augmentation probability in compound mode (default 0.4).
        p_transform: Overall probability of applying the OneOf block in legacy mode.
    """
    ops = [Scaling(scale_range=scale_range), Drift()]

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

    if p_temporal_crop and p_temporal_crop > 0:
        pipeline.append(
            RandomApply(
                TemporalCrop(output_length=output_length, crop_ratio_range=crop_ratio_range),
                p=p_temporal_crop,
            )
        )

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

    def __init__(self, root: str = "./data/pems08", mask_zeros: bool = True, similarity: str = "distance", threshold: float | None = None):
        self.root = root
        self.mask_zeros = mask_zeros
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
        PeMS08(root="./data/pems08", mask_zeros=True)

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
        
        view_builder = AugmentationViewBuilder(
            transform=self.transform,
            repeat_factor=int(repeat_factor),
            num_prev=include_prev,
            include_clean_t0=include_clean_t0,
        )

        return ViewDataset(window_ds, view_builder)

    def setup(self, stage=None):
        pems08 = PeMS08(root="./data/pems08", mask_zeros=True)
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
            full_scaled_3d = self.scaler.fit_transform(data)
            full_scaled_ct = full_scaled_3d.squeeze(0)  # [C, T]

            base_train = self._make_view_dataset(full_scaled_ct, time_features, repeat_factor=repeat_factor_train)
            base_val = self._make_view_dataset(full_scaled_ct, time_features, repeat_factor=repeat_factor_val)

            n_windows = len(base_train)
            n_train = int(n_windows * split_frac)
            g = torch.Generator().manual_seed(split_seed)
            perm = torch.randperm(n_windows, generator=g).tolist()
            train_idx = perm[:n_train]
            val_idx = perm[n_train:]
            self.train_ds = Subset(base_train, train_idx)
            self.val_ds = Subset(base_val, val_idx)
            return

        if split_mode == "ts_cv":
            n_splits = int(getattr(self.cfg, "cv_folds", 5))
            fold = int(getattr(self.cfg, "cv_fold", 0))
            gap = int(getattr(self.cfg, "cv_gap", 0))

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

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=getattr(self.cfg, "num_workers", 0),
            collate_fn=collate_batches,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,  # Validation must not shuffle for reproducible metrics
            num_workers=getattr(self.cfg, "num_workers", 0),
            collate_fn=collate_batches,
        )


class PeMS08ProbeDataModule(L.LightningDataModule):
    """PeMS08 DataModule for forecasting probe evaluation."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.scaler = TorchStandardScaler()

    def prepare_data(self):
        PeMS08(root="./data/pems08", mask_zeros=True)

    def setup(self, stage=None):
        pems08 = PeMS08(root="./data/pems08", mask_zeros=True)
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

