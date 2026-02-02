import os
import urllib.request
import zipfile
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
from torch_geometric.utils import dense_to_sparse
import random
import lightning as L
from typing import Optional, List
from omegaconf import DictConfig
from ..augementation.v1 import TimeSeriesTransform
from .probe_dataset import ForecastProbeDataset
from .view_builders import AugmentationViewBuilder
from .view_dataset import ViewDataset, collate_batches
from .window_dataset import WindowDataset
from ..preprocessing.scalers import TorchStandardScaler
from ..preprocessing.time_encoding import encode_timestamps_torch, NUM_TIME_FEATURES
from dataclasses import dataclass

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

class PeMS08AugmentedDataset(Dataset):
    def __init__(
        self,
        pems08: torch.Tensor,
        time_features: torch.Tensor,
        transform,
        window_size: int = 512,
        target_window_size: int = 12,
        temporal_shift: int = 10,
        stride: int = 1,
        sensors=None,
        repeat_factor: int = 1,
    ):
        """
        Args:
            pems08: Sensor data tensor [C, T]
            time_features: Time encoding tensor [T, 6] with cyclical features
            transform: Augmentation transform for sensor data
            window_size: Length of input windows
            target_window_size: Forecast horizon
            temporal_shift: Shift between t-1, t0, t+1 views
            stride: Stride for sliding window
            sensors: List of sensor indices to use (None = all)
        """
        super().__init__()
        self.data = pems08
        self.time_features = time_features  # [T, 6]
        self.C, self.T = self.data.shape
        self.window_size = window_size
        self.target_window_size = target_window_size
        self.temporal_shift = temporal_shift
        self.stride = stride
        self.sensors = sensors if sensors is not None else list(range(self.C))
        self.transform = transform
        self.total_input_span = self.temporal_shift + self.window_size
        self.total_span = max(self.total_input_span, self.window_size + self.target_window_size)
        self.num_windows = (self.T - self.total_span - self.temporal_shift) // stride + 1
        self.num_windows = max(0, self.num_windows)
        self.repeat_factor = int(repeat_factor)
        if self.repeat_factor < 1:
            raise ValueError(f"repeat_factor must be >= 1, got {self.repeat_factor}")

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        base_start = self.temporal_shift + (idx * self.stride)
        
        # Sensor data windows
        start_prev = base_start - self.temporal_shift
        t_prev = self.data[self.sensors, start_prev : start_prev + self.window_size]
        t_curr = self.data[self.sensors, base_start : base_start + self.window_size]
        start_next = base_start + self.temporal_shift
        t_next = self.data[self.sensors, start_next : start_next + self.window_size]
        target_start = base_start + self.window_size
        target = self.data[self.sensors, target_start : target_start + self.target_window_size]
        
        # Apply transforms to sensor data.
        # View order matters: the model expects (prev, curr..., next).
        view_prev = self.transform(t_prev.clone())
        view_next = self.transform(t_next.clone())
        curr_views = [self.transform(t_curr.clone()) for _ in range(self.repeat_factor)]
        views = torch.stack([view_prev] + curr_views + [view_next])  # [V, C, L], V = 2 + repeat_factor
        
        # Time features for each view window [3, L, 6]
        time_prev = self.time_features[start_prev : start_prev + self.window_size]
        time_curr = self.time_features[base_start : base_start + self.window_size]
        time_next = self.time_features[start_next : start_next + self.window_size]
        view_times = torch.stack([time_prev] + [time_curr for _ in range(self.repeat_factor)] + [time_next])  # [V, L, 6]
        
        # Future time features for forecast horizon [H, 6]
        future_times = self.time_features[target_start : target_start + self.target_window_size]
        
        # NOTE: Target is returned as [H, C] (via `.t()`) so the batch becomes [B, H, C].
        # The LightningModule later transposes to [B, C, H] if needed.
        return views, target.t(), view_times, future_times

class PeMS08DataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.aug_transform = TimeSeriesTransform(
            output_length=self.cfg.window_size, 
            scale_range=getattr(self.cfg, "scale_range", (1.0, 1.0)), 
            jitter_std=getattr(self.cfg, "jitter_std", 0.1), 
            p_noise=getattr(self.cfg, "p_noise", 0.3), 
            p_freq_mask=getattr(self.cfg, "p_freq_mask", 0.0),
            max_freq_ratio=getattr(self.cfg, "max_freq_ratio", 0.0),
            p_temporal_mask=getattr(self.cfg, "p_temporal_mask", 0.0),
            p_transform=getattr(self.cfg, "p_transform", 0.7)
        )
        self.test_transform = TimeSeriesTransform(
            output_length=self.cfg.window_size, 
            scale_range=(1.0, 1.0),
            jitter_std=0.0, 
            p_noise=0.0, 
            p_freq_mask=0.0,
            max_freq_ratio=0.0,
            p_temporal_mask=0.0,
            p_transform=0.0
        )
        self.scaler = TorchStandardScaler()

    def prepare_data(self):
        PeMS08(root="./data/pems08", mask_zeros=True)

    def setup(self, stage=None):
        pems08 = PeMS08(root="./data/pems08", mask_zeros=True)
        data = torch.tensor(pems08.target.values, dtype=torch.float32)  # [T, C]

        # Time features for entire dataset
        time_features = encode_timestamps_torch(pems08.timestamps)  # [T, 6]

        # Split configuration
        # - split_mode='temporal' (default): first train_split fraction train, remainder val
        # - split_mode='random_windows': random split over window indices (diagnostic; can be leaky with stride=1)
        # - split_mode='ts_cv': expanding-window time series CV (TimeSeriesSplit-like) with optional gap
        split_mode = getattr(self.cfg, "split_mode", None)
        if split_mode is None:
            split_mode = "random_windows" if bool(getattr(self.cfg, "random_split", False)) else "temporal"

        split_seed = int(getattr(self.cfg, "split_seed", 0))
        split_frac = float(getattr(self.cfg, "train_split", 0.8))

        window_size = int(self.cfg.window_size)
        target_window_size = int(self.cfg.target_window_size)
        temporal_shift = int(getattr(self.cfg, "temporal_shift", 10))
        stride = int(getattr(self.cfg, "stride", 1))
        repeat_factor_train = int(getattr(self.cfg, "repeat_factor", 1))
        repeat_factor_val = int(getattr(self.cfg, "repeat_factor_val", 1))
        total_input_span = temporal_shift + window_size
        total_span = max(total_input_span, window_size + target_window_size)
        min_required_len = total_span + temporal_shift + 1

        if split_mode not in {"temporal", "random_windows", "ts_cv"}:
            raise ValueError(f"Unknown cfg.split_mode={split_mode!r}. Use 'temporal', 'random_windows', or 'ts_cv'.")

        if split_mode in {"temporal", "random_windows"} and not (0.0 < split_frac < 1.0):
            raise ValueError(f"cfg.train_split must be in (0, 1), got {split_frac}")

        if split_mode == "random_windows":
            # Random split over WINDOW INDICES (not time points) to preserve contiguous windows.
            full_scaled_3d = self.scaler.fit_transform(data)
            full_scaled_ct = full_scaled_3d.squeeze(0)  # [C, T]

            base_train = PeMS08AugmentedDataset(
                full_scaled_ct,
                time_features,
                transform=self.aug_transform,
                window_size=self.cfg.window_size,
                target_window_size=self.cfg.target_window_size,
                temporal_shift=getattr(self.cfg, "temporal_shift", 10),
                stride=getattr(self.cfg, "stride", 1),
                repeat_factor=repeat_factor_train,
            )
            base_val = PeMS08AugmentedDataset(
                full_scaled_ct,
                time_features,
                transform=self.test_transform,
                window_size=self.cfg.window_size,
                target_window_size=self.cfg.target_window_size,
                temporal_shift=getattr(self.cfg, "temporal_shift", 10),
                stride=getattr(self.cfg, "stride", 1),
                repeat_factor=repeat_factor_val,
            )

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
            # Expanding-window CV similar to sklearn.model_selection.TimeSeriesSplit.
            # Fold i: train=[0:train_end), val=[val_start:val_end)
            # with optional gap in between to reduce leakage from overlapping windows.
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
            if T < min_required_len * 2:
                raise ValueError(
                    f"Sequence too short for time-series split with current windowing. "
                    f"Need at least ~{min_required_len * 2} time steps, got {T}."
                )

            test_size = int(getattr(self.cfg, "cv_test_size", 0))
            if test_size <= 0:
                test_size = T // (n_splits + 1)
            if test_size < min_required_len:
                raise ValueError(
                    f"cv_test_size too small for windowing. Need >= {min_required_len}, got {test_size}."
                )

            train_end = (fold + 1) * test_size
            val_start = train_end + gap
            val_end = min(val_start + test_size, T)

            if train_end < min_required_len:
                raise ValueError(
                    f"Training segment too short for fold {fold}. train_end={train_end}, need >= {min_required_len}."
                )
            if (val_end - val_start) < min_required_len:
                raise ValueError(
                    f"Validation segment too short for fold {fold}. "
                    f"val_len={val_end - val_start}, need >= {min_required_len}. "
                    f"Try smaller cv_gap or larger cv_test_size."
                )

            train_raw = data[:train_end]
            val_raw = data[val_start:val_end]
            train_time = time_features[:train_end]
            val_time = time_features[val_start:val_end]

            train_scaled_3d = self.scaler.fit_transform(train_raw)
            val_scaled_3d = self.scaler.transform(val_raw)
            train_scaled_ct = train_scaled_3d.squeeze(0)
            val_scaled_ct = val_scaled_3d.squeeze(0)

            self.train_ds = PeMS08AugmentedDataset(
                train_scaled_ct,
                train_time,
                transform=self.aug_transform,
                window_size=window_size,
                target_window_size=target_window_size,
                temporal_shift=temporal_shift,
                stride=stride,
                repeat_factor=repeat_factor_train,
            )
            self.val_ds = PeMS08AugmentedDataset(
                val_scaled_ct,
                val_time,
                transform=self.test_transform,
                window_size=window_size,
                target_window_size=target_window_size,
                temporal_shift=temporal_shift,
                stride=stride,
                repeat_factor=repeat_factor_val,
            )
            return

        # Temporal split (default): first `train_split` portion for train, rest for val.
        n_train = int(len(data) * split_frac)
        train_raw = data[:n_train]
        test_raw = data[n_train:]
        train_time = time_features[:n_train]
        test_time = time_features[n_train:]

        train_scaled_3d = self.scaler.fit_transform(train_raw)
        test_scaled_3d = self.scaler.transform(test_raw)
        train_scaled_ct = train_scaled_3d.squeeze(0)  # [C, T_train]
        test_scaled_ct = test_scaled_3d.squeeze(0)    # [C, T_val]

        self.train_ds = PeMS08AugmentedDataset(
            train_scaled_ct,
            train_time,
            transform=self.aug_transform,
            window_size=window_size,
            target_window_size=target_window_size,
            temporal_shift=temporal_shift,
            stride=stride,
            repeat_factor=repeat_factor_train,
        )
        self.val_ds = PeMS08AugmentedDataset(
            test_scaled_ct,
            test_time,
            transform=self.test_transform,
            window_size=window_size,
            target_window_size=target_window_size,
            temporal_shift=temporal_shift,
            stride=stride,
            repeat_factor=repeat_factor_val,
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.cfg.batch_size, shuffle=True, num_workers=getattr(self.cfg, "num_workers", 0))

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.cfg.batch_size, shuffle=False, num_workers=getattr(self.cfg, "num_workers", 0))


class PeMS08MultiScaleViewsDataset(Dataset):
    """PeMS08 dataset that returns global/local *scale* views with identical output length.

    View convention:
      - index 0: previous window (t-1)
      - indices 1..(1 + num_local_views): current window views
          * index 1 is the "global" current view
          * following are "local" current views
      - last index: next window (t+1)

    All views have length `window_size` (the transforms enforce this via output_length).
    """

    def __init__(
        self,
        pems08: torch.Tensor,
        time_features: torch.Tensor,
        *,
        global_transform,
        local_transform,
        window_size: int = 512,
        target_window_size: int = 12,
        temporal_shift: int = 10,
        stride: int = 1,
        sensors=None,
        num_local_views: int = 6,
    ):
        super().__init__()
        self.data = pems08
        self.time_features = time_features  # [T, 6]
        self.C, self.T = self.data.shape
        self.window_size = int(window_size)
        self.target_window_size = int(target_window_size)
        self.temporal_shift = int(temporal_shift)
        self.stride = int(stride)
        self.sensors = sensors if sensors is not None else list(range(self.C))

        self.global_transform = global_transform
        self.local_transform = local_transform
        self.num_local_views = int(num_local_views)
        if self.num_local_views < 0:
            raise ValueError(f"num_local_views must be >= 0, got {self.num_local_views}")

        self.total_input_span = self.temporal_shift + self.window_size
        self.total_span = max(self.total_input_span, self.window_size + self.target_window_size)
        self.num_windows = (self.T - self.total_span - self.temporal_shift) // self.stride + 1
        self.num_windows = max(0, self.num_windows)

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        base_start = self.temporal_shift + (idx * self.stride)

        start_prev = base_start - self.temporal_shift
        start_next = base_start + self.temporal_shift
        target_start = base_start + self.window_size

        t_prev = self.data[self.sensors, start_prev : start_prev + self.window_size]
        t_curr = self.data[self.sensors, base_start : base_start + self.window_size]
        t_next = self.data[self.sensors, start_next : start_next + self.window_size]
        target = self.data[self.sensors, target_start : target_start + self.target_window_size]

        # Build views (all same output length).
        view_prev = self.global_transform(t_prev.clone())
        view_next = self.global_transform(t_next.clone())

        # Current window: one global-scale view + multiple local-scale views.
        curr_global = self.global_transform(t_curr.clone())
        curr_locals = [self.local_transform(t_curr.clone()) for _ in range(self.num_local_views)]
        views = torch.stack([view_prev, curr_global] + curr_locals + [view_next])

        # Time features per view
        time_prev = self.time_features[start_prev : start_prev + self.window_size]
        time_curr = self.time_features[base_start : base_start + self.window_size]
        time_next = self.time_features[start_next : start_next + self.window_size]
        view_times = torch.stack([time_prev] + [time_curr for _ in range(1 + self.num_local_views)] + [time_next])

        future_times = self.time_features[target_start : target_start + self.target_window_size]

        return views, target.t(), view_times, future_times


class PeMS08MultiScaleDataModule(L.LightningDataModule):
    """PeMS08 DataModule that creates global+local scale views of the current window.

    Compared to `PeMS08DataModule`, this does NOT use `repeat_factor`.
    Instead it uses:
      - `num_local_views`: number of local-scale views for t0
      - `global_crop_ratio_range`: crop ratio range for the global view (close to 1.0)
      - `local_crop_ratio_range`: crop ratio range for local views (smaller)
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        window_size = int(self.cfg.window_size)

        # Global (mild) view of t0
        global_crop_ratio_range = getattr(self.cfg, "global_crop_ratio_range", (0.8, 1.0))
        # Local (stronger) views of t0 (cropped smaller, then resized back to window_size)
        local_crop_ratio_range = getattr(self.cfg, "local_crop_ratio_range", (0.2, 0.5))

        self.global_aug_transform = TimeSeriesTransform(
            output_length=window_size,
            scale_range=getattr(self.cfg, "scale_range", (1.0, 1.0)),
            crop_ratio_range=global_crop_ratio_range,
            jitter_std=getattr(self.cfg, "jitter_std", 0.1),
            p_noise=getattr(self.cfg, "p_noise", 0.3),
            p_freq_mask=getattr(self.cfg, "p_freq_mask", 0.0),
            max_freq_ratio=getattr(self.cfg, "max_freq_ratio", 0.0),
            p_temporal_mask=getattr(self.cfg, "p_temporal_mask", 0.9),
            p_transform=getattr(self.cfg, "p_transform_global", getattr(self.cfg, "p_transform", 0.7)),
        )
        self.local_aug_transform = TimeSeriesTransform(
            output_length=window_size,
            scale_range=getattr(self.cfg, "scale_range", (1.0, 1.0)),
            crop_ratio_range=local_crop_ratio_range,
            jitter_std=getattr(self.cfg, "jitter_std", 0.1),
            p_noise=getattr(self.cfg, "p_noise", 0.3),
            p_freq_mask=getattr(self.cfg, "p_freq_mask", 0.0),
            max_freq_ratio=getattr(self.cfg, "max_freq_ratio", 0.0),
            p_temporal_mask=getattr(self.cfg, "p_temporal_mask", 0.9),
            p_transform=getattr(self.cfg, "p_transform_local", getattr(self.cfg, "p_transform", 0.7)),
        )

        # Validation/test: deterministic, no augmentation (all views become identical resizes).
        self.eval_transform = TimeSeriesTransform(
            output_length=window_size,
            scale_range=(1.0, 1.0),
            crop_ratio_range=(1.0, 1.0),
            jitter_std=0.0,
            p_noise=0.0,
            p_freq_mask=0.0,
            max_freq_ratio=0.0,
            p_temporal_mask=0.0,
            p_transform=0.0,
        )

        self.scaler = TorchStandardScaler()

    def prepare_data(self):
        PeMS08(root="./data/pems08", mask_zeros=True)

    def setup(self, stage=None):
        pems08 = PeMS08(root="./data/pems08", mask_zeros=True)
        data = torch.tensor(pems08.target.values, dtype=torch.float32)  # [T, C]
        time_features = encode_timestamps_torch(pems08.timestamps)  # [T, 6]

        split_mode = getattr(self.cfg, "split_mode", None)
        if split_mode is None:
            split_mode = "random_windows" if bool(getattr(self.cfg, "random_split", False)) else "temporal"

        split_seed = int(getattr(self.cfg, "split_seed", 0))
        split_frac = float(getattr(self.cfg, "train_split", 0.8))

        window_size = int(self.cfg.window_size)
        target_window_size = int(self.cfg.target_window_size)
        temporal_shift = int(getattr(self.cfg, "temporal_shift", 10))
        stride = int(getattr(self.cfg, "stride", 1))
        num_local_views_train = int(getattr(self.cfg, "num_local_views", 6))
        num_local_views_val = int(getattr(self.cfg, "num_local_views_val", num_local_views_train))

        total_input_span = temporal_shift + window_size
        total_span = max(total_input_span, window_size + target_window_size)
        min_required_len = total_span + temporal_shift + 1

        if split_mode not in {"temporal", "random_windows", "ts_cv"}:
            raise ValueError(f"Unknown cfg.split_mode={split_mode!r}. Use 'temporal', 'random_windows', or 'ts_cv'.")

        if split_mode in {"temporal", "random_windows"} and not (0.0 < split_frac < 1.0):
            raise ValueError(f"cfg.train_split must be in (0, 1), got {split_frac}")

        if split_mode == "random_windows":
            full_scaled_3d = self.scaler.fit_transform(data)
            full_scaled_ct = full_scaled_3d.squeeze(0)  # [C, T]

            base_train = PeMS08MultiScaleViewsDataset(
                full_scaled_ct,
                time_features,
                global_transform=self.global_aug_transform,
                local_transform=self.local_aug_transform,
                window_size=window_size,
                target_window_size=target_window_size,
                temporal_shift=temporal_shift,
                stride=stride,
                num_local_views=num_local_views_train,
            )
            base_val = PeMS08MultiScaleViewsDataset(
                full_scaled_ct,
                time_features,
                global_transform=self.eval_transform,
                local_transform=self.eval_transform,
                window_size=window_size,
                target_window_size=target_window_size,
                temporal_shift=temporal_shift,
                stride=stride,
                num_local_views=num_local_views_val,
            )

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
            if T < min_required_len * 2:
                raise ValueError(
                    f"Sequence too short for time-series split with current windowing. "
                    f"Need at least ~{min_required_len * 2} time steps, got {T}."
                )

            test_size = int(getattr(self.cfg, "cv_test_size", 0))
            if test_size <= 0:
                test_size = T // (n_splits + 1)
            if test_size < min_required_len:
                raise ValueError(
                    f"cv_test_size too small for windowing. Need >= {min_required_len}, got {test_size}."
                )

            train_end = (fold + 1) * test_size
            val_start = train_end + gap
            val_end = min(val_start + test_size, T)

            if train_end < min_required_len:
                raise ValueError(
                    f"Training segment too short for fold {fold}. train_end={train_end}, need >= {min_required_len}."
                )
            if (val_end - val_start) < min_required_len:
                raise ValueError(
                    f"Validation segment too short for fold {fold}. "
                    f"val_len={val_end - val_start}, need >= {min_required_len}. "
                    "Try smaller cv_gap or larger cv_test_size."
                )

            train_raw = data[:train_end]
            val_raw = data[val_start:val_end]
            train_time = time_features[:train_end]
            val_time = time_features[val_start:val_end]

            train_scaled_3d = self.scaler.fit_transform(train_raw)
            val_scaled_3d = self.scaler.transform(val_raw)
            train_scaled_ct = train_scaled_3d.squeeze(0)
            val_scaled_ct = val_scaled_3d.squeeze(0)

            self.train_ds = PeMS08MultiScaleViewsDataset(
                train_scaled_ct,
                train_time,
                global_transform=self.global_aug_transform,
                local_transform=self.local_aug_transform,
                window_size=window_size,
                target_window_size=target_window_size,
                temporal_shift=temporal_shift,
                stride=stride,
                num_local_views=num_local_views_train,
            )
            self.val_ds = PeMS08MultiScaleViewsDataset(
                val_scaled_ct,
                val_time,
                global_transform=self.eval_transform,
                local_transform=self.eval_transform,
                window_size=window_size,
                target_window_size=target_window_size,
                temporal_shift=temporal_shift,
                stride=stride,
                num_local_views=num_local_views_val,
            )
            return

        # Temporal split (default)
        n_train = int(len(data) * split_frac)
        train_raw = data[:n_train]
        test_raw = data[n_train:]
        train_time = time_features[:n_train]
        test_time = time_features[n_train:]

        train_scaled_3d = self.scaler.fit_transform(train_raw)
        test_scaled_3d = self.scaler.transform(test_raw)
        train_scaled_ct = train_scaled_3d.squeeze(0)
        test_scaled_ct = test_scaled_3d.squeeze(0)

        self.train_ds = PeMS08MultiScaleViewsDataset(
            train_scaled_ct,
            train_time,
            global_transform=self.global_aug_transform,
            local_transform=self.local_aug_transform,
            window_size=window_size,
            target_window_size=target_window_size,
            temporal_shift=temporal_shift,
            stride=stride,
            num_local_views=num_local_views_train,
        )
        self.val_ds = PeMS08MultiScaleViewsDataset(
            test_scaled_ct,
            test_time,
            global_transform=self.eval_transform,
            local_transform=self.eval_transform,
            window_size=window_size,
            target_window_size=target_window_size,
            temporal_shift=temporal_shift,
            stride=stride,
            num_local_views=num_local_views_val,
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


class PeMS08SSLDataModule(L.LightningDataModule):
    """PeMS08 SSL DataModule using past-only windows and view builders.

    This module constructs t0 views plus optional t-1 view (no t+1).
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        self.scaler = TorchStandardScaler()

        self.transform = TimeSeriesTransform(
            output_length=self.cfg.window_size,
            scale_range=getattr(self.cfg, "scale_range", (1.0, 1.0)),
            jitter_std=getattr(self.cfg, "jitter_std", 0.1),
            p_noise=getattr(self.cfg, "p_noise", 0.3),
            p_freq_mask=getattr(self.cfg, "p_freq_mask", 0.0),
            max_freq_ratio=getattr(self.cfg, "max_freq_ratio", 0.0),
            p_temporal_mask=getattr(self.cfg, "p_temporal_mask", 0.8),
            p_transform=getattr(self.cfg, "p_transform", 0.7),
        )

    def prepare_data(self):
        PeMS08(root="./data/pems08", mask_zeros=True)

    def _make_view_dataset(self, data: torch.Tensor, time_features: torch.Tensor, *, repeat_factor: int) -> ViewDataset:
        include_prev = bool(getattr(self.cfg, "include_prev", True))
        prev_shift = int(getattr(self.cfg, "prev_shift", getattr(self.cfg, "temporal_shift", 10)))
        horizon = int(getattr(self.cfg, "target_window_size", 0))

        window_ds = WindowDataset(
            data,
            time_features,
            window_size=int(self.cfg.window_size),
            stride=int(getattr(self.cfg, "stride", 1)),
            include_prev=include_prev,
            prev_shift=prev_shift,
            horizon=horizon,
        )

        view_builder = AugmentationViewBuilder(
            transform=self.transform,
            repeat_factor=int(repeat_factor),
            include_prev=include_prev,
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
            shuffle=True,# needs to be for sigreg
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

