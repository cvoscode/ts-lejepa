import os
import urllib.request
import zipfile
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from torch_geometric.utils import dense_to_sparse
import random
import lightning as L
from typing import Optional, List
from omegaconf import DictConfig
from ..augementation.v1 import TimeSeriesTransform
from ..preprocessing.scalers import TorchStandardScaler
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
            return np.exp(-np.square(self.dist / sigma))
        return (~np.isinf(self.dist)).astype('float32')

    def get_graph(self, device=None):
        if device is None:
            return self.edge_index, self.edge_weight
        return self.edge_index.to(device), self.edge_weight.to(device)

class PeMS08AugmentedDataset(Dataset):
    def __init__(self, pems08: torch.Tensor, transform, window_size: int = 512, target_window_size: int = 12, temporal_shift: int = 10, stride: int = 1, sensors=None):
        super().__init__()
        self.data = pems08
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

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        base_start = self.temporal_shift + (idx * self.stride)
        start_prev = base_start - self.temporal_shift
        t_prev = self.data[self.sensors, start_prev : start_prev + self.window_size]
        t_curr = self.data[self.sensors, base_start : base_start + self.window_size]
        start_next = base_start + self.temporal_shift
        t_next = self.data[self.sensors, start_next : start_next + self.window_size]
        target_start = base_start + self.window_size
        target = self.data[self.sensors, target_start : target_start + self.target_window_size]
        view_prev = self.transform(t_prev.clone())
        view_curr = self.transform(t_curr.clone())
        view_next = self.transform(t_next.clone())
        views = torch.stack([view_prev, view_curr, view_next])
        # Transpose target to [horizon, channels] -> [12, 170]
        return views, target.t()

class PeMS08DataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.aug_transform = TimeSeriesTransform(
            output_length=self.cfg.window_size, 
            scale_range=getattr(self.cfg, "scale_range", (1.0, 1.0)), 
            jitter_std=getattr(self.cfg, "jitter_std", 0.0), 
            p_noise=getattr(self.cfg, "p_noise", 0.1), 
            p_freq_mask=getattr(self.cfg, "p_freq_mask", 0.0),
            max_freq_ratio=getattr(self.cfg, "max_freq_ratio", 0.0),
            p_temporal_mask=getattr(self.cfg, "p_temporal_mask", 0.9),
            p_transform=getattr(self.cfg, "p_transform", 0.6)
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
        data = torch.tensor(pems08.target.values, dtype=torch.float32)
        n_train = int(len(data) * 0.8)
        train_raw = data[:n_train]
        test_raw = data[n_train:]
        train_scaled_3d = self.scaler.fit_transform(train_raw)
        test_scaled_3d = self.scaler.transform(test_raw)
        train_scaled_ct = train_scaled_3d.squeeze(0)
        test_scaled_ct = test_scaled_3d.squeeze(0)
        self.train_ds = PeMS08AugmentedDataset(
            train_scaled_ct, transform=self.aug_transform, window_size=self.cfg.window_size,
            target_window_size=self.cfg.target_window_size, temporal_shift=getattr(self.cfg, "temporal_shift", 10),
            stride=getattr(self.cfg, "stride", 1)
        )
        self.val_ds = PeMS08AugmentedDataset(
            test_scaled_ct, transform=self.test_transform, window_size=self.cfg.window_size,
            target_window_size=self.cfg.target_window_size, temporal_shift=getattr(self.cfg, "temporal_shift", 10),
            stride=getattr(self.cfg, "stride", 1)
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.cfg.batch_size, shuffle=True, num_workers=getattr(self.cfg, "num_workers", 0))

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.cfg.batch_size, shuffle=False, num_workers=getattr(self.cfg, "num_workers", 0))
