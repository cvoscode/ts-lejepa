from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
import lightning as L

from ..data.window_dataset import WindowDataset
from ..data.view_dataset import ViewDataset, collate_batches
from ..data.view_builders import AugmentationViewBuilder
from ..transforms.ops.basic import Scaling, Drift, FeatureJitter
from ..transforms.compose import Compose
from .pems import Pems08Dataset # We might still use the legacy loader logic or generic one.

class GenericTimeSeriesDataModule(L.LightningDataModule):
    """
    Generic DataModule for time series SSL.
    Loads data from .npz or .csv.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.data_path = cfg.get("data_path", "data/pems08/pems08.npz")
        self.batch_size = cfg.get("batch_size", 32)
        self.num_workers = cfg.get("num_workers", 4)
        
        # Scaling params
        self.scaler = None 

    def setup(self, stage: Optional[str] = None):
        # 1. Load Data
        if self.data_path.endswith('.npz'):
            data = np.load(self.data_path)
            # Assume 'data' key exists, shape [T, C, F] or [T, C]
            if 'data' in data:
                raw = data['data']
            else:
                raw = data[list(data.keys())[0]]
                
            # Take only the first feature if 3D
            if raw.ndim == 3:
                raw = raw[..., 0] # [T, N]
        elif self.data_path.endswith('.csv'):
             df = pd.read_csv(self.data_path)
             # Basic handling: assume all columns except optional date are targets
             if 'date' in df.columns:
                 df = df.drop(columns=['date'])
             raw = df.values
        else:
             raise ValueError("Unsupported file format")

        # 2. Split
        L_data = len(raw)
        train_len = int(L_data * self.cfg.get("train_split", 0.7))
        val_len = int(L_data * 0.2)
        test_len = L_data - train_len - val_len
        
        train_data = raw[:train_len]
        val_data = raw[train_len:train_len+val_len]
        test_data = raw[train_len+val_len:]
        
        # 3. Scale (Fit on train)
        # Simple standard scaler
        mean = train_data.mean(axis=0)
        std = train_data.std(axis=0) + 1e-5
        
        # Store for inverse transform
        # Create a dummy scaler object with inverse_transform for compatibility
        self.scaler = SimpleScaler(mean, std)
        
        train_data = (train_data - mean) / std
        val_data = (val_data - mean) / std
        test_data = (test_data - mean) / std
        
        # 4. Create Window Datasets
        # Convert to Tensor [C, T] -> [T, C] (WindowDataset expects [T, C])
        
        window_size = self.cfg.get("window_size", 96)
        
        self.train_ds = WindowDataset(
            torch.tensor(train_data, dtype=torch.float32), 
            window_size=window_size,
            input_size=window_size,
            horizon=self.cfg.get("target_window_size", 12)
        )
        self.val_ds = WindowDataset(
            torch.tensor(val_data, dtype=torch.float32), 
            window_size=window_size,
            input_size=window_size,
            horizon=self.cfg.get("target_window_size", 12)
        )
        self.test_ds = WindowDataset(
            torch.tensor(test_data, dtype=torch.float32), 
            window_size=window_size,
            input_size=window_size,
            horizon=self.cfg.get("target_window_size", 12)
        )
        
        # 5. Wrap with ViewDataset
        # Define default transform
        transform = Compose([
            Scaling(scale_range=(0.9, 1.1)),
            Drift(slope_range=(-0.01, 0.01)),
            FeatureJitter(jitter_std=0.01)
        ])
        
        view_builder = AugmentationViewBuilder(
            transform=transform,
            repeat_factor=self.cfg.get("repeat_factor", 2),
            include_prev=self.cfg.get("include_prev", True)
        )
        
        self.train_view_ds = ViewDataset(self.train_ds, view_builder)
        self.val_view_ds = ViewDataset(self.val_ds, view_builder) # Validation usually doesn't need augmentation but for SSL loss it does.

    def train_dataloader(self):
        return DataLoader(self.train_view_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_batches)
        
    def val_dataloader(self):
        return DataLoader(self.val_view_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_batches)


class SimpleScaler:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        
    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def inverse_transform(self, x):
        return x * self.std + self.mean
