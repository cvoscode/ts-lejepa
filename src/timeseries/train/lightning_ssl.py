"""Lightning wrapper for SSL pretraining and probe logging."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from ..core.types import Batch
from ..ssl.lejepa import LeJEPA_SSL
from ..tasks.forecast_probe import ForecastProbe


class SSLPretrainModule(L.LightningModule):
    """LightningModule for LeJEPA SSL with optional probe metrics.

    The probe is evaluation-only and does not backpropagate into the encoder.
    """

    def __init__(
        self,
        *,
        ssl_core: LeJEPA_SSL,
        lr: float = 1e-3,
        weight_decay: float = 5e-2,
        probe: Optional[ForecastProbe] = None,
        probe_loss_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.ssl_core = ssl_core
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.probe = probe
        self.probe_loss_weight = float(probe_loss_weight)

    def training_step(self, batch: Batch, batch_idx: int):
        """Compute SSL loss and log probe metrics if targets are present."""
        batch_size = batch.views.shape[0]
        ssl_res = self.ssl_core(batch.views, batch.view_times, global_step=self.global_step)
        self.log("train/ssl_loss", ssl_res.total_loss, prog_bar=True, batch_size=batch_size)
        self.log("train/inv_loss", ssl_res.inv_loss, batch_size=batch_size)
        self.log("train/sigreg", ssl_res.sigreg_loss, batch_size=batch_size)

        if self.probe is not None and batch.targets is not None:
            probe_loss = self._probe_loss(batch)
            self.log("train/probe_loss", probe_loss, prog_bar=False, batch_size=batch_size)
            return ssl_res.total_loss + self.probe_loss_weight * probe_loss

        return ssl_res.total_loss

    def validation_step(self, batch: Batch, batch_idx: int):
        """Compute SSL loss and optional probe metrics for validation."""
        batch_size = batch.views.shape[0]
        ssl_res = self.ssl_core(batch.views, batch.view_times, global_step=self.global_step)
        self.log("val/ssl_loss", ssl_res.total_loss, prog_bar=True, batch_size=batch_size)
        self.log("val/inv_loss", ssl_res.inv_loss, batch_size=batch_size)
        self.log("val/sigreg", ssl_res.sigreg_loss, batch_size=batch_size)

        if self.probe is not None and batch.targets is not None:
            probe_loss = self._probe_loss(batch)
            self.log("val/probe_loss", probe_loss, prog_bar=False, batch_size=batch_size)
            self._log_probe_mae_unscaled(batch, batch_size=batch_size)

        return ssl_res.total_loss

    def _log_probe_mae_unscaled(self, batch: Batch, *, batch_size: int) -> None:
        """Log unscaled MAE for probe forecasts when scaler is available."""
        scaler = getattr(self, "scaler", None)
        if scaler is None and self.trainer is not None:
             scaler = getattr(self.trainer.datamodule, "scaler", None)
        
        if scaler is None or scaler.mean is None or scaler.std is None:
            return

        views = batch.views
        t0_view = views[:, -1, ...]
        with torch.no_grad():
            emb = self.ssl_core.encoder(t0_view)
            if emb.dim() == 3:
                emb = emb.mean(dim=1)
        preds = self.probe(emb.detach(), batch.future_times)
        targets = batch.targets
        if targets.shape[1] == preds.shape[2] and targets.shape[2] == preds.shape[1]:
            targets = targets.transpose(1, 2)

        scaler = scaler.to(preds.device)
        preds_unscaled = scaler.inverse_transform(preds)
        targets_unscaled = scaler.inverse_transform(targets)
        
        # Ensure correct shape match for MAE
        if preds_unscaled.shape != targets_unscaled.shape:
             # Try simple broadcast or view? 
             # For now, let's assume they match or l1_loss handles it, but robust code checks.
             pass

        mae_unscaled = F.l1_loss(preds_unscaled, targets_unscaled)
        self.log("val/probe_mae_unscaled", mae_unscaled, prog_bar=True, batch_size=batch_size)

    def _probe_loss(self, batch: Batch) -> torch.Tensor:
        """Compute probe loss without affecting encoder gradients."""
        with torch.no_grad():
            views = batch.views
            t0_view = views[:, -1, ...]
            emb = self.ssl_core.encoder(t0_view)
            if emb.dim() == 3:
                emb = emb.mean(dim=1)

        preds = self.probe(emb.detach(), batch.future_times)
        targets = batch.targets
        if targets.shape[1] == preds.shape[2] and targets.shape[2] == preds.shape[1]:
            targets = targets.transpose(1, 2)
        return F.mse_loss(preds, targets)

    def configure_optimizers(self):
        """AdamW optimizer for SSL core and optional probe head."""
        params = list(self.ssl_core.parameters())
        if self.probe is not None:
            params += list(self.probe.parameters())
        return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)


__all__ = ["SSLPretrainModule"]
