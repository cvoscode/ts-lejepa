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
    It uses the CLEAN t0 view (unaugmented) for proper forecasting supervision.
    
    View structure expected: [t-N, ..., t-1, t0_clean, t0_aug1, ..., t0_augR]
    The clean_t0_index should point to the unaugmented t0 view.
    """

    def __init__(
        self,
        *,
        ssl_core: LeJEPA_SSL,
        lr: float = 1e-3,
        weight_decay: float = 5e-2,
        probe: Optional[ForecastProbe] = None,
        probe_loss_weight: float = 0.0,
        probe_lr: Optional[float] = None,
        probe_weight_decay: Optional[float] = None,
        probe_start_epoch: int = 0,
        probe_start_step: int = 0,
        clean_t0_index: Optional[int] = None,  # Index of clean t0 view for probe
        # LR schedule params
        scheduler: str = "cosine",  # 'cosine' or 'none'
        warmup_epochs: int = 5,
        min_lr: float = 1e-6,
        # Gradient clipping. SIGReg can spike early in training, so we keep a
        # tighter norm-clip than the Lightning default of 0. 0.5 is the safe
        # default; bump to 1.0 if you see loss-of-signal at the cost of more
        # occasional spikes.
        gradient_clip_val: float = 0.5,
        gradient_clip_algorithm: str = "norm",
    ) -> None:
        super().__init__()
        self.ssl_core = ssl_core
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.probe = probe
        self.probe_loss_weight = float(probe_loss_weight)
        self.probe_lr = float(probe_lr) if probe_lr is not None else None
        self.probe_weight_decay = float(probe_weight_decay) if probe_weight_decay is not None else None
        self.probe_start_epoch = int(probe_start_epoch)
        self.probe_start_step = int(probe_start_step)
        # Default: clean t0 is at index num_prev_views (first t0 after prev views)
        self._clean_t0_index = clean_t0_index if clean_t0_index is not None else ssl_core.num_prev_views
        # LR schedule
        self.scheduler_type = str(scheduler)
        self.warmup_epochs = int(warmup_epochs)
        self.min_lr = float(min_lr)
        # Gradient clipping. Persisted as hparams so Lightning restores them
        # on checkpoint resume.
        self.gradient_clip_val = float(gradient_clip_val)
        self.gradient_clip_algorithm = str(gradient_clip_algorithm)
        # If the Trainer hasn't been told to clip, we still want the safer
        # default to be active. We expose a hook below.
        self._clip_logged = False

        torch.set_float32_matmul_precision('medium')

    def _probe_is_active(self) -> bool:
        if self.probe is None:
            return False
        if self.current_epoch < self.probe_start_epoch:
            return False
        if self.global_step < self.probe_start_step:
            return False
        return True

    def on_train_start(self) -> None:
        """Verify that the Trainer's gradient-clip setting matches the module's.

        SIGReg can spike early in training; we want a tight norm-clip
        (``gradient_clip_val=0.5`` by default). If the user instantiates the
        ``Trainer`` without specifying a clip, fall back to the module's value
        so the safer default always wins.
        """
        if self._clip_logged:
            return
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return
        # Lightning stores the configured clip on the trainer. ``0`` means "no
        # clipping"; in that case we leave a warning and let the module-level
        # gradient_clip_val guide our own manual clip below.
        configured = getattr(trainer, "gradient_clip_val", 0)
        if configured in (None, 0):
            import warnings
            warnings.warn(
                f"Trainer has no gradient_clip_val; SSLPretrainModule will "
                f"apply its own clip={self.gradient_clip_val} "
                f"({self.gradient_clip_algorithm}). Pass "
                f"gradient_clip_val={self.gradient_clip_val} to the Trainer "
                "explicitly to silence this warning.",
                UserWarning,
                stacklevel=2,
            )
        else:
            self.log("config/gradient_clip_val", float(configured), prog_bar=False)
        self._clip_logged = True

    def on_after_backward(self) -> None:
        """Fallback gradient clipping when the Trainer didn't enable it.

        This is a no-op when the Trainer already does the clip — Lightning's
        clip runs after ``on_after_backward`` and overrides whatever we did.
        When the Trainer has ``gradient_clip_val=0`` we still want our safer
        default to be active.
        """
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return
        configured = getattr(trainer, "gradient_clip_val", 0)
        if configured not in (None, 0):
            return
        if not torch.is_grad_enabled():
            return
        clip_val = float(self.gradient_clip_val)
        if clip_val <= 0:
            return
        if self.gradient_clip_algorithm == "norm":
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.parameters() if p.requires_grad],
                clip_val,
            )
        else:  # "value"
            torch.nn.utils.clip_grad_value_(
                [p for p in self.parameters() if p.requires_grad],
                clip_val,
            )

    def training_step(self, batch: Batch, batch_idx: int):
        """Compute SSL loss and log probe metrics if targets are present."""
        batch_size = batch.views.shape[0]
        ssl_res = self.ssl_core(batch.views, batch.view_times, global_step=self.global_step)
        self.log("train/ssl_loss", ssl_res.total_loss, prog_bar=True, batch_size=batch_size)
        self.log("train/inv_loss", ssl_res.inv_loss, batch_size=batch_size)
        self.log("train/sigreg", ssl_res.sigreg_loss, batch_size=batch_size)
        
        # Log temporal alignment if available (measures how well prev views align with t0)
        if ssl_res.temporal_alignment is not None:
            self.log("train/temporal_alignment", ssl_res.temporal_alignment, batch_size=batch_size)
        
        # Log embedding health diagnostics
        if ssl_res.embedding_std is not None:
            self.log("train/embedding_std", ssl_res.embedding_std, batch_size=batch_size)
        if ssl_res.feature_collapse_ratio is not None:
            self.log("train/feature_collapse_ratio", ssl_res.feature_collapse_ratio, batch_size=batch_size)

        if self._probe_is_active() and batch.targets is not None:
            probe_loss = self._probe_loss(batch)
            self.log("train/probe_loss", probe_loss, prog_bar=False, batch_size=batch_size)
            self._log_probe_mae_unscaled(batch, batch_size=batch_size, stage="train")
            return ssl_res.total_loss + self.probe_loss_weight * probe_loss

        return ssl_res.total_loss

    def validation_step(self, batch: Batch, batch_idx: int):
        """Compute SSL loss and optional probe metrics for validation."""
        batch_size = batch.views.shape[0]
        ssl_res = self.ssl_core(batch.views, batch.view_times, global_step=self.global_step)
        self.log("val/ssl_loss", ssl_res.total_loss, prog_bar=True, batch_size=batch_size)
        self.log("val/inv_loss", ssl_res.inv_loss, batch_size=batch_size)
        self.log("val/sigreg", ssl_res.sigreg_loss, batch_size=batch_size)
        
        # Log temporal alignment if available
        if ssl_res.temporal_alignment is not None:
            self.log("val/temporal_alignment", ssl_res.temporal_alignment, batch_size=batch_size)
        
        # Log embedding health diagnostics
        if ssl_res.embedding_std is not None:
            self.log("val/embedding_std", ssl_res.embedding_std, batch_size=batch_size)
        if ssl_res.feature_collapse_ratio is not None:
            self.log("val/feature_collapse_ratio", ssl_res.feature_collapse_ratio, batch_size=batch_size)

        if self._probe_is_active() and batch.targets is not None:
            probe_loss = self._probe_loss(batch)
            self.log("val/probe_loss", probe_loss, prog_bar=False, batch_size=batch_size)
            self._log_probe_mae_unscaled(batch, batch_size=batch_size, stage="val")

        return ssl_res.total_loss

    def _log_probe_mae_unscaled(self, batch: Batch, *, batch_size: int, stage: str) -> None:
        """Log unscaled MAE for probe forecasts when scaler is available.
        
        Uses the CLEAN t0 view (at _clean_t0_index) for proper evaluation.
        """
        scaler = getattr(self, "scaler", None)
        if scaler is None and self.trainer is not None:
             scaler = getattr(self.trainer.datamodule, "scaler", None)
        
        if scaler is None or scaler.mean is None or scaler.std is None:
            return

        views = batch.views
        # Use clean t0 view (unaugmented) for probe evaluation
        clean_t0_view = views[:, self._clean_t0_index, ...]
        clean_t0_times = None
        if batch.view_times is not None:
            clean_t0_times = batch.view_times[:, self._clean_t0_index, ...]
        
        with torch.no_grad():
            emb = self._encode_probe_features(clean_t0_view, clean_t0_times)
        
        if not self._probe_is_active():
            return

        preds = self.probe(emb, batch.future_times)
        targets = batch.targets
        
        # Ensure correct shape: targets should be [B, C, H] to match preds
        if targets.shape[1] == preds.shape[2] and targets.shape[2] == preds.shape[1]:
            targets = targets.transpose(1, 2)

        scaler = scaler.to(preds.device)
        preds_unscaled = scaler.inverse_transform(preds)
        targets_unscaled = scaler.inverse_transform(targets)

        mae_unscaled = F.l1_loss(preds_unscaled, targets_unscaled)
        self.log(f"{stage}/probe_mae_unscaled", mae_unscaled, prog_bar=True, batch_size=batch_size)

    def _probe_loss(self, batch: Batch) -> torch.Tensor:
        """Compute probe loss using the CLEAN t0 view (not augmented).
        
        This ensures the probe learns to forecast from ground truth embeddings,
        not from augmented/corrupted views.
        """
        views = batch.views
        # Use clean t0 view (unaugmented) at the known index
        clean_t0_view = views[:, self._clean_t0_index, ...]
        clean_t0_times = None
        if batch.view_times is not None:
            clean_t0_times = batch.view_times[:, self._clean_t0_index, ...]
        
        with torch.no_grad():
            emb = self._encode_probe_features(clean_t0_view, clean_t0_times)

        preds = self.probe(emb, batch.future_times)
        targets = batch.targets
        
        # Ensure correct shape: targets should be [B, C, H] to match preds
        if targets.shape[1] == preds.shape[2] and targets.shape[2] == preds.shape[1]:
            targets = targets.transpose(1, 2)
        
        return F.mse_loss(preds, targets)

    def _encode_probe_features(
        self,
        clean_t0_view: torch.Tensor,
        clean_t0_times: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        encoder = self.ssl_core.encoder
        if hasattr(encoder, "forward_multilevel"):
            try:
                emb = encoder.forward_multilevel(clean_t0_view, clean_t0_times)
            except TypeError:
                emb = encoder.forward_multilevel(clean_t0_view)
        else:
            if clean_t0_times is not None:
                try:
                    emb = encoder(clean_t0_view, clean_t0_times)
                except TypeError:
                    emb = encoder(clean_t0_view)
            else:
                emb = encoder(clean_t0_view)

        if isinstance(emb, (list, tuple)):
            processed: list[torch.Tensor] = []
            for level in emb:
                if level.dim() == 3:
                    level = level.mean(dim=1)
                processed.append(level)
            return processed

        if emb.dim() == 3:
            emb = emb.mean(dim=1)
        return emb

    def configure_optimizers(self):
        """AdamW optimizer with optional cosine LR schedule + linear warmup."""
        params = [{"params": self.ssl_core.parameters(), "lr": self.lr, "weight_decay": self.weight_decay}]
        if self.probe is not None and self.probe_loss_weight > 0:
            params.append(
                {
                    "params": self.probe.parameters(),
                    "lr": self.probe_lr if self.probe_lr is not None else self.lr,
                    "weight_decay": (
                        self.probe_weight_decay if self.probe_weight_decay is not None else self.weight_decay
                    ),
                }
            )
        optimizer = torch.optim.AdamW(params)

        if self.scheduler_type == "none":
            return optimizer

        # Cosine annealing with linear warmup
        max_epochs = self.trainer.max_epochs if self.trainer and self.trainer.max_epochs else 100
        warmup_epochs = min(self.warmup_epochs, max_epochs)

        remaining_epochs = max_epochs - warmup_epochs
        if remaining_epochs <= 0:
            if warmup_epochs > 0:
                scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.01, total_iters=warmup_epochs
                )
            else:
                return optimizer
        else:
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=remaining_epochs, eta_min=self.min_lr
            )

            if warmup_epochs > 0:
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.01, total_iters=warmup_epochs
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
                )
            else:
                scheduler = cosine_scheduler

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


__all__ = ["SSLPretrainModule"]
