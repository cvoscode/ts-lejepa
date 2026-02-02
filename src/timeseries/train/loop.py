"""Minimal PyTorch training loop for SSL (Ray Tune friendly)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from ..core.types import Batch
from ..ssl.lejepa import LeJEPA_SSL
from ..tasks.forecast_probe import ForecastProbe


def run_ssl_train_loop(
    *,
    cfg: dict,
    ssl_core: LeJEPA_SSL,
    train_loader,
    val_loader=None,
    probe: Optional[ForecastProbe] = None,
    probe_loader=None,
    device: Optional[torch.device] = None,
) -> dict:
    """Run a minimal training loop using dict config.

    Args:
        cfg: Dict config with keys: epochs, lr, weight_decay, probe_weight.
        ssl_core: LeJEPA SSL core module.
        train_loader: Dataloader yielding Batch.
        val_loader: Optional dataloader yielding Batch.
        probe: Optional forecasting probe for evaluation only.
        probe_loader: Optional dataloader yielding probe batches.
        device: Optional device override.
    """
    epochs = int(cfg.get("epochs", 10))
    lr = float(cfg.get("lr", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 5e-2))
    probe_weight = float(cfg.get("probe_weight", 0.0))

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ssl_core.to(device)
    if probe is not None:
        probe.to(device)

    params = list(ssl_core.parameters())
    if probe is not None:
        params += list(probe.parameters())

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    history: dict[str, list[float]] = {"train_ssl": [], "val_ssl": [], "probe": []}

    for epoch in range(epochs):
        ssl_core.train()
        epoch_loss = 0.0
        for batch in train_loader:
            if isinstance(batch, Batch):
                views = batch.views.to(device)
                view_times = batch.view_times.to(device) if batch.view_times is not None else None
                targets = batch.targets.to(device) if batch.targets is not None else None
                future_times = batch.future_times.to(device) if batch.future_times is not None else None
            else:
                # Handle tuple unpacking for flexible loader outputs
                views = batch[0].to(device)
                view_times = batch[1].to(device) if len(batch) > 1 and batch[1] is not None else None
                targets = batch[2].to(device) if len(batch) > 2 and batch[2] is not None else None
                future_times = batch[3].to(device) if len(batch) > 3 and batch[3] is not None else None

            opt.zero_grad(set_to_none=True)
            ssl_res = ssl_core(views, view_times)
            loss = ssl_res.total_loss

            if probe is not None and targets is not None:
                probe_loss = _probe_loss(ssl_core, probe, views, targets, future_times)
                loss = loss + probe_weight * probe_loss

            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        history["train_ssl"].append(epoch_loss / max(1, len(train_loader)))

        if val_loader is not None:
            ssl_core.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, Batch):
                        views = batch.views.to(device)
                        view_times = batch.view_times.to(device) if batch.view_times is not None else None
                    else:
                        views = batch[0].to(device)
                        view_times = batch[1].to(device) if len(batch) > 1 and batch[1] is not None else None

                    ssl_res = ssl_core(views, view_times)
                    val_loss += ssl_res.total_loss.item()
            history["val_ssl"].append(val_loss / max(1, len(val_loader)))

        if probe is not None and probe_loader is not None:
            ssl_core.eval()
            probe.eval()
            probe_loss = 0.0
            with torch.no_grad():
                for batch in probe_loader:
                    window = batch.window.to(device)
                    target = batch.target.to(device)
                    future_times = batch.future_times.to(device) if batch.future_times is not None else None
                    pred = _probe_forward(ssl_core, probe, window, future_times)
                    # Contract: target is [B, H, C], pred is [B, C, H]. Align to [B, C, H].
                    if target.ndim == 3 and pred.ndim == 3 and target.shape[-1] == pred.shape[-2]:
                        target = target.transpose(1, 2)
                    probe_loss += F.mse_loss(pred, target).item()
            history["probe"].append(probe_loss / max(1, len(probe_loader)))

    return history


def _probe_forward(
    ssl_core: LeJEPA_SSL,
    probe: ForecastProbe,
    window: torch.Tensor,
    future_times: Optional[torch.Tensor],
) -> torch.Tensor:
    """Compute probe predictions using detached encoder embeddings."""
    emb = ssl_core.encoder(window)
    if emb.dim() == 3:
        emb = emb.mean(dim=1)
    return probe(emb.detach(), future_times)


def _probe_loss(
    ssl_core: LeJEPA_SSL,
    probe: ForecastProbe,
    views: torch.Tensor,
    targets: torch.Tensor,
    future_times: Optional[torch.Tensor],
) -> torch.Tensor:
    """Compute probe loss with probe gradients but detached encoder."""
    # Assumption: last view is t0 view suitable for probing
    t0_view = views[:, -1, ...]
    
    with torch.no_grad():
        emb = ssl_core.encoder(t0_view)
        if emb.dim() == 3:
            emb = emb.mean(dim=1)
        emb = emb.detach()

    preds = probe(emb, future_times)

    # Contract: targets is [B, H, C], preds is [B, C, H]. Align to [B, C, H].
    if targets.ndim == 3 and preds.ndim == 3 and targets.shape[-1] == preds.shape[-2]:
        targets = targets.transpose(1, 2)
    return F.mse_loss(preds, targets)


__all__ = ["run_ssl_train_loop"]
