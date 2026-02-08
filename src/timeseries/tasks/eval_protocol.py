"""Standardized downstream evaluation protocol for SSL encoders.

Implements the standard evaluation pipeline:
1. Frozen encoder + linear probe
2. Frozen encoder + MLP probe
3. Fine-tuned encoder + MLP head

Supports forecasting, classification, virtual sensing, and anomaly detection tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class EvalResult:
    """Container for evaluation results."""

    task: str
    protocol: str  # 'linear_probe', 'mlp_probe', 'fine_tune'
    train_loss: float
    val_loss: float
    metrics: dict = field(default_factory=dict)


class DownstreamEvaluator:
    """Standardized evaluation protocol for SSL representations.

    Runs the following evaluation suite:
    1. Train linear probe on frozen encoder → report accuracy/loss
    2. Train MLP probe on frozen encoder → report accuracy/loss
    3. Fine-tune encoder + head → report accuracy/loss

    Example:
        >>> evaluator = DownstreamEvaluator(encoder, device='cuda')
        >>> # For forecasting
        >>> results = evaluator.evaluate_forecast(
        ...     train_loader, val_loader,
        ...     input_dim=256, horizon=12, output_channels=170,
        ... )
        >>> for r in results:
        ...     print(f"{r.protocol}: val_loss={r.val_loss:.4f}")
    """

    def __init__(
        self,
        encoder: nn.Module,
        device: Optional[torch.device] = None,
        epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.encoder = encoder
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay

    @torch.no_grad()
    def _extract_embeddings(
        self, loader: DataLoader
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract frozen embeddings from the encoder.

        Args:
            loader: DataLoader yielding (input, target) tuples.
                    input shape: [B, C, T] or batch dict with 'window' key.

        Returns:
            embeddings: [N, D] pooled embeddings for all samples.
            targets: [N, ...] targets for all samples.
        """
        self.encoder.eval()
        self.encoder.to(self.device)

        all_emb = []
        all_targets = []

        for batch in loader:
            if isinstance(batch, (tuple, list)):
                x, target = batch[0], batch[1]
            elif hasattr(batch, "window"):
                x = batch.window
                target = batch.target if hasattr(batch, "target") else batch.targets
            else:
                raise ValueError(f"Unsupported batch type: {type(batch)}")

            x = x.to(self.device)
            emb = self.encoder(x)

            # Pool if needed
            if isinstance(emb, (list, tuple)):
                pooled = []
                for e in emb:
                    if e.dim() == 3:
                        e = e.mean(dim=1)
                    pooled.append(e)
                emb = torch.stack(pooled, dim=1).mean(dim=1)
            elif emb.dim() == 3:
                emb = emb.mean(dim=1)

            all_emb.append(emb.cpu())
            all_targets.append(target if isinstance(target, torch.Tensor) else torch.tensor(target))

        return torch.cat(all_emb, dim=0), torch.cat(all_targets, dim=0)

    def _train_probe(
        self,
        probe: nn.Module,
        train_emb: torch.Tensor,
        train_targets: torch.Tensor,
        val_emb: torch.Tensor,
        val_targets: torch.Tensor,
        loss_fn: nn.Module,
    ) -> tuple[float, float, dict]:
        """Train a probe head on frozen embeddings.

        Returns:
            train_loss, val_loss, metrics dict
        """
        probe = probe.to(self.device)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        train_emb = train_emb.to(self.device)
        train_targets = train_targets.to(self.device)
        val_emb = val_emb.to(self.device)
        val_targets = val_targets.to(self.device)

        best_val_loss = float("inf")
        batch_size = min(256, len(train_emb))

        for epoch in range(self.epochs):
            probe.train()
            perm = torch.randperm(len(train_emb), device=self.device)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(train_emb), batch_size):
                idx = perm[i : i + batch_size]
                emb_batch = train_emb[idx]
                target_batch = train_targets[idx]

                optimizer.zero_grad(set_to_none=True)
                preds = probe(emb_batch)
                loss = loss_fn(preds, target_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # Validation
            probe.eval()
            with torch.no_grad():
                val_preds = probe(val_emb)
                val_loss = loss_fn(val_preds, val_targets).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss

        final_train_loss = epoch_loss / max(1, n_batches)

        # Compute extra metrics
        metrics = {}
        probe.eval()
        with torch.no_grad():
            val_preds = probe(val_emb)
            # MAE for regression tasks
            if val_preds.shape == val_targets.shape:
                metrics["mae"] = F.l1_loss(val_preds, val_targets).item()
            # Accuracy for classification
            if val_preds.dim() == 2 and val_targets.dim() == 1:
                pred_classes = val_preds.argmax(dim=1)
                metrics["accuracy"] = (pred_classes == val_targets).float().mean().item()

        return final_train_loss, best_val_loss, metrics

    def _train_finetune(
        self,
        head: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
    ) -> tuple[float, float, dict]:
        """Fine-tune encoder + head end-to-end.

        Returns:
            train_loss, val_loss, metrics dict
        """
        self.encoder.to(self.device)
        head = head.to(self.device)
        self.encoder.train()
        head.train()

        all_params = list(self.encoder.parameters()) + list(head.parameters())
        optimizer = torch.optim.AdamW(all_params, lr=self.lr * 0.1, weight_decay=self.weight_decay)

        best_val_loss = float("inf")
        final_train_loss = 0.0

        for epoch in range(self.epochs):
            self.encoder.train()
            head.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                if isinstance(batch, (tuple, list)):
                    x, target = batch[0], batch[1]
                elif hasattr(batch, "window"):
                    x = batch.window
                    target = batch.target if hasattr(batch, "target") else batch.targets
                else:
                    raise ValueError(f"Unsupported batch type: {type(batch)}")

                x = x.to(self.device)
                target = target.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                emb = self.encoder(x)
                if isinstance(emb, (list, tuple)):
                    pooled = []
                    for e in emb:
                        if e.dim() == 3:
                            e = e.mean(dim=1)
                        pooled.append(e)
                    emb = torch.stack(pooled, dim=1).mean(dim=1)
                elif emb.dim() == 3:
                    emb = emb.mean(dim=1)

                preds = head(emb)
                loss = loss_fn(preds, target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            final_train_loss = epoch_loss / max(1, n_batches)

            # Validation
            self.encoder.eval()
            head.eval()
            val_loss_total = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, (tuple, list)):
                        x, target = batch[0], batch[1]
                    elif hasattr(batch, "window"):
                        x = batch.window
                        target = batch.target if hasattr(batch, "target") else batch.targets
                    else:
                        continue

                    x = x.to(self.device)
                    target = target.to(self.device)
                    emb = self.encoder(x)
                    if isinstance(emb, (list, tuple)):
                        pooled = []
                        for e in emb:
                            if e.dim() == 3:
                                e = e.mean(dim=1)
                            pooled.append(e)
                        emb = torch.stack(pooled, dim=1).mean(dim=1)
                    elif emb.dim() == 3:
                        emb = emb.mean(dim=1)

                    preds = head(emb)
                    val_loss_total += loss_fn(preds, target).item()
                    val_batches += 1

            val_loss = val_loss_total / max(1, val_batches)
            if val_loss < best_val_loss:
                best_val_loss = val_loss

        return final_train_loss, best_val_loss, {}

    def evaluate_forecast(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        input_dim: int,
        horizon: int,
        output_channels: int,
        protocols: Optional[list[str]] = None,
    ) -> list[EvalResult]:
        """Run evaluation protocol for forecasting task.

        Args:
            train_loader: DataLoader yielding (window, target) pairs, window [B, C, T].
            val_loader: Validation DataLoader.
            input_dim: Encoder output dimension.
            horizon: Forecast horizon.
            output_channels: Number of output channels.
            protocols: Which protocols to run. Default: all three.

        Returns:
            List of EvalResult, one per protocol.
        """
        if protocols is None:
            protocols = ["linear_probe", "mlp_probe", "fine_tune"]

        results = []
        output_size = output_channels * horizon
        loss_fn = nn.MSELoss()

        for protocol in protocols:
            if protocol == "linear_probe":
                train_emb, train_targets = self._extract_embeddings(train_loader)
                val_emb, val_targets = self._extract_embeddings(val_loader)
                train_targets = train_targets.reshape(-1, output_size)
                val_targets = val_targets.reshape(-1, output_size)

                probe = nn.Linear(train_emb.shape[-1], output_size)
                train_loss, val_loss, metrics = self._train_probe(
                    probe, train_emb, train_targets, val_emb, val_targets, loss_fn
                )
                results.append(EvalResult("forecast", protocol, train_loss, val_loss, metrics))

            elif protocol == "mlp_probe":
                train_emb, train_targets = self._extract_embeddings(train_loader)
                val_emb, val_targets = self._extract_embeddings(val_loader)
                train_targets = train_targets.reshape(-1, output_size)
                val_targets = val_targets.reshape(-1, output_size)

                probe = nn.Sequential(
                    nn.LayerNorm(train_emb.shape[-1]),
                    nn.Linear(train_emb.shape[-1], 256),
                    nn.GELU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, output_size),
                )
                train_loss, val_loss, metrics = self._train_probe(
                    probe, train_emb, train_targets, val_emb, val_targets, loss_fn
                )
                results.append(EvalResult("forecast", protocol, train_loss, val_loss, metrics))

            elif protocol == "fine_tune":
                head = nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, 256),
                    nn.GELU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, output_size),
                )
                train_loss, val_loss, metrics = self._train_finetune(
                    head, train_loader, val_loader, loss_fn
                )
                results.append(EvalResult("forecast", protocol, train_loss, val_loss, metrics))

        return results

    def evaluate_classification(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        input_dim: int,
        num_classes: int,
        protocols: Optional[list[str]] = None,
    ) -> list[EvalResult]:
        """Run evaluation protocol for classification task.

        Args:
            train_loader: DataLoader yielding (window, label) pairs.
            val_loader: Validation DataLoader.
            input_dim: Encoder output dimension.
            num_classes: Number of output classes.
            protocols: Which protocols to run.

        Returns:
            List of EvalResult, one per protocol.
        """
        if protocols is None:
            protocols = ["linear_probe", "mlp_probe", "fine_tune"]

        results = []
        loss_fn = nn.CrossEntropyLoss()

        for protocol in protocols:
            if protocol == "linear_probe":
                train_emb, train_targets = self._extract_embeddings(train_loader)
                val_emb, val_targets = self._extract_embeddings(val_loader)

                probe = nn.Linear(train_emb.shape[-1], num_classes)
                train_loss, val_loss, metrics = self._train_probe(
                    probe, train_emb, train_targets.long(), val_emb, val_targets.long(), loss_fn
                )
                results.append(EvalResult("classification", protocol, train_loss, val_loss, metrics))

            elif protocol == "mlp_probe":
                train_emb, train_targets = self._extract_embeddings(train_loader)
                val_emb, val_targets = self._extract_embeddings(val_loader)

                probe = nn.Sequential(
                    nn.LayerNorm(train_emb.shape[-1]),
                    nn.Linear(train_emb.shape[-1], 128),
                    nn.GELU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, num_classes),
                )
                train_loss, val_loss, metrics = self._train_probe(
                    probe, train_emb, train_targets.long(), val_emb, val_targets.long(), loss_fn
                )
                results.append(EvalResult("classification", protocol, train_loss, val_loss, metrics))

            elif protocol == "fine_tune":
                head = nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, 128),
                    nn.GELU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, num_classes),
                )
                train_loss, val_loss, metrics = self._train_finetune(
                    head, train_loader, val_loader, loss_fn
                )
                results.append(EvalResult("classification", protocol, train_loss, val_loss, metrics))

        return results

    def evaluate_anomaly(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        input_dim: int,
        protocols: Optional[list[str]] = None,
    ) -> list[EvalResult]:
        """Run evaluation protocol for anomaly detection task.

        Args:
            train_loader: DataLoader yielding (window, anomaly_label) pairs.
                          anomaly_label: 0=normal, 1=anomaly.
            val_loader: Validation DataLoader.
            input_dim: Encoder output dimension.
            protocols: Which protocols to run.

        Returns:
            List of EvalResult, one per protocol.
        """
        if protocols is None:
            protocols = ["linear_probe", "mlp_probe"]

        results = []
        loss_fn = nn.BCEWithLogitsLoss()

        for protocol in protocols:
            if protocol in ("linear_probe", "mlp_probe"):
                train_emb, train_targets = self._extract_embeddings(train_loader)
                val_emb, val_targets = self._extract_embeddings(val_loader)
                train_targets = train_targets.float().unsqueeze(-1)
                val_targets = val_targets.float().unsqueeze(-1)

                if protocol == "linear_probe":
                    probe = nn.Linear(train_emb.shape[-1], 1)
                else:
                    probe = nn.Sequential(
                        nn.LayerNorm(train_emb.shape[-1]),
                        nn.Linear(train_emb.shape[-1], 64),
                        nn.GELU(),
                        nn.Dropout(0.3),
                        nn.Linear(64, 1),
                    )

                train_loss, val_loss, metrics = self._train_probe(
                    probe, train_emb, train_targets, val_emb, val_targets, loss_fn
                )
                results.append(EvalResult("anomaly", protocol, train_loss, val_loss, metrics))

        return results


__all__ = ["DownstreamEvaluator", "EvalResult"]
