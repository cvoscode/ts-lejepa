from __future__ import annotations

"""Ray Tune integration for LeJEPA SSL experiments."""

from typing import Any, Dict, Optional

import os

import torch
import lightning as L
from omegaconf import DictConfig, OmegaConf
from torchvision.ops import MLP

from ray import tune
from ray.tune import Tuner
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from ..core.types import InvarianceMix
from ..data.pems import PeMS08SSLDataModule
from ..encoders.rnn import LSTMEncoder, GRUEncoder
from ..encoders.cnn import CNNEncoder
from ..encoders.transformer import TransformerEncoder
from ..encoders.mamba import MambaEncoder
from ..ssl.lejepa import LeJEPA_SSL
from ..tasks.forecast_probe import ForecastProbe
from .lightning_ssl import SSLPretrainModule


def build_default_search_space() -> Dict[str, Any]:
    """Return a default Ray Tune search space for SSL experiments."""
    return {
        "seed": tune.choice([0, 1, 2]),
        "epochs": tune.choice([30, 60, 100]),
        "batch_size": tune.choice([64, 128]),
        "num_workers": tune.choice([4, 8, 16]),
        "lr": tune.loguniform(1e-4, 5e-3),
        "weight_decay": tune.loguniform(1e-6, 1e-2),
        "lamb": tune.uniform(0.0, 0.9),
        "proj_dim": tune.choice([16, 32, 64]),
        "proj_hidden": tune.choice([256, 512, 768]),
        "sigreg_slices": tune.choice([256, 512, 1024, 2048]),
        "sigreg_knots": tune.choice([9, 17, 33]),
        "include_prev": tune.choice([True, False]),
        "prev_shift": tune.choice([6, 12, 24]),
        "repeat_factor": tune.choice([2, 4, 8, 12]),
        "repeat_factor_val": tune.choice([2, 4, 8]),
        "split_mode": tune.choice(["temporal", "random_windows"]),
        "train_split": tune.uniform(0.7, 0.9),
        "p_t0": tune.uniform(0.6, 1.0),
        "p_tminus1": tune.uniform(0.0, 0.4),
        "sample_policy": tune.choice(["at_least_one_tminus1", "t0_only_if_missing"]),
        "scale_range": tune.choice([(0.8, 1.2), (0.9, 1.1), (1.0, 1.0)]),
        "crop_ratio_range": tune.choice([(0.8, 1.0), (0.6, 1.0)]),
        "jitter_std": tune.choice([0.0, 0.05, 0.1]),
        "p_noise": tune.uniform(0.0, 0.5),
        "p_freq_mask": tune.uniform(0.0, 0.5),
        "max_freq_ratio": tune.choice([0.0, 0.1, 0.2]),
        "p_temporal_mask": tune.uniform(0.0, 0.9),
        "p_transform": tune.uniform(0.2, 0.9),
        "encoder_name": tune.choice(["lstm", "gru", "cnn", "transformer", "mamba"]),
        "encoder_output_dim": tune.choice([256, 512]),
        "pool_mode": tune.choice(["mean", "none"]),
        "lstm_hidden_channels": tune.choice([32, 64, 128]),
        "lstm_layers": tune.choice([1, 2, 3]),
        "lstm_dropout": tune.uniform(0.0, 0.3),
        "lstm_bidirectional": tune.choice([True, False]),
        "cnn_stem_channels": tune.choice([64, 128, 256]),
        "cnn_kernel_size": tune.choice([3, 5, 7]),
        "cnn_dropout": tune.uniform(0.0, 0.2),
        "transformer_heads": tune.choice([2, 4, 8]),
        "transformer_layers": tune.choice([2, 3, 4]),
        "transformer_ff": tune.choice([256, 512, 1024]),
        "transformer_dropout": tune.uniform(0.0, 0.2),
        "mamba_d_state": tune.choice([8, 16, 32]),
        "mamba_d_conv": tune.choice([4, 8]),
        "mamba_expand": tune.choice([2, 4]),
        "mamba_layers": tune.choice([1, 2, 3]),
        "mamba_dropout": tune.uniform(0.0, 0.2),
        "probe_loss_weight": tune.choice([0.0, 0.05, 0.1]),
        "probe_use": tune.choice([True, False]),
        "gradient_clip_val": tune.choice([0.0, 0.3, 1.0]),
        "precision": tune.choice([32, 16]),
        "accelerator": tune.choice(["auto"]),
        "devices": tune.choice([1]),
    }


def run_ray_tune_ssl(
    *,
    base_cfg: DictConfig | Dict[str, Any],
    search_space: Optional[Dict[str, Any]] = None,
    num_samples: int = 20,
    metric: str = "val/ssl_loss",
    mode: str = "min",
    max_epochs: Optional[int] = None,
    local_dir: str = "ray_results",
    resources_per_trial: Optional[Dict[str, Any]] = None,
) -> Any:
    """Run Ray Tune with ASHA for SSL pretraining.

    Args:
        base_cfg: Base config merged with Tune samples.
        search_space: Ray Tune search space (overrides defaults if provided).
        num_samples: Number of trial samples.
        metric: Metric to optimize.
        mode: "min" or "max".
        max_epochs: Optional override for ASHA max_t.
        local_dir: Output directory for Ray results.
        resources_per_trial: Optional resources per trial, e.g., {"cpu": 4, "gpu": 1}.
    """
    if search_space is None:
        search_space = build_default_search_space()

    base_cfg = OmegaConf.create(base_cfg)
    scheduler = ASHAScheduler(
        metric=metric,
        mode=mode,
        max_t=int(max_epochs) if max_epochs is not None else int(base_cfg.get("epochs", 100)),
        grace_period=1,
        reduction_factor=2,
    )

    trainable = tune.with_parameters(_ssl_trainable, base_cfg=base_cfg)
    if resources_per_trial is not None:
        trainable = tune.with_resources(trainable, resources_per_trial)

    tuner = Tuner(
        trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=num_samples,
            metric=metric,
            mode=mode,
            max_concurrent_trials=None,
        ),
        run_config=tune.RunConfig(
            name="lejepa_ssl_tune",
            local_dir=local_dir,
            verbose=1,
        ),
    )

    return tuner.fit()


def _ssl_trainable(config: Dict[str, Any], *, base_cfg: DictConfig) -> None:
    cfg = OmegaConf.merge(base_cfg, OmegaConf.create(config))

    _seed_everything(int(cfg.get("seed", 0)))

    datamodule = PeMS08SSLDataModule(cfg)
    datamodule.prepare_data()
    datamodule.setup()

    encoder = _build_encoder(cfg)
    projector = MLP(
        in_channels=encoder.output_dim,
        hidden_channels=[int(cfg.get("proj_hidden", 512)), int(cfg.proj_dim)],
    )

    ssl_core = LeJEPA_SSL(
        encoder=encoder,
        projector=projector,
        proj_dim=int(cfg.proj_dim),
        lamb=float(cfg.lamb),
        invariance_mix=InvarianceMix(
            p_t0=float(cfg.get("p_t0", 0.9)),
            p_tminus1=float(cfg.get("p_tminus1", 0.1)),
            sample_policy=str(cfg.get("sample_policy", "at_least_one_tminus1")),
        ),
        has_prev_view=bool(cfg.get("include_prev", True)),
        sigreg_slices=int(cfg.get("sigreg_slices", 1024)),
        sigreg_knots=int(cfg.get("sigreg_knots", 17)),
        sigreg_seed=int(cfg.get("sigreg_seed", 0)),
    )

    probe = None
    if bool(cfg.get("probe_use", True)):
        probe = ForecastProbe(
            input_dim=encoder.output_dim,
            horizon=int(cfg.target_window_size),
            output_channels=int(cfg.get("input_channels", 170)),
            use_covariates=False,
        )

    model = SSLPretrainModule(
        ssl_core=ssl_core,
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
        probe=probe,
        probe_loss_weight=float(cfg.get("probe_loss_weight", 0.0)),
    )

    metrics = {"val/ssl_loss": "val/ssl_loss"}
    tune_callback = TuneReportCallback(metrics, on="validation_end")

    trainer = L.Trainer(
        max_epochs=int(cfg.epochs),
        accelerator=str(cfg.get("accelerator", "auto")),
        devices=int(cfg.get("devices", 1)),
        callbacks=[tune_callback],
        enable_checkpointing=False,
        logger=False,
        log_every_n_steps=10,
        gradient_clip_val=float(cfg.get("gradient_clip_val", 0.0)),
        precision=int(cfg.get("precision", 32)),
    )

    trainer.fit(model, datamodule=datamodule)


def _build_encoder(cfg: DictConfig) -> torch.nn.Module:
    name = str(cfg.get("encoder_name", "lstm")).lower()
    input_channels = int(cfg.get("input_channels", 170))
    output_dim = int(cfg.get("encoder_output_dim", 512))
    pool_mode = str(cfg.get("pool_mode", "mean"))

    if name == "lstm":
        return LSTMEncoder(
            input_channels=input_channels,
            output_dim=output_dim,
            pool_mode=pool_mode,
            hidden_channels=int(cfg.get("lstm_hidden_channels", 64)),
            num_layers=int(cfg.get("lstm_layers", 2)),
            dropout=float(cfg.get("lstm_dropout", 0.2)),
            bidirectional=bool(cfg.get("lstm_bidirectional", True)),
        )

    if name == "gru":
        return GRUEncoder(
            input_channels=input_channels,
            output_dim=output_dim,
            pool_mode=pool_mode,
            hidden_channels=int(cfg.get("lstm_hidden_channels", 64)),
            num_layers=int(cfg.get("lstm_layers", 2)),
            dropout=float(cfg.get("lstm_dropout", 0.2)),
            bidirectional=bool(cfg.get("lstm_bidirectional", True)),
        )

    if name == "cnn":
        return CNNEncoder(
            input_channels=input_channels,
            output_dim=output_dim,
            pool_mode=pool_mode,
            stem_channels=int(cfg.get("cnn_stem_channels", 128)),
            kernel_size=int(cfg.get("cnn_kernel_size", 5)),
            dropout=float(cfg.get("cnn_dropout", 0.1)),
        )

    if name == "transformer":
        return TransformerEncoder(
            input_channels=input_channels,
            output_dim=output_dim,
            pool_mode=pool_mode,
            n_heads=int(cfg.get("transformer_heads", 4)),
            n_layers=int(cfg.get("transformer_layers", 3)),
            dim_feedforward=int(cfg.get("transformer_ff", 512)),
            dropout=float(cfg.get("transformer_dropout", 0.1)),
        )

    if name == "mamba":
        return MambaEncoder(
            input_channels=input_channels,
            output_dim=output_dim,
            pool_mode=pool_mode,
            d_state=int(cfg.get("mamba_d_state", 16)),
            d_conv=int(cfg.get("mamba_d_conv", 4)),
            expand=int(cfg.get("mamba_expand", 2)),
            num_layers=int(cfg.get("mamba_layers", 2)),
            dropout=float(cfg.get("mamba_dropout", 0.1)),
        )

    raise ValueError(f"Unknown encoder_name={name!r}")


def _seed_everything(seed: int) -> None:
    if seed <= 0:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


__all__ = ["build_default_search_space", "run_ray_tune_ssl"]
