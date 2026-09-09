"""Train LeJEPA SSL on PeMS08 with a strong config and TensorBoard logging.

Run:
    python scripts/train_ssl.py                # default: 10 epochs, 170 sensors
    python scripts/train_ssl.py --epochs 20 --batch 64 --channels 64

TensorBoard logs land in: outputs/tb_logs/<run_name>/
Launch TensorBoard with:
    tensorboard --logdir outputs/tb_logs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src/timeseries importable when run from repo root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import lightning as L
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf
from torchvision.ops import MLP

from timeseries.core.types import InvarianceMix
from timeseries.data.generic_datamodule import TimeSeriesSSLDataModule
from timeseries.data.pems import PeMS08
from timeseries.encoders.rnn import LSTMEncoder
from timeseries.preprocessing.time_encoding import encode_timestamps_torch
from timeseries.ssl.lejepa import LeJEPA_SSL
from timeseries.tasks.forecast_probe import ForecastProbe
from timeseries.train.lightning_ssl import SSLPretrainModule


def build_config(args: argparse.Namespace) -> OmegaConf:
    """Build a strong SSL training config tuned for PeMS08."""
    cfg_dict = {
        # Data
        "window_size": args.window,
        "target_window_size": args.horizon,
        "stride": 6,                    # ~30min stride to keep windows diverse
        "batch_size": args.batch,
        "num_workers": args.workers,
        "train_split": 0.8,
        "split_mode": "temporal",
        "split_seed": 0,

        # Augmentations (compound: each op applied independently)
        "augmentation_mode": "compound",
        "p_per_aug": 0.4,
        "scale_range": (0.9, 1.1),
        "jitter_std": 0.05,
        "p_noise": 0.3,
        "p_freq_mask": 0.1,
        "max_freq_ratio": 0.1,
        "p_magnitude_warp": 0.3,
        "p_temporal_mask": 0.3,
        "p_temporal_crop": 0.0,
        "crop_ratio_range": (0.85, 1.0),

        # Views
        "repeat_factor": 4,             # 4 augmented t0 views
        "repeat_factor_val": 4,
        "include_prev": 1,              # 1 augmented t-1 view (no t-2 etc.)
        "prev_shift": 6,
        "include_clean_t0": True,

        # Encoder
        "encoder_name": "lstm",
        "input_channels": args.channels,
        "encoder_output_dim": args.encoder_dim,
        "pool_mode": "mean",
        "lstm_hidden_channels": 64,
        "lstm_layers": 2,
        "lstm_dropout": 0.1,
        "lstm_bidirectional": True,

        # Projector + SSL
        "proj_dim": args.proj_dim,
        "proj_hidden": 512,
        "lamb": 0.5,                    # SIGReg weight
        "sigreg_slices": 512,
        "sigreg_knots": 17,
        "sigreg_seed": 0,
        "p_t0": 0.9,
        "p_tminus1": 0.1,
        "sample_policy": "at_least_one_tminus1",

        # Optim / schedule
        "lr": 5e-4,
        "weight_decay": 1e-4,

        # Probe (eval-only head, helps monitor downstream utility)
        "probe_use": True,
        "probe_loss_weight": 0.0,       # no probe gradient into encoder (eval only)
    }
    return OmegaConf.create(cfg_dict)


def build_encoder(cfg: OmegaConf):
    return LSTMEncoder(
        input_channels=int(cfg.input_channels),
        output_dim=int(cfg.encoder_output_dim),
        pool_mode=str(cfg.pool_mode),
        hidden_channels=int(cfg.lstm_hidden_channels),
        num_layers=int(cfg.lstm_layers),
        dropout=float(cfg.lstm_dropout),
        bidirectional=bool(cfg.lstm_bidirectional),
    )


def build_modules(cfg: OmegaConf):
    encoder = build_encoder(cfg)
    projector = MLP(
        in_channels=int(cfg.encoder_output_dim),
        hidden_channels=[int(cfg.proj_hidden), int(cfg.proj_dim)],
    )
    ssl_core = LeJEPA_SSL(
        encoder=encoder,
        projector=projector,
        proj_dim=int(cfg.proj_dim),
        lamb=float(cfg.lamb),
        invariance_mix=InvarianceMix(
            p_t0=float(cfg.p_t0),
            p_tminus1=float(cfg.p_tminus1),
            sample_policy=str(cfg.sample_policy),
        ),
        num_prev_views=int(cfg.include_prev),
        sigreg_slices=int(cfg.sigreg_slices),
        sigreg_knots=int(cfg.sigreg_knots),
        sigreg_seed=int(cfg.sigreg_seed),
    )
    probe = None
    if bool(cfg.probe_use):
        probe = ForecastProbe(
            input_dim=int(cfg.encoder_output_dim),
            horizon=int(cfg.target_window_size),
            output_channels=int(cfg.input_channels),
            use_covariates=False,
        )
    return ssl_core, probe


def main() -> None:
    p = argparse.ArgumentParser(description="Train LeJEPA SSL on PeMS08.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--channels", type=int, default=170, help="Sensor channels (subsample of 170).")
    p.add_argument("--window", type=int, default=96)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--encoder-dim", type=int, default=128)
    p.add_argument("--proj-dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lamb", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--precision", type=int, default=16, help="Lightning precision (16, 32).")
    p.add_argument("--max-steps", type=int, default=-1, help="Cap optimizer steps for quick smoke runs.")
    p.add_argument("--run-name", type=str, default="ssl_pems08")
    p.add_argument("--ckpt-dir", type=str, default="outputs/checkpoints")
    args = p.parse_args()

    L.seed_everything(args.seed)

    cfg = build_config(args)
    cfg.lr = args.lr
    cfg.lamb = args.lamb

    # Data: load PeMS08 and slice to the requested channel count.
    pems = PeMS08(root=str(ROOT / "data" / "pems08"))
    data = torch.tensor(pems.target.values, dtype=torch.float32)  # [T, 170]
    time_features = encode_timestamps_torch(pems.timestamps)     # [T, 6]
    if args.channels < data.shape[1]:
        # Deterministic slice of sensors
        data = data[:, : args.channels]
        print(f"[train_ssl] sliced PeMS08 sensors: {data.shape[1]} -> {args.channels}")
    elif args.channels > data.shape[1]:
        raise ValueError(
            f"--channels ({args.channels}) exceeds PeMS08 sensor count ({data.shape[1]})"
        )

    datamodule = TimeSeriesSSLDataModule(
        cfg,
        data=data,
        timestamps=pems.timestamps,
    )
    datamodule.setup()

    print(f"[train_ssl] train_ds={len(datamodule.train_ds)}  val_ds={len(datamodule.val_ds)}")

    # Model
    ssl_core, probe = build_modules(cfg)
    model = SSLPretrainModule(
        ssl_core=ssl_core,
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
        probe=probe,
        probe_loss_weight=float(cfg.probe_loss_weight),
        scheduler="cosine",
        warmup_epochs=2,
        min_lr=1e-6,
    )

    # Logger + callbacks
    tb_logger = TensorBoardLogger(
        save_dir=str(ROOT / "outputs" / "tb_logs"),
        name=args.run_name,
        default_hp_metric=False,
    )
    ckpt_cb = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=str(ROOT / args.ckpt_dir / args.run_name),
        filename="ssl-{epoch:02d}-{val/ssl_loss:.4f}",
        monitor="val/ssl_loss",
        mode="min",
        save_top_k=2,
        save_last=True,
    )
    lr_cb = L.pytorch.callbacks.LearningRateMonitor(logging_interval="epoch")

    trainer = L.Trainer(
        max_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        accelerator="auto",
        devices=1,
        precision=args.precision if args.precision in (16, 32) else "16-mixed",
        logger=tb_logger,
        callbacks=[ckpt_cb, lr_cb],
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        enable_progress_bar=True,
        check_val_every_n_epoch=1,
    )

    print(f"[train_ssl] TensorBoard logdir: {tb_logger.log_dir}")
    trainer.fit(model, datamodule=datamodule)

    print(f"[train_ssl] done. best ckpt: {ckpt_cb.best_model_path}")


if __name__ == "__main__":
    main()