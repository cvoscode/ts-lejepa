"""Smoke runner for the LeJEPA model concept in src/timeseries.

Builds multi-view windows, runs them through a real encoder + projector,
computes invariance + SIGReg losses, and performs a backward step.

Usage:
    python scripts/run_concept.py
    python scripts/run_concept.py --steps 5 --batch 4 --channels 8 --length 96
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the src/timeseries package importable when run from repo root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn as nn

from timeseries.encoders.rnn import LSTMEncoder
from timeseries.ssl.lejepa import LeJEPA_SSL


def build_views(batch: int, channels: int, length: int, num_prev: int, num_aug: int) -> torch.Tensor:
    """Construct the canonical view tensor.

    Layout: [t-1_aug..., t0_clean, t0_aug..., ...]
    Shape: [B, V, C, L]
    """
    # t-1 windows (augmented copies)
    t1 = torch.randn(batch, num_prev, channels, length)
    # Clean t0 anchor
    t0_clean = torch.randn(batch, 1, channels, length)
    # Augmented t0 views
    t0_aug = torch.randn(batch, num_aug, channels, length)
    return torch.cat([t1, t0_clean, t0_aug], dim=1)


def main() -> None:
    p = argparse.ArgumentParser(description="Run the LeJEPA model concept on synthetic data.")
    p.add_argument("--steps", type=int, default=3, help="Optimization steps.")
    p.add_argument("--batch", type=int, default=4, help="Batch size.")
    p.add_argument("--channels", type=int, default=8, help="Sensor channels.")
    p.add_argument("--length", type=int, default=96, help="Window length (timesteps).")
    p.add_argument("--num-prev", type=int, default=1, help="Number of augmented t-1 views.")
    p.add_argument("--num-aug", type=int, default=4, help="Number of augmented t0 views.")
    p.add_argument("--proj-dim", type=int, default=64, help="Projector output dimension.")
    p.add_argument("--encoder-dim", type=int, default=32, help="Encoder output dimension.")
    p.add_argument("--lamb", type=float, default=0.5, help="SIGReg weight.")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed.")
    args = p.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run_concept] device={device}")

    # Real encoder from the package.
    encoder = LSTMEncoder(
        input_channels=args.channels,
        output_dim=args.encoder_dim,
        pool_mode="mean",
        hidden_channels=32,
        num_layers=1,
        bidirectional=False,
    )

    projector = nn.Sequential(
        nn.Linear(args.encoder_dim, args.encoder_dim),
        nn.GELU(),
        nn.Linear(args.encoder_dim, args.proj_dim),
    )

    ssl = LeJEPA_SSL(
        encoder=encoder,
        projector=projector,
        proj_dim=args.proj_dim,
        lamb=args.lamb,
        num_prev_views=args.num_prev,
        include_clean_t0=True,
    ).to(device)

    optim = torch.optim.Adam(ssl.parameters(), lr=args.lr)

    views = build_views(
        batch=args.batch,
        channels=args.channels,
        length=args.length,
        num_prev=args.num_prev,
        num_aug=args.num_aug,
    ).to(device)

    print(
        f"[run_concept] views shape={tuple(views.shape)} "
        f"(B,V,C,L)  V={views.shape[1]} (num_prev + clean_t0 + num_aug)"
    )

    for step in range(1, args.steps + 1):
        optim.zero_grad()
        res = ssl(views)
        res.total_loss.backward()
        optim.step()

        std = res.embedding_std.item() if res.embedding_std is not None else float("nan")
        collapse = res.feature_collapse_ratio.item() if res.feature_collapse_ratio is not None else float("nan")
        print(
            f"[step {step:02d}] total={res.total_loss.item():.4f} "
            f"inv={res.inv_loss.item():.4f} sigreg={res.sigreg_loss.item():.4f} "
            f"emb_std={std:.4f} collapse_ratio={collapse:.4f}"
        )

    print("[run_concept] done.")


if __name__ == "__main__":
    main()