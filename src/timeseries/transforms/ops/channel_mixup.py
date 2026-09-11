"""Per-channel mixup across samples in a batch.

For each augmented view, replace a random subset of channels with values from
another sample in the batch. This is the channel-mixing signal that drives
channel-independent representations (PatchTST / iTransformer / FDS-Net style).

The op is stateful across calls because it needs to draw a partner from the
batch. The view builder seeds it by calling ``set_batch(other_views)`` once
per batch, then ``__call__`` consumes the partner pool view-by-view.

Expected input shape: ``[B, C, T]`` (matches the SSL view contract).
Returns the same shape.
"""

from __future__ import annotations

from typing import Optional

import torch

from ..base import Transform


class ChannelMixup(Transform):
    """Replace a random subset of channels with another sample's channels.

    Args:
        p: Probability of applying channel mixup to a given view.
        max_channels: Maximum number of channels to swap per view (drawn
            uniformly from ``[1, max_channels]`` when ``max_channels >= 1``).
            If None, draw ``U[0.1, 0.5] * C`` channels.
        partner_pool: Optional pre-built pool of partner views to draw from.
            If None, uses an internal pool set via :meth:`set_batch`.
    """

    def __init__(
        self,
        p: float = 0.3,
        max_channels: Optional[int] = None,
        partner_pool: Optional[torch.Tensor] = None,
    ) -> None:
        self.p = float(p)
        self.max_channels = max_channels
        self._partner_pool: Optional[torch.Tensor] = partner_pool

    def set_batch(self, partner_pool: torch.Tensor) -> None:
        """Seed the partner pool (called once per batch by the view builder).

        ``partner_pool`` should be a tensor of clean windows, shape ``[P, C, T]``.
        Each augmented view will sample one partner from this pool.
        """
        self._partner_pool = partner_pool

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self._partner_pool is None or self._partner_pool.shape[0] == 0:
            return x

        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(0)
        # Accept [B, C, T] or [B, V, C, T] by collapsing V into B for the
        # mixup math, then reshaping back.
        leading_dims = x.shape[:-2]
        C, T = x.shape[-2], x.shape[-1]
        x_flat = x.reshape(-1, C, T)
        B = x_flat.shape[0]

        # Decide how many channels to mix per row.
        if self.max_channels is not None and self.max_channels > 0:
            max_mix = int(self.max_channels)
        else:
            max_mix = max(1, C // 2)
        # n_mix[i] in [1, max_mix]; rows that "skip" mixup get n_mix=0.
        n_mix = torch.randint(0, max_mix + 1, (B,), device=x.device)
        # Probability gate
        if self.p < 1.0:
            keep_prob = torch.rand(B, device=x.device) < self.p
            n_mix = torch.where(keep_prob, n_mix, torch.zeros_like(n_mix))

        # Partner per row.
        P = self._partner_pool.shape[0]
        partner_idx = torch.randint(0, P, (B,), device=x.device)
        partner = self._partner_pool[partner_idx]  # [B, C, T]

        # Build a [B, C] boolean mask: True = swap with partner.
        # Strategy: per row, draw n_mix[i] distinct random channels by sorting
        # a uniform random vector and taking the top-n_mix[i] indices.
        mask = torch.zeros(B, C, dtype=torch.bool, device=x.device)
        if max_mix > 0:
            rand = torch.rand(B, C, device=x.device)
            # For each row, argsort gives a permutation; the first n_mix[i]
            # indices are the channels to swap.
            perm = torch.argsort(rand, dim=-1)  # [B, C]
            # Build an arange that is "1..n_mix[i]" for the first n_mix[i] indices
            # of perm. We do this by scatter from arange.
            range_idx = torch.arange(C, device=x.device).unsqueeze(0).expand(B, -1)  # [B, C]
            # range_after_perm[i, j] = rank of channel perm[i, j] within the row
            rank = torch.empty_like(range_idx)
            rank.scatter_(1, perm, range_idx)
            # Channels whose rank < n_mix[i] are swapped.
            swap = rank < n_mix.unsqueeze(1)
            mask = swap

        mask = mask.unsqueeze(-1)  # [B, C, 1]
        out = torch.where(mask, partner, x_flat)
        out = out.reshape(*leading_dims, C, T)
        return out.squeeze(0) if squeeze else out


__all__ = ["ChannelMixup"]
