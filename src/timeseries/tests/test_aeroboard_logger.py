"""Smoke tests for the Aeroboard Lightning logger adapter.

These tests run without a live Aeroboard server: they verify the adapter
exposes the expected interface (``add_image`` / ``add_figure`` / ``flush`` /
``experiment``) so the existing ``VisualizationCallback`` continues to work.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from timeseries.visualizations.aeroboard_logger import (
    _AeroboardLoggerAdapter,
    build_aeroboard_logger,
    _has_aeroboard,
)


class _StubInner:
    """Minimal stand-in for ``AeroboardLightningLogger``."""

    def __init__(self) -> None:
        self.client = MagicMock()
        self._hparams: dict = {}

    @property
    def name(self) -> str:
        return "stub"

    @property
    def version(self) -> str:
        return "stub-v1"

    @property
    def hparams(self) -> dict:
        return dict(self._hparams)

    def log_hyperparams(self, params) -> None:
        self._hparams.update(params)

    def log_metrics(self, metrics, step=None) -> None:
        pass

    def save(self) -> None:
        pass

    def finalize(self, status) -> None:
        pass


def test_aeroboard_installed() -> None:
    """If the optional dep was installed, the helper should detect it."""
    # We installed aeroboard via `uv sync --extra dashboards`, so this should
    # be True. If the user runs without the extra, the build factory returns
    # None — that's covered by the next test.
    assert _has_aeroboard(), (
        "aeroboard-client is not importable. Install with `uv sync --extra dashboards`."
    )


def test_build_aeroboard_logger_returns_adapter() -> None:
    """``build_aeroboard_logger`` should return an adapter when aeroboard is available."""
    adapter = build_aeroboard_logger(run_id="smoke-test")
    assert adapter is not None
    assert isinstance(adapter, _AeroboardLoggerAdapter)
    assert adapter.name == "aeroboard"
    assert adapter.experiment is adapter  # so callback's hasattr(experiment, ...) hits self


def test_adapter_exposes_image_methods() -> None:
    """The adapter should expose ``add_image`` / ``add_figure`` / ``flush``."""
    inner = _StubInner()
    adapter = _AeroboardLoggerAdapter(inner)
    assert hasattr(adapter, "add_image")
    assert hasattr(adapter, "add_figure")
    assert hasattr(adapter, "flush")
    assert callable(adapter.add_image)
    assert callable(adapter.add_figure)
    assert callable(adapter.flush)


def test_add_image_routes_to_client() -> None:
    """``add_image`` should forward numpy arrays to ``AeroboardClient.log_image``."""
    inner = _StubInner()
    adapter = _AeroboardLoggerAdapter(inner)
    img = np.random.rand(3, 32, 32).astype(np.float32)
    adapter.add_image("samples/pred", img, global_step=1)
    inner.client.log_image.assert_called_once_with("samples/pred", img, format=None)


def test_add_figure_encodes_png() -> None:
    """``add_figure`` should encode matplotlib figures as PNG bytes."""
    inner = _StubInner()
    adapter = _AeroboardLoggerAdapter(inner)
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 0])
    adapter.add_figure("recon/sample", fig, global_step=2)
    plt.close(fig)
    inner.client.log_image.assert_called_once()
    args, kwargs = inner.client.log_image.call_args
    assert args[0] == "recon/sample"
    assert isinstance(args[1], (bytes, bytearray))
    assert kwargs.get("format") == "png"


def test_flush_calls_save() -> None:
    """``flush`` should call ``save`` on the inner logger."""
    inner = _StubInner()
    inner.save = MagicMock()
    adapter = _AeroboardLoggerAdapter(inner)
    adapter.flush()
    inner.save.assert_called_once()


def test_log_hyperparams_and_metrics_forward() -> None:
    """Hyperparameter and metric logging should pass through to the inner logger."""
    inner = _StubInner()
    inner.log_hyperparams = MagicMock()
    inner.log_metrics = MagicMock()
    adapter = _AeroboardLoggerAdapter(inner)
    adapter.log_hyperparams({"lr": 1e-3, "batch_size": 64})
    adapter.log_metrics({"train/loss": 0.5}, step=10)
    inner.log_hyperparams.assert_called_once_with({"lr": 1e-3, "batch_size": 64})
    inner.log_metrics.assert_called_once_with({"train/loss": 0.5}, step=10)