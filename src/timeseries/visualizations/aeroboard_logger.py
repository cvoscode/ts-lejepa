"""Aeroboard integration for the lejepa SSL training pipeline.

Provides a thin adapter that augments ``AeroboardLightningLogger`` with the
``add_image`` / ``add_figure`` / ``flush`` methods expected by the existing
``VisualizationCallback``. When the ``aeroboard-client`` extra is not installed,
the factory returns ``None`` and callers should fall back to TensorBoard.
"""

from __future__ import annotations

import io
import logging
from typing import Any, Optional

from lightning.pytorch.loggers import Logger as LightningLoggerBase

logger = logging.getLogger(__name__)


def _has_aeroboard() -> bool:
    try:
        import aeroboard_client  # noqa: F401
        return True
    except ImportError:
        return False


def build_aeroboard_logger(
    *,
    run_id: str,
    url: str = "grpc://localhost:50051",
    api_base_url: Optional[str] = None,
    batch_size: int = 100,
) -> Optional[LightningLoggerBase]:
    """Construct an Aeroboard Lightning logger with image-logging shims.

    Returns ``None`` if ``aeroboard-client`` is not installed so callers can
    fall back to TensorBoard without hard-failing.
    """
    if not _has_aeroboard():
        logger.warning(
            "aeroboard-client is not installed; skipping Aeroboard logger. "
            "Install with `uv sync --extra dashboards`."
        )
        return None

    from aeroboard_client import AeroboardClient, AeroboardLightningLogger

    client = AeroboardClient(
        run_id=run_id,
        url=url,
        api_base_url=api_base_url,
        batch_size=batch_size,
    )
    base = AeroboardLightningLogger(client=client)
    return _AeroboardLoggerAdapter(base)


class _AeroboardLoggerAdapter(LightningLoggerBase):
    """Wraps an ``AeroboardLightningLogger`` and exposes image/figure methods.

    The existing ``VisualizationCallback`` checks ``hasattr(experiment, ...)``
    for ``add_image`` / ``add_figure`` / ``flush`` (see
    ``timeseries.visualizations.callbacks``). Aeroboard's logger exposes the
    raw ``AeroboardClient`` as ``experiment``, which only supports
    ``log_metrics`` / ``log_image`` / ``log_hyperparameters``. This adapter
    forwards ``experiment`` lookups to the underlying logger so scalar and
    hparam logging is unaffected, while adding the shim methods the callback
    expects.
    """

    def __init__(self, inner: LightningLoggerBase) -> None:
        super().__init__()
        self._inner = inner

    # ---- Delegation to the inner logger -----------------------------------

    @property
    def name(self) -> str:
        return self._inner.name  # type: ignore[no-any-return]

    @property
    def version(self) -> str:
        return self._inner.version  # type: ignore[no-any-return]

    @property
    def experiment(self) -> Any:
        """Return ``self`` so ``experiment.add_image(...)`` hits this adapter."""
        return self

    @property
    def hparams(self) -> dict[str, Any]:
        return getattr(self._inner, "hparams", {})

    def log_hyperparams(self, params: Any) -> None:
        self._inner.log_hyperparams(params)

    def log_metrics(self, metrics: Any, step: Optional[int] = None) -> None:
        self._inner.log_metrics(metrics, step=step)

    def save(self) -> None:
        self._inner.save()

    def finalize(self, status: str) -> None:
        self._inner.finalize(status)

    # ---- Image / figure shims used by VisualizationCallback ----------------

    def add_image(
        self,
        tag: str,
        img: Any,
        global_step: Optional[int] = None,
    ) -> None:
        """Forward an image to Aeroboard's ``log_image``.

        Accepts the same input formats as ``AeroboardClient.log_image``
        (PIL images, numpy arrays, torch tensors, raw bytes).
        """
        client = getattr(self._inner, "client", None)
        if client is None:
            logger.debug("Aeroboard client unavailable; skipping add_image(%s)", tag)
            return
        client.log_image(tag, img, format=None)

    def add_figure(
        self,
        tag: str,
        figure: Any,
        global_step: Optional[int] = None,
    ) -> None:
        """Encode a matplotlib figure to PNG bytes and forward as an image."""
        try:
            buf = io.BytesIO()
            figure.savefig(buf, format="png", bbox_inches="tight")
            png_bytes = buf.getvalue()
        except Exception as exc:  # pragma: no cover - plotting failures are noisy
            logger.warning("Failed to encode figure for Aeroboard (%s): %s", tag, exc)
            return
        client = getattr(self._inner, "client", None)
        if client is None:
            return
        client.log_image(tag, png_bytes, format="png")

    def flush(self) -> None:
        """Flush buffered metrics/images to the Aeroboard server."""
        try:
            self.save()
        except Exception as exc:  # pragma: no cover
            logger.debug("Aeroboard flush failed: %s", exc)


__all__ = ["build_aeroboard_logger"]