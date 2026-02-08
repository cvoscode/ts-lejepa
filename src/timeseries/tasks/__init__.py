"""Task heads and probes for SSL evaluation."""

from .forecast_probe import ForecastProbe
from .sensor_probe import SensorProbe
from .classification_probe import ClassificationProbe
from .anomaly_probe import AnomalyProbe
from .eval_protocol import DownstreamEvaluator, EvalResult

__all__ = [
    "ForecastProbe",
    "SensorProbe",
    "ClassificationProbe",
    "AnomalyProbe",
    "DownstreamEvaluator",
    "EvalResult",
]
