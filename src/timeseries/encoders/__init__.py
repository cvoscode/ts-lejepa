"""
Time series encoders following the BaseEncoder pattern.

All encoders inherit from BaseEncoder and implement:
- forward_backbone(x, time_features) -> [B, T, D]
- _init_weights(m) for proper initialization
- Support for pool_mode: "mean", "max", "last", "none"
"""

from .rnn import LSTMEncoder, GRUEncoder
from .cnn import CNNEncoder
from .tcn import TCNEncoder
from .transformer import TransformerEncoder
from .upernet import UPerNetEncoder
from .mamba import MambaEncoder
#from .graphlstm import GraphLSTMEncoder
from .covariate import CovariateEncoder, FusedEncoder, FutureCovariateEncoder

__all__ = [
    "LSTMEncoder",
    "GRUEncoder",
    "CNNEncoder",
    "TCNEncoder",
    "TransformerEncoder",
    "UPerNetEncoder",
    "MambaEncoder",
    #"GraphLSTMEncoder",
    "CovariateEncoder",
    "FusedEncoder",
    "FutureCovariateEncoder",
]
