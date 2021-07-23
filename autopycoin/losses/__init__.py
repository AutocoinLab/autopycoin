""" This subpackage defines additional losses needed to train a deep learning.
"""

from .losses import smape
from .losses import mase
from .losses import owa
from .losses import quantile_loss
from .losses import MeanAbsoluteScaledError
from .losses import SymetricMeanAbsolutePercentageError
from .losses import OverallWeightedAverageError
from .losses import QuantileLossError

__all__ = [
    "smape",
    "mase",
    "owa",
    "quantile_loss",
    "MeanAbsoluteScaledError",
    "SymetricMeanAbsolutePercentageError",
    "OverallWeightedAverageError",
    "QuantileLossError"
    ]