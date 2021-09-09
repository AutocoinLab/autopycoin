""" This subpackage defines additional losses needed to train a deep learning.
"""

from .losses import smape
from .losses import quantile_loss
from .losses import SymetricMeanAbsolutePercentageError
from .losses import QuantileLossError

__all__ = [
    "smape",
    "quantile_loss",
    "SymetricMeanAbsolutePercentageError",
    "QuantileLossError"
    ]