""" This subpackage defines additional losses needed to train a deep learning.
"""

from .losses import (
    SymetricMeanAbsolutePercentageError,
    QuantileLossError,
    LagError,
    quantile_loss,
    smape,
)


__all__ = [
    "smape",
    "quantile_loss",
    "SymetricMeanAbsolutePercentageError",
    "QuantileLossError",
    "LagError",
]
