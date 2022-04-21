""" This subpackage defines additional losses needed to train a deep learning model.
"""

from .losses import (
    SymetricMeanAbsolutePercentageError,
    QuantileLossError,
    LossQuantileDimWrapper,
    quantile_loss,
    smape,
    remove_dims,
    expand_dims
)


__all__ = [
    "smape",
    "quantile_loss",
    "LossQuantileDimWrapper",
    "SymetricMeanAbsolutePercentageError",
    "QuantileLossError",
]
