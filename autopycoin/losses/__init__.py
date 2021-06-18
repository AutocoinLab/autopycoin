""" This subpackage defines additional losses needed to train a deep learning.
"""

from ._losses import smape
from ._losses import mase
from ._losses import owa
from ._losses import MeanAbsoluteScaledError
from ._losses import SymetricMeanAbsolutePercentageError
from ._losses import OverallWeightedAverageError

__all__ = [
    "smape",
    "mase",
    "owa",
    "MeanAbsoluteScaledError",
    "SymetricMeanAbsolutePercentageError",
    "OverallWeightedAverageError"
    ]