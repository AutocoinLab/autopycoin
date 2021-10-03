""" This subpackage defines all metrics needed to train a deep learning.
"""

from .data_utils import check_infinity, range_dims
from .testing_utils import layer_test

__all__ = ["check_infinity", "layer_test", "range_dims"]
