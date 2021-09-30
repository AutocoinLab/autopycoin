""" This subpackage defines all metrics needed to train a deep learning.
"""

from .check_data import check_infinity, layer_test

__all__ = ["check_infinity", "layer_test"]
