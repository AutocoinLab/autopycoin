"""
Initialization of autopycoin package.
"""

from ._version import __version__
from .baseclass import AutopycoinBaseClass, AutopycoinBaseLayer

from . import dataset
from . import data
from . import layers
from . import losses
from . import models
from . import test_utils
from . import utils

__all__ = [
    "__version__",
    "AutopycoinBaseClass",
    "AutopycoinBaseLayer",
    "dataset",
    "data",
    "layers",
    "losses",
    "models",
    "test_utils",
    "utils",
]
