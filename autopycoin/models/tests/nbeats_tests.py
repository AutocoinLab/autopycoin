"""Test for nbeats model

Protocol for testing N_BEATS model:

We decided to split up the model in multiple part as it can be easier to debug and test each part.
We will test each composant known as TrendBlock, SeasonalityBlock, Stack and finally the wrapper N_BEATS.
Each of them will be compare to expected values then we will perform some classical checks
that can be found in tensorflow.

TrendBlock : 
- It is important to perform check with constant dense weights.
FC_backcast, FC_forecast, FC_stack will be constrained. dropout will be set to 0.

forecast_coef and backcast_coef will be test and n_quantiles will be test.
"""

from autopycoin import keras
from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.eager import context
from tensorflow.python.keras import combinations
from tensorflow.python.keras import testing_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test

"""@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class TrendBlockError(test.TestCase):

  def test_config(self):
    y_train = constant_op.constant([3, 2, 10, 3, 5, 2], shape=(2, 3))
    seasonality = constant_op.constant(1)
    mase_obj = losses.MeanAbsoluteScaledError(
        reduction=losses_utils.ReductionV2.SUM, name='mase',
        y_train=y_train, seasonality=seasonality)
    self.assertEqual(mase_obj.name, 'mase')
    self.assertEqual(mase_obj.reduction, losses_utils.ReductionV2.SUM)"""
