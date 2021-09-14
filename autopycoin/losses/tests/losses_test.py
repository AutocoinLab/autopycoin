"""Tests for autopycoin loss functions."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from autopycoin import losses
from tensorflow.python.keras import combinations
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.platform import test


@combinations.generate(combinations.combine(mode=["graph", "eager"]))
class MeanAbsoluteScaledErrorTest(test.TestCase):
    def test_config(self):
        y_train = constant_op.constant([3, 2, 10, 3, 5, 2], shape=(2, 3))
        seasonality = constant_op.constant(1)
        mase_obj = losses.MeanAbsoluteScaledError(
            reduction=losses_utils.ReductionV2.SUM,
            name="mase",
            y_train=y_train,
            seasonality=seasonality,
        )
        self.assertEqual(mase_obj.name, "mase")
        self.assertEqual(mase_obj.reduction, losses_utils.ReductionV2.SUM)
