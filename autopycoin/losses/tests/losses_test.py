"""Tests for autopycoin loss functions."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from autopycoin import losses
from tensorflow.python.keras import combinations
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.platform import test


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class MeanAbsoluteScaledErrorTest(test.TestCase):

  def test_config(self):
    y_train = constant_op.constant([3, 2, 10, 3, 5, 2], shape=(2, 3))
    seasonality = constant_op.constant(1)
    mase_obj = losses.MeanAbsoluteScaledError(
        reduction=losses_utils.ReductionV2.SUM, name='mase',
        y_train=y_train, seasonality=seasonality)
    self.assertEqual(mase_obj.name, 'mase')
    self.assertEqual(mase_obj.reduction, losses_utils.ReductionV2.SUM)

"""

  def test_all_correct_unweighted(self):
    y_train = constant_op.constant([3, 2, 10, 3, 5, 2], shape=(2, 3))
    seasonality = constant_op.constant(1)
    mase_obj = losses.MeanAbsoluteScaledError(y_train=y_train, seasonality=seasonality)
    y_true = constant_op.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
    loss = mase_obj(y_true, y_true)
    self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

  def test_unweighted(self):
    y_train = constant_op.constant([3, 2, 10, 3, 5, 2], shape=(2, 3))
    seasonality = constant_op.constant(1)
    mase_obj = losses.MeanAbsoluteScaledError(y_train=y_train, seasonality=seasonality)
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = mase_obj(y_true, y_pred)
    self.assertAlmostEqual(self.evaluate(loss), 1.0419412, 3)

  def test_mask_unweighted(self):
    y_train = constant_op.constant([3, 2, 10, 3, 5, 2], shape=(2, 3))
    seasonality = constant_op.constant(1)
    mase_obj = losses.MeanAbsoluteScaledError(y_train=y_train, seasonality=seasonality, mask=True)
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = mase_obj(y_true, y_pred)
    self.assertAlmostEqual(self.evaluate(loss), 0.6763285, 3)

  def test_scalar_weighted(self):
    y_train = constant_op.constant([3, 2, 10, 3, 5, 2], shape=(2, 3))
    seasonality = constant_op.constant(1)
    mase_obj = losses.MeanAbsoluteScaledError(y_train=y_train, seasonality=seasonality)
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = mase_obj(y_true, y_pred, sample_weight=2.3)
    self.assertAlmostEqual(self.evaluate(loss), 2.3964646, 3)

  def test_sample_weighted(self):
    y_train = constant_op.constant([3, 2, 10, 3, 5, 2], shape=(2, 3))
    seasonality = constant_op.constant(1)
    mase_obj = losses.MeanAbsoluteScaledError(y_train=y_train, seasonality=seasonality)
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    sample_weight = constant_op.constant([1.2, 3.4], shape=(2, 1))
    loss = mase_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 2.7648222, 3)

  def test_timestep_weighted(self):
    y_train = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3, 1))
    seasonality = constant_op.constant(1)
    mase_obj = losses.MeanAbsoluteScaledError(y_train=y_train, seasonality=seasonality)
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3, 1))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3, 1),
                                  dtype=dtypes.float32)
    sample_weight = constant_op.constant([3, 6], shape=(2, 1))
    loss = mase_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 4.00782, 3)

  def test_zero_weighted(self):
    y_train = constant_op.constant([3, 2, 10, 3, 5, 2], shape=(2, 3))
    seasonality = constant_op.constant(1)
    mase_obj = losses.MeanAbsoluteScaledError(y_train=y_train, seasonality=seasonality)
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = mase_obj(y_true, y_pred, sample_weight=0)
    self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

  def test_no_reduction(self):
    y_train = constant_op.constant([3, 2, 10, 3, 5, 2], shape=(2, 3))
    seasonality = constant_op.constant(1)
    mase_obj = losses.MeanAbsoluteScaledError(y_train=y_train, 
                                              seasonality=seasonality, 
                                              reduction=losses_utils.ReductionV2.NONE)
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = mase_obj(y_true, y_pred, sample_weight=2.3)
    loss = self.evaluate(loss)
    self.assertArrayNear(loss, [1.626262, 3.166667], 1e-3)

  def test_sum_reduction(self):
    y_train = constant_op.constant([3, 2, 10, 3, 5, 2], shape=(2, 3))
    seasonality = constant_op.constant(1)
    mase_obj = losses.MeanAbsoluteScaledError(y_train=y_train,
                                              seasonality=seasonality,
                                              reduction=losses_utils.ReductionV2.SUM)
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = mase_obj(y_true, y_pred, sample_weight=2.3)
    self.assertAlmostEqual(self.evaluate(loss), 4.792929, 3)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class SymetricMeanAbsolutePercentageErrorTest(test.TestCase):

  def test_config(self):
    smape_obj = losses.SymetricMeanAbsolutePercentageError(
        reduction=losses_utils.ReductionV2.SUM, name='mase')
    self.assertEqual(smape_obj.name, 'mase')
    self.assertEqual(smape_obj.reduction, losses_utils.ReductionV2.SUM)

  def test_all_correct_unweighted(self):
    smape_obj = losses.SymetricMeanAbsolutePercentageError()
    y_true = constant_op.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
    loss = smape_obj(y_true, y_true)
    self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

  def test_unweighted(self):
    smape_obj = losses.SymetricMeanAbsolutePercentageError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = smape_obj(y_true, y_pred)
    self.assertAlmostEqual(self.evaluate(loss), 123.54809, 3)

  def test_mask_unweighted(self):
    smape_obj = losses.SymetricMeanAbsolutePercentageError(mask=True)
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = smape_obj(y_true, y_pred)
    self.assertAlmostEqual(self.evaluate(loss), 41.182697, 3)

  def test_scalar_weighted(self):
    smape_obj = losses.SymetricMeanAbsolutePercentageError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = smape_obj(y_true, y_pred, sample_weight=2.3)
    self.assertAlmostEqual(self.evaluate(loss), 284.16058, 3)

  def test_sample_weighted(self):
    smape_obj = losses.SymetricMeanAbsolutePercentageError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    sample_weight = constant_op.constant([1.2, 3.4], shape=(2, 1))
    loss = smape_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 319.36884, 3)

  def test_timestep_weighted(self):
    smape_obj = losses.SymetricMeanAbsolutePercentageError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3, 1))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3, 1),
                                  dtype=dtypes.float32)
    sample_weight = constant_op.constant([3, 6], shape=(2, 1))
    loss = smape_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 603.9776, 3)

  def test_zero_weighted(self):
    smape_obj = losses.SymetricMeanAbsolutePercentageError()
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = smape_obj(y_true, y_pred, sample_weight=0)
    self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

  def test_no_reduction(self):
    smape_obj = losses.SymetricMeanAbsolutePercentageError(
                                              reduction=losses_utils.ReductionV2.NONE)
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = smape_obj(y_true, y_pred, sample_weight=2.3)
    loss = self.evaluate(loss)
    self.assertArrayNear(loss, [210.543427, 357.777771], 1e-3)

  def test_sum_reduction(self):
    smape_obj = losses.SymetricMeanAbsolutePercentageError(
                                              reduction=losses_utils.ReductionV2.SUM)
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = smape_obj(y_true, y_pred, sample_weight=2.3)
    self.assertAlmostEqual(self.evaluate(loss), 568.32117, 3)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class OverallWeightedAverageErrorTest(test.TestCase):

  def test_config(self):
    y_train = constant_op.constant([3, 2, 10, 3, 5, 2], shape=(2, 3))
    seasonality = constant_op.constant(1)
    owa_obj = losses.OverallWeightedAverageError(
        reduction=losses_utils.ReductionV2.SUM, name='mase',
        y_train=y_train, seasonality=seasonality)
    self.assertEqual(owa_obj.name, 'mase')
    self.assertEqual(owa_obj.reduction, losses_utils.ReductionV2.SUM)

  def test_all_correct_unweighted(self):
    y_train = constant_op.constant([3, 2, 10, 3, 5, 2], shape=(2, 3))
    seasonality = constant_op.constant(1)
    owa_obj = losses.OverallWeightedAverageError(y_train=y_train, seasonality=seasonality)
    y_true = constant_op.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
    loss = owa_obj(y_true, y_true)
    self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

  def test_unweighted(self):
    y_train = constant_op.constant([3, 2, 10, 3, 5, 2], shape=(2, 3))
    seasonality = constant_op.constant(1)
    owa_obj = losses.OverallWeightedAverageError(y_train=y_train, seasonality=seasonality)
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = owa_obj(y_true, y_pred)
    self.assertAlmostEqual(self.evaluate(loss), 62.295013, 3)

  def test_mask_unweighted(self):
    y_train = constant_op.constant([3, 2, 10, 3, 5, 2], shape=(2, 3))
    seasonality = constant_op.constant(1)
    owa_obj = losses.OverallWeightedAverageError(y_train=y_train, seasonality=seasonality, mask=True)
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = owa_obj(y_true, y_pred)
    self.assertAlmostEqual(self.evaluate(loss), 20.929512, 3)

  def test_scalar_weighted(self):
    y_train = constant_op.constant([3, 2, 10, 3, 5, 2], shape=(2, 3))
    seasonality = constant_op.constant(1)
    owa_obj = losses.OverallWeightedAverageError(y_train=y_train, seasonality=seasonality)
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = owa_obj(y_true, y_pred, sample_weight=2.3)
    self.assertAlmostEqual(self.evaluate(loss), 143.27853, 3)

  def test_sample_weighted(self):
    y_train = constant_op.constant([3, 2, 10, 3, 5, 2], shape=(2, 3))
    seasonality = constant_op.constant(1)
    owa_obj = losses.OverallWeightedAverageError(y_train=y_train, seasonality=seasonality)
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    sample_weight = constant_op.constant([1.2, 3.4], shape=(2, 1))
    loss = owa_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 161.0668, 3)

  def test_timestep_weighted(self):
    y_train = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3, 1))
    seasonality = constant_op.constant(1)
    owa_obj = losses.OverallWeightedAverageError(y_train=y_train, seasonality=seasonality)
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3, 1))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3, 1),
                                  dtype=dtypes.float32)
    sample_weight = constant_op.constant([3, 6], shape=(2, 1))
    loss = owa_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), 303.9927, 3)

  def test_zero_weighted(self):
    y_train = constant_op.constant([3, 2, 10, 3, 5, 2], shape=(2, 3))
    seasonality = constant_op.constant(1)
    owa_obj = losses.OverallWeightedAverageError(y_train=y_train, seasonality=seasonality)
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = owa_obj(y_true, y_pred, sample_weight=0)
    self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

  def test_no_reduction(self):
    y_train = constant_op.constant([3, 2, 10, 3, 5, 2], shape=(2, 3))
    seasonality = constant_op.constant(1)
    owa_obj = losses.OverallWeightedAverageError(y_train=y_train, 
                                              seasonality=seasonality, 
                                              reduction=losses_utils.ReductionV2.NONE)
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = owa_obj(y_true, y_pred, sample_weight=2.3)
    loss = self.evaluate(loss)
    self.assertArrayNear(loss, [106.084839, 180.472214], 1e-3)

  def test_sum_reduction(self):
    y_train = constant_op.constant([3, 2, 10, 3, 5, 2], shape=(2, 3))
    seasonality = constant_op.constant(1)
    owa_obj = losses.OverallWeightedAverageError(y_train=y_train,
                                              seasonality=seasonality,
                                              reduction=losses_utils.ReductionV2.SUM)
    y_true = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    y_pred = constant_op.constant([4, 8, 12, 8, 1, 3],
                                  shape=(2, 3),
                                  dtype=dtypes.float32)
    loss = owa_obj(y_true, y_pred, sample_weight=2.3)
    self.assertAlmostEqual(self.evaluate(loss), 286.55707, 3)
"""