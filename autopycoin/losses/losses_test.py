"""Tests for autopycoin loss functions."""

import numpy as np

import tensorflow as tf
from tensorflow import test
from tensorflow.python.keras import combinations
from tensorflow.python.keras.utils import losses_utils
from . import losses


@combinations.generate(combinations.combine(mode=["eager", "graph"]))
class QuantileLossTest(test.TestCase):
    def test_config(self):
        ql_obj = losses.QuantileLossError(
            quantiles=[0.1, 0.5, 0.9],
            reduction=losses_utils.ReductionV2.SUM,
            name="ql_1",
        )
        self.assertEqual(ql_obj.name, "ql_1")
        self.assertEqual(ql_obj.reduction, losses_utils.ReductionV2.SUM)

    def test_all_correct_unweighted(self):
        ql_obj = losses.QuantileLossError(quantiles=[0.1, 0.5, 0.9])
        y_true = tf.constant(
            [[4, 8, 12, 8, 1, 3], [4, 8, 12, 8, 1, 3], [4, 8, 12, 8, 1, 3]],
            shape=(3, 2, 3),
            dtype=tf.float32
        )
        loss = ql_obj(y_true, y_true)
        self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

    def test_unweighted(self):
        ql_obj = losses.QuantileLossError(quantiles=[0.1, 0.5, 0.9])
        y_true = tf.constant(
            [[1, 9, 2, -5, -2, 6], [1, 9, 2, -5, -2, 6], [1, 9, 2, -5, -2, 6]],
            shape=(3, 2, 3),
        )
        y_pred = tf.constant(
            [[4, 8, 12, 8, 1, 3], [4, 8, 12, 8, 1, 3], [4, 8, 12, 8, 1, 3]],
            shape=(3, 2, 3),
            dtype=tf.float32,
        )
        loss = ql_obj(y_true, y_pred)
        self.assertAlmostEqual(self.evaluate(loss), 24.75, 3)

    def test_scalar_weighted(self):
        ql_obj = losses.QuantileLossError(quantiles=[0.1, 0.5, 0.9])
        y_true = tf.constant(
            [[1, 9, 2, -5, -2, 6], [1, 9, 2, -5, -2, 6], [1, 9, 2, -5, -2, 6]],
            shape=(3, 2, 3),
        )
        y_pred = tf.constant(
            [[4, 8, 12, 8, 1, 3], [4, 8, 12, 8, 1, 3], [4, 8, 12, 8, 1, 3]],
            shape=(3, 2, 3),
            dtype=tf.float32,
        )
        loss = ql_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(self.evaluate(loss), 56.925, 3)

    def test_sample_weighted(self):
        ql_obj = losses.QuantileLossError(quantiles=[0.1, 0.5, 0.9])
        y_true = tf.constant(
            [[1, 9, 2, -5, -2, 6], [1, 9, 2, -5, -2, 6], [1, 9, 2, -5, -2, 6]],
            shape=(3, 2, 3),
        )
        y_pred = tf.constant(
            [[4, 8, 12, 8, 1, 3], [4, 8, 12, 8, 1, 3], [4, 8, 12, 8, 1, 3]],
            shape=(3, 2, 3),
            dtype=tf.float32,
        )
        sample_weight = tf.constant([1.2, 3.4], shape=(2, 1))
        loss = ql_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(self.evaluate(loss), 122.1 / 2, 3)

    def test_ragged_tensors(self):
        ql_obj = losses.QuantileLossError(quantiles=[0.1, 0.5, 0.9])

        y_true = tf.ragged.constant(
            [
                [[1.0, 1.0, 9.0], [2.0, 5.0]],
                [[1.0, 1.0, 9.0], [2.0, 5.0]],
                [[1.0, 1.0, 9.0], [2.0, 5.0]],
            ]
        )
        y_pred = tf.ragged.constant(
            [
                [[4.0, 1.0, 8.0], [12.0, 3.0]],
                [[4.0, 1.0, 8.0], [12.0, 3.0]],
                [[4.0, 1.0, 8.0], [12.0, 3.0]],
            ],
            dtype=tf.float32,
        )
        sample_weight = tf.constant([1.2, 0.5])
        loss = ql_obj(y_true, y_pred, sample_weight=sample_weight)

        self.assertAllClose(self.evaluate(loss), 8.1, 1e-2)

    def test_timestep_weighted(self):
        ql_obj = losses.QuantileLossError(quantiles=[0.1, 0.5, 0.9])
        y_true = tf.constant(
            [[1, 9, 2, -5, -2, 6], [1, 9, 2, -5, -2, 6], [1, 9, 2, -5, -2, 6]],
            shape=(3, 2, 3, 1),
        )
        y_pred = tf.constant(
            [[4, 8, 12, 8, 1, 3], [4, 8, 12, 8, 1, 3], [4, 8, 12, 8, 1, 3]],
            shape=(3, 2, 3, 1),
            dtype=tf.float32,
        )
        sample_weight = tf.constant([3, 6, 5, 0, 4, 2], shape=(2, 3))
        loss = ql_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(self.evaluate(loss), 62.25, 3)

    def test_zero_weighted(self):
        ql_obj = losses.QuantileLossError(quantiles=[0.1, 0.5, 0.9])
        y_true = tf.constant(
            [[1, 9, 2, -5, -2, 6], [1, 9, 2, -5, -2, 6], [1, 9, 2, -5, -2, 6]],
            shape=(3, 2, 3),
        )
        y_pred = tf.constant(
            [[4, 8, 12, 8, 1, 3], [4, 8, 12, 8, 1, 3], [4, 8, 12, 8, 1, 3]],
            shape=(3, 2, 3),
            dtype=tf.float32,
        )
        loss = ql_obj(y_true, y_pred, sample_weight=0)
        self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

    def test_no_reduction(self):
        ql_obj = losses.QuantileLossError(
            quantiles=[0.1, 0.5, 0.9], reduction=losses_utils.ReductionV2.NONE
        )
        y_true = tf.constant(
            [[1, 9, 2, -5, -2, 6], [1, 9, 2, -5, -2, 6], [1, 9, 2, -5, -2, 6]],
            shape=(3, 2, 3),
        )
        y_pred = tf.constant(
            [[4, 8, 12, 8, 1, 3], [4, 8, 12, 8, 1, 3], [4, 8, 12, 8, 1, 3]],
            shape=(3, 2, 3),
            dtype=tf.float32,
        )
        loss = ql_obj(y_true, y_pred, sample_weight=2.3)
        loss = self.evaluate(loss)
        self.assertArrayNear(loss, [10.5 * 2.3, 14.25 * 2.3], 1e-3)

    def test_mean_reduction(self):
        ql_obj = losses.QuantileLossError(
            quantiles=[0.1, 0.5, 0.9], reduction=losses_utils.ReductionV2.AUTO
        )
        y_true = tf.constant(
            [[1, 9, 2, -5, -2, 6], [1, 9, 2, -5, -2, 6], [1, 9, 2, -5, -2, 6]],
            shape=(3, 2, 3),
        )
        y_pred = tf.constant(
            [[4, 8, 12, 8, 1, 3], [4, 8, 12, 8, 1, 3], [4, 8, 12, 8, 1, 3]],
            shape=(3, 2, 3),
            dtype=tf.float32,
        )
        loss = ql_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(self.evaluate(loss), 28.46, 2)


@combinations.generate(combinations.combine(mode=["eager", "graph"]))
class SymetricMeanAbsolutePercentageTest(test.TestCase):
    def test_config(self):
        smape_obj = losses.SymetricMeanAbsolutePercentageError(
            reduction=losses_utils.ReductionV2.AUTO,
            name="smape_1",
        )
        self.assertEqual(smape_obj.name, "smape_1")
        self.assertEqual(smape_obj.reduction, losses_utils.ReductionV2.AUTO)

    def test_all_correct_unweighted(self):
        smape_obj = losses.SymetricMeanAbsolutePercentageError()
        y_true = tf.constant(
            [4, 8, 12, 8, 1, 3],
            shape=(2, 3),
            dtype=tf.float32
        )
        loss = smape_obj(y_true, y_true)
        self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

    def test_unweighted(self):
        smape_obj = losses.SymetricMeanAbsolutePercentageError()
        y_true = tf.constant(
            [1, 9, 2, -5, -2, 6],
            shape=(2, 3),
        )
        y_pred = tf.constant(
            [4, 8, 12, 8, 1, 3],
            shape=(2, 3),
            dtype=tf.float32
        )
        loss = smape_obj(y_true, y_pred)
        self.assertAlmostEqual(self.evaluate(loss), 123.548, 3)

    def test_scalar_weighted(self):
        smape_obj = losses.SymetricMeanAbsolutePercentageError()
        y_true = tf.constant(
            [1, 9, 2, -5, -2, 6],
            shape=(2, 3),
        )
        y_pred = tf.constant(
            [4, 8, 12, 8, 1, 3],
            shape=(2, 3),
            dtype=tf.float32,
        )
        loss = smape_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(self.evaluate(loss), 2.3 * 123.548, 3)

    def test_sample_weighted(self):
        smape_obj = losses.SymetricMeanAbsolutePercentageError()
        y_true = tf.constant(
            [1, 9, 2, -5, -2, 6],
            shape=(2, 3),
        )
        y_pred = tf.constant(
            [4, 8, 12, 8, 1, 3],
            shape=(2, 3),
            dtype=tf.float32,
        )

        sample_weight = tf.constant([1.2, 3.4], shape=(2, 1))
        loss = smape_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(self.evaluate(loss), 319.369, 3)

    def test_ragged_tensors(self):
        smape_obj = losses.SymetricMeanAbsolutePercentageError()

        y_true = tf.ragged.constant(
            
                [[1.0, 1.0, 9.0], [2.0, 5.0]],
            
        )
        y_pred = tf.ragged.constant(
            [
                [4.0, 1.0, 8.0], [12.0, 3.0]
            ],
            dtype=tf.float32,
        )
        sample_weight = tf.constant([1.2, 0.5])
        loss = smape_obj(y_true, y_pred, sample_weight=sample_weight)

        self.assertAllClose(self.evaluate(loss), 50.460, 1e-2)

    def test_timestep_weighted(self):
        smape_obj = losses.SymetricMeanAbsolutePercentageError()
        y_true = tf.constant(
            [1, 9, 2, -5, -2, 6],
            shape=(2, 3, 1),
        )
        y_pred = tf.constant(
            [4, 8, 12, 8, 1, 3],
            shape=(2, 3, 1),
            dtype=tf.float32,
        )

        sample_weight = tf.constant([3, 6, 5, 0, 4, 2], shape=(2, 3))
        loss = smape_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(self.evaluate(loss), 115.456, 3)

    def test_zero_weighted(self):
        smape_obj = losses.SymetricMeanAbsolutePercentageError()
        y_true = tf.constant(
            [1, 9, 2, -5, -2, 6],
            shape=(2, 3),
        )
        y_pred = tf.constant(
            [4, 8, 12, 8, 1, 3],
            shape=(2, 3),
            dtype=tf.float32,
        )
        loss = smape_obj(y_true, y_pred, sample_weight=0)
        self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

    def test_no_reduction(self):
        smape_obj = losses.SymetricMeanAbsolutePercentageError(
             reduction=losses_utils.ReductionV2.NONE
        )
        y_true = tf.constant(
            [1, 9, 2, -5, -2, 6],
            shape=(2, 3),
        )
        y_pred = tf.constant(
            [4, 8, 12, 8, 1, 3],
            shape=(2, 3),
            dtype=tf.float32,
        )
        loss = smape_obj(y_true, y_pred, sample_weight=2.3)
        loss = self.evaluate(loss)
        self.assertArrayNear(loss, [91.5406 * 2.3, 155.5556 * 2.3], 1e-3)

    def test_sum_reduction(self):
        smape_obj = losses.SymetricMeanAbsolutePercentageError(
            reduction=losses_utils.ReductionV2.SUM
        )
        y_true = tf.constant(
            [1, 9, 2, -5, -2, 6],
            shape=(2, 3),
        )
        y_pred = tf.constant(
            [4, 8, 12, 8, 1, 3],
            shape=(2, 3),
            dtype=tf.float32,
        )
        loss = smape_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(self.evaluate(loss), np.sum([91.5406 * 2.3, 155.5556 * 2.3]), 3)

    def test_unweighted_with_mask(self):
        smape_obj = losses.SymetricMeanAbsolutePercentageError(mask=True)
        y_true = tf.constant(
            [1, 9, 0, -5, -2, 6],
            shape=(2, 3),
        )
        y_pred = tf.constant(
            [4, 8, 0, 8, 1, 3],
            shape=(2, 3),
            dtype=tf.float32
        )
        loss = smape_obj(y_true, y_pred)
        self.assertAlmostEqual(self.evaluate(loss), 155.556, 3)

    def test_ragged_tensors_with_mask(self):
        smape_obj = losses.SymetricMeanAbsolutePercentageError(mask=True)

        y_true = tf.ragged.constant(
            
                [[1.0, 0., 9.0], [2.0, 5.0]],
            
        )
        y_pred = tf.ragged.constant(
            [
                [4.0, 0., 8.0], [12.0, 3.0]
            ],
            dtype=tf.float32,
        )
        sample_weight = tf.constant([1.2, 0.5])
        loss = smape_obj(y_true, y_pred, sample_weight=sample_weight)

        self.assertAllClose(self.evaluate(loss), 81.964, 1e-2)