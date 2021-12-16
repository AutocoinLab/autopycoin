# pylint: skip-file

"""
Unit test for training.
"""

from absl.testing import parameterized

import tensorflow as tf
from tensorflow.python.keras import keras_parameterized

from ..models import create_interpretable_nbeats
from ..losses.losses import QuantileLossError


@keras_parameterized.run_all_keras_modes
class ModelTest(tf.test.TestCase, parameterized.TestCase):
    """
    Unit tests for the nbeats model.
    """

    def test_attributes(self):
        model = create_interpretable_nbeats(
            label_width=10,
            forecast_periods=[10],
            backcast_periods=[10],
            forecast_fourier_order=[10],
            backcast_fourier_order=[10],
            p_degree=2,
            trend_n_neurons=5,
            seasonality_n_neurons=5,
            drop_rate=0.5,
            share=True,
        )

        self.assertEqual(model.quantiles, None)
        model.compile(loss=QuantileLossError([0.5, 0.6]))
        self.assertEqual(model.quantiles, [0.4, 0.5, 0.6])

        for block in model.stacks[0].blocks:
            self.assertEqual(block.quantiles, [0.4, 0.5, 0.6])
            self.assertEqual(block._n_quantiles, 3)
