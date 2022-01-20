# pylint: skip-file

"""
Unit test for training.
"""

from absl.testing import parameterized
import pandas as pd

import tensorflow as tf
from autopycoin.data.generate import random_ts
from tensorflow.python.keras import keras_parameterized

from . import create_interpretable_nbeats
from ..dataset.generator import WindowGenerator
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

        for stack in model.stacks:
            self.assertEqual(stack.quantiles, [0.4, 0.5, 0.6])

        for block in model.stacks[0].blocks:
            self.assertEqual(block.quantiles, [0.4, 0.5, 0.6])

    def test_predict_output_shape(self):
        data = random_ts(
            n_steps=400,
            trend_degree=2,
            periods=[10],
            fourier_orders=[10],
            trend_mean=0,
            trend_std=1,
            seasonality_mean=0,
            seasonality_std=1,
            batch_size=1,
            n_variables=1,
            noise=True,
            seed=42,
        )

        data = pd.DataFrame(data[0].numpy(), columns=["test"])
        data["date"] = range(400)

        w = WindowGenerator(
            label_width=50,
            input_width=20,
            shift=50,
            test_size=10,
            valid_size=10,
            flat=True,
            batch_size=32,
        )

        w = w.from_array(
            data=data,
            input_columns=["test"],
            label_columns=["test"],
            date_columns=["date"],
        )

        model = create_interpretable_nbeats(
            label_width=50,
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

        model.compile(
            tf.keras.optimizers.Adam(
                learning_rate=0.015,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07,
                amsgrad=False,
                name="Adam",
            ),
            loss=QuantileLossError([0.1, 0.3, 0.5]),
            metrics=["mae"],
        )

        model.fit(w.train, validation_data=w.valid, epochs=1)
        output = model.predict(w.train)

        self.assertEqual(output.shape, (213, 50, 5))
