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
from .training import Model
from ..layers import Layer


class Dense(Layer):

    def build(self, input_shape):
        shape = [20, 50]
        if self.n_quantiles > 1:
            shape = [self.n_quantiles, 20, 50]

        self.dense = self.add_weight(
            shape=shape,
            name=f"fc",
        )
        super().build(input_shape)

    def call(
        self, inputs, **kwargs
    ):
        return inputs @ self.dense


@keras_parameterized.run_all_keras_modes
class ModelTest(tf.test.TestCase, parameterized.TestCase):
    """
    Unit tests for the nbeats model.
    """

    def test_attributes(self):
        inputs = tf.keras.Input(shape=(20))
        layer1 = Dense()
        output = layer1(inputs)
        model = Model(inputs=inputs, outputs=output)

        self.assertEqual(model.quantiles, None)
        model.compile(loss=QuantileLossError([0.5, 0.6]))
        self.assertEqual(model.quantiles, [0.4, 0.5, 0.6])

        for layer in model.layers[1:]: # W don't take Input layer
            self.assertEqual(layer.quantiles, [0.4, 0.5, 0.6])

    
    @parameterized.parameters(
        [
        (QuantileLossError([0.1, 0.3, 0.5]), (213, 50, 5), None),
        #([QuantileLossError([0.1, 0.3, 0.5]), 'mse'], (213, 50, 5), None),
        ([QuantileLossError([0.1, 0.3, 0.5]), QuantileLossError([0.1, 0.3, 0.5])], (213, 50, 5), None),
        ([QuantileLossError([0.1, 0.3, 0.5]), QuantileLossError([0.1, 0.5])], (213, 50, 5), ValueError),
        ('mse', (213, 50), None)
        ]
    )
    def test_predict_output_shape(self, loss, shape, error):
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
            input_width=20,
            label_width=50,
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
        )

        inputs = tf.keras.Input(shape=(20))
        output = Dense()(inputs)
        model = Model(inputs=inputs, outputs=output)

        if not error:
            model.compile(
                tf.keras.optimizers.Adam(
                    learning_rate=0.015,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-07,
                    amsgrad=False,
                    name="Adam",
                ),
                loss=loss,
                metrics=["mae"],
            )

            model.fit(w.train, validation_data=w.valid, epochs=1)
            output = model.predict(w.train)

            self.assertEqual(output.shape, shape)
        else:
            with self.assertRaisesRegexp(ValueError, f"""`quantiles` has to be identical through losses"""):
                model.compile(
                tf.keras.optimizers.Adam(
                    learning_rate=0.015,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-07,
                    amsgrad=False,
                    name="Adam",
                ),
                loss=loss,
                metrics=["mae"],
            )

