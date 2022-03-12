# pylint: skip-file

"""
Unit test for training.
"""

from absl.testing import parameterized
import pandas as pd
import pytest

import tensorflow as tf
from tensorflow.python.keras import keras_parameterized


from ..data.generate import random_ts
from ..dataset.generator import WindowGenerator
from ..losses.losses import QuantileLossError
from .training import UnivariateModel


class DenseModel(UnivariateModel):

    def build(self, input_shape):

        shape = [20, 50]

        if self.has_quantiles:
            shape = [self.n_quantiles] + shape

        self.dense = self.add_weight(
            shape=shape,
            name=f"fc",
        )

        super().build(input_shape)

    def call(
        self, inputs, **kwargs
    ):
        return inputs @ self.dense

class DoubleDenseModel(DenseModel):
    def call(
        self, inputs, **kwargs
    ):
        return inputs @ self.dense, inputs @ self.dense

class DenseUnivariateModel(UnivariateModel):

    def build(self, input_shape):

        self.init_quantile_build(input_shape)
        self.init_univariate_build(input_shape)

        shape = [20, 50]

        if self.is_multivariate:
            shape = [self.n_variates] + shape

        if self.has_quantiles:
            shape = [self.n_quantiles] + shape

        self.dense = self.add_weight(
            shape=shape,
            name=f"fc",
        )

        super().build(input_shape)

    def call(
        self, inputs, **kwargs
    ):
        inputs = super().call(inputs)
        return inputs @ self.dense

    def compute_output_shape(input_shape):
        
        input_shape = tf.TensorShape(input_shape)

        self.init_univariate_build(input_shape)

        if self.n_variates

        return [
            tf.TensorShape(()),
            input_shape,
        ]


@pytest.fixture(scope="class")
def prepare_data(request):

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

    request.cls.w_uni = w.from_array(
        data=data,
        input_columns=["test"],
        label_columns=["test"],
    )

    w = WindowGenerator(
        input_width=20,
        label_width=50,
        shift=50,
        test_size=10,
        valid_size=10,
        flat=False,
        batch_size=32,
    )

    request.cls.w_multi = w.from_array(
        data=data,
        input_columns=["test", "test"],
        label_columns=["test", "test"],
    )


@keras_parameterized.run_all_keras_modes
@pytest.mark.usefixtures("prepare_data")
class ModelTest(tf.test.TestCase, parameterized.TestCase):
    """
    Unit tests for the nbeats model.
    """

    def test_attributes(self):
        model = DenseModel()

        self.assertEqual(model.quantiles, None)
        model.compile(loss=QuantileLossError([0.5, 0.6]))
        self.assertEqual(model.quantiles, [0.4, 0.5, 0.6])

        for layer in model.layers[1:]: # don't take Input layer
            self.assertEqual(layer.quantiles, [0.4, 0.5, 0.6])

    
    @parameterized.parameters(
        [
        (QuantileLossError([0.1, 0.3, 0.5]), (213, 50, 5), None),
        ([QuantileLossError([0.1, 0.3, 0.5])], (213, 50, 5), None),
        ([QuantileLossError([0.1, 0.3, 0.5]), 'mse'], (213, 50, 5), None),
        ([QuantileLossError([0.1, 0.3, 0.5]), QuantileLossError([0.1, 0.3, 0.5])], (213, 50, 5), None),
        ([QuantileLossError([0.1, 0.3, 0.5]), QuantileLossError([0.1, 0.5])], (213, 50, 5), ValueError),
        ('mse', (213, 50), None)
        ]
    )
    def test_predict_one_output_shape(self, loss, shape, error):

        model = DenseModel()

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


            model.fit(self.w_uni.train, validation_data=self.w_uni.valid, epochs=1)
            output = model.predict(self.w_uni.train)

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

    @parameterized.parameters(
        [
        (QuantileLossError([0.1, 0.3, 0.5]), ((213, 50, 5), (213, 50, 5)), None),
        ([QuantileLossError([0.1, 0.3, 0.5])], ((213, 50, 5), (213, 50, 5)), None),
        ([QuantileLossError([0.1, 0.3, 0.5]), 'mse'], ((213, 50, 5), (213, 50)), None),
        ([QuantileLossError([0.1, 0.3, 0.5]), QuantileLossError([0.1, 0.3, 0.5])], ((213, 50, 5), (213, 50, 5)), None),
        ([QuantileLossError([0.1, 0.3, 0.5]), QuantileLossError([0.1, 0.5])], ((213, 50, 5), (213, 50, 3)), ValueError),
        ('mse', ((213, 50), (213, 50)), None),
        ([QuantileLossError([0.1, 0.3, 0.5]), 'mse', QuantileLossError([0.1, 0.3, 0.5]), 'mse'], ((213, 50, 5), (213, 50, 5)), None),
        ]
    )
    def test_predict_multi_output_shape(self, loss, shape, error):

        model = DoubleDenseModel()

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

            model.fit(self.w_uni.train, validation_data=self.w_uni.valid, epochs=1)
            output = model.predict(self.w_uni.train)

            for o, s in zip(output, shape):
                self.assertEqual(o.shape, s)


    @parameterized.parameters(
        [
            (QuantileLossError([0.1, 0.3, 0.5]), ((213, 50, 5, 2), (213, 50, 5, 2)), None),
            ([QuantileLossError([0.1, 0.3, 0.5])], ((213, 50, 5, 2), (213, 50, 5, 2)), None),
            ([QuantileLossError([0.1, 0.3, 0.5]), 'mse'], ((213, 50, 5, 2), (213, 50, 2)), None),
            ([QuantileLossError([0.1, 0.3, 0.5]), QuantileLossError([0.1, 0.3, 0.5])], ((213, 50, 5, 2), (213, 50, 5, 2)), None),
            ([QuantileLossError([0.1, 0.3, 0.5]), QuantileLossError([0.1, 0.5])], ((213, 50, 5, 2), (213, 50, 3, 2)), ValueError),
            ('mse', ((213, 50, 2), (213, 5, 20)), None),
            ([QuantileLossError([0.1, 0.3, 0.5]), 'mse', QuantileLossError([0.1, 0.3, 0.5]), 'mse'], ((213, 50, 5, 2), (213, 50, 5, 2)), None)
        ]
    )
    def test_predict_one_output_multivariate_shape(self, loss, shape, error):

        model = DenseUnivariateModel()

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

            print('inputs nnn', self.w_multi.train)
            model.fit(self.w_multi.train, validation_data=self.w_multi.valid, epochs=1)
            output = model.predict(self.w_multi.train)

            for o, s in zip(output, shape):
                self.assertEqual(o.shape, s)

