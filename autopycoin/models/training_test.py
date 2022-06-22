# pylint: skip-file

"""
Unit test for training.
"""

from absl.testing import parameterized
import pandas as pd
import pytest

import tensorflow as tf
from tensorflow.python.keras import keras_parameterized

from ..test_utils import model_test
from ..data.generate import random_ts
from ..dataset.generator import WindowGenerator
from ..losses.losses import QuantileLossError
from .training import UnivariateModel


class DenseModel(UnivariateModel):
    def __init__(
        self, *args: list, **kwargs: dict
    ):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):

        shape = self.additional_shape + [20, 50]

        self.dense = self.add_weight(shape=shape, name=f"fc",)

        self.dropout = tf.keras.layers.Dropout(0.0)

        super().build(input_shape)

    def call(self, inputs, **kwargs):

        return tf.matmul(inputs, self.dropout(self.dense))


class DoubleDenseModel(DenseModel):
    def __init__(
        self, *args: list, **kwargs: dict
    ):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):

        shape = self.additional_shape + [20, 50]

        self.dense2 = self.add_weight(shape=shape, name=f"fc",)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.matmul(inputs, self.dense), tf.matmul(inputs, self.dense2)


class DoubleDenseModel2(DoubleDenseModel):
    def __init__(
        self, *args: list, **kwargs: dict
    ):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):

        self.dense2 = self.add_weight(shape=[20, 50], name=f"fc",)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.matmul(inputs, self.dense), tf.matmul(inputs, self.dense2)


class DenseUnivariateModel(UnivariateModel):
    def __init__(
        self, *args: list, **kwargs: dict
    ):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):

        shape = self.additional_shape + [20, 50]

        self.dense = self.add_weight(shape=shape, name=f"fc",)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.matmul(inputs, self.dense)


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
        data=data, input_columns=["test"], label_columns=["test"],
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
        data=data, input_columns=["test", "test"], label_columns=["test", "test"],
    )


@keras_parameterized.run_all_keras_modes
@pytest.mark.usefixtures("prepare_data")
class ModelTest(tf.test.TestCase, parameterized.TestCase):
    """
    Unit tests for the nbeats model.
    """

    def test_attributes(self):
        model = DenseModel(quantiles=[0.5])

        self.assertEqual(model.quantiles, [0.5])
        model.compile(loss=QuantileLossError())

        for layer in model.layers[1:]:  # don't take Input layer
            self.assertEqual(layer.quantiles, [0.5, 0.6])

    @parameterized.parameters(
        [
            (
                DenseModel,
                [QuantileLossError(), "mse"],
                [0.2, 0.5, 0.8],
                ((213, 50, 3), (213, 50, 3)),
                None,
                None,
            ),
            (DenseModel, "mse", None, ((213, 50), (213, 50)), None, None),
            (
                DoubleDenseModel,
                [QuantileLossError(), "mse"],
                [0.2, 0.5, 0.8],
                ((213, 50, 3), (213, 50, 3)),
                None,
                None,
            ),
            (DoubleDenseModel, "mse", None, ((213, 50), (213, 50)), None, None),
            (
                DoubleDenseModel2,
                [QuantileLossError(), "mse"],
                [0.2, 0.5, 0.8],
                ((213, 50, 3), (213, 50, 3)),
                None,
                None,
            ),
            (DoubleDenseModel2, "mse", None, ((213, 50), (213, 50)), None, None),
        ]
    )
    def test_predict_output_shape(self, model, loss, quantiles, shape, error, msg):

        if not error:
            model_test(
                model,
                shape,
                loss,
                self.w_uni.train,
                self.w_uni.valid,
                expected_output_shape=(None, 50),
                kwargs={'quantiles': quantiles}
            )
        else:
            with self.assertRaisesRegexp(error, msg):
                model_test(
                    model,
                    shape,
                    loss,
                    self.w_uni.train,
                    self.w_uni.valid,
                    expected_output_shape=(None, 50),
                    kwargs={'quantiles': quantiles}
                )

    @parameterized.parameters(
        [
            (QuantileLossError(), [0.2, 0.5, 0.8], (213, 50, 2, 3), None, None),
            ([QuantileLossError()], [0.2, 0.5, 0.8], (213, 50, 2, 3), None, None),
            ([QuantileLossError(), "mse"], [0.2, 0.5, 0.8], (213, 50, 2, 3), None, None),
            (
                [
                    QuantileLossError(),
                    QuantileLossError(),
                ],
                [0.2, 0.5, 0.8],
                (213, 50, 2, 3),
                None,
                None,
            ),
            ("mse", None, (213, 50, 2), None, None),
            (
                [
                    QuantileLossError(),
                    "mse",
                    QuantileLossError(),
                    "mse",
                ],
                [0.2, 0.5, 0.8],
                (213, 50, 2, 3),
                None,
                None,
            ),
        ]
    )
    def test_predict_one_output_multivariate_shape(self, loss, quantiles, shape, error, msg):
        if not error:
            model_test(
                DenseUnivariateModel,
                (shape,),
                loss,
                self.w_multi.train,
                self.w_multi.valid,
                expected_output_shape=(None, 50, 2),
                kwargs={'quantiles': quantiles}
            )
        else:
            with self.assertRaisesRegexp(error, msg):
                model_test(
                    DenseUnivariateModel,
                    (shape,),
                    loss,
                    self.w_multi.train,
                    self.w_multi.valid,
                    expected_output_shape=(None, 50, 2),
                    kwargs={'quantiles': quantiles}
                )
