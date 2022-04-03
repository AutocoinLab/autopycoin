# pylint: skip-file

"""
Unit test for training.
"""

from absl.testing import parameterized
from numpy import quantile
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
        self, apply_multivariate_transpose: bool = True, *args: list, **kwargs: dict
    ):
        super().__init__(apply_multivariate_transpose, *args, **kwargs)

    def build(self, input_shape):

        shape = self.get_additional_shapes(0) + [20, 50]

        self.dense = self.add_weight(shape=shape, name=f"fc",)

        self.dropout = tf.keras.layers.Dropout(0.0)

        super().build(input_shape)

    def call(self, inputs, **kwargs):

        return tf.matmul(inputs, self.dropout(self.dense))


class DoubleDenseModel(DenseModel):
    def __init__(
        self, apply_multivariate_transpose: bool = True, *args: list, **kwargs: dict
    ):
        super().__init__(apply_multivariate_transpose, *args, **kwargs)

    def build(self, input_shape):

        shape = self.get_additional_shapes(1) + [20, 50]

        self.dense2 = self.add_weight(shape=shape, name=f"fc",)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.matmul(inputs, self.dense), tf.matmul(inputs, self.dense2)


class DoubleDenseModel2(DoubleDenseModel):
    def __init__(
        self, apply_multivariate_transpose: bool = True, *args: list, **kwargs: dict
    ):
        super().__init__(apply_multivariate_transpose, *args, **kwargs)

    def build(self, input_shape):

        self.dense2 = self.add_weight(shape=[20, 50], name=f"fc",)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.matmul(inputs, self.dense), tf.matmul(inputs, self.dense2)


class DenseUnivariateModel(UnivariateModel):
    def __init__(
        self, apply_multivariate_transpose: bool = True, *args: list, **kwargs: dict
    ):
        super().__init__(apply_multivariate_transpose, *args, **kwargs)

    def build(self, input_shape):

        shape = self.get_additional_shapes(0) + [20, 50]

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
        model = DenseModel()

        self.assertEqual(model.quantiles, None)
        model.compile(loss=QuantileLossError([0.5, 0.6]))
        self.assertEqual(model.quantiles, [[0.5, 0.6]])

        for layer in model.layers[1:]:  # don't take Input layer
            self.assertEqual(layer.quantiles, [[0.5, 0.6]])

    @parameterized.parameters(
        [
            (QuantileLossError([0.1, 0.3, 0.5]), (213, 50, 3), None, None),
            ([QuantileLossError([0.1, 0.3, 0.5])], (213, 50, 3), None, None),
            ([QuantileLossError([0.1, 0.3, 0.5]), "mse"], (213, 50, 3), None, None),
            (
                [
                    QuantileLossError([0.1, 0.3, 0.5]),
                    QuantileLossError([0.1, 0.3, 0.5]),
                ],
                (213, 50, 3),
                None,
                None,
            ),
            (
                [QuantileLossError([0.1, 0.3, 0.5]), QuantileLossError([0.1, 0.5])],
                (213, 50, 3),
                ValueError,
                "`quantiles` has to be identical through losses",
            ),
            ("mse", (213, 50), None, None),
        ]
    )
    def test_predict_one_output_shape(self, loss, shape, error, msg):

        if not error:
            model_test(
                DenseModel,
                (shape,),
                loss,
                self.w_uni.train,
                self.w_uni.valid,
                expected_output_shape=(None, 50),
            )
        else:
            with self.assertRaisesRegexp(error, msg):
                model_test(
                    DenseModel,
                    (shape,),
                    loss,
                    self.w_uni.train,
                    self.w_uni.valid,
                    expected_output_shape=(None, 50),
                )

    @parameterized.parameters(
        [
            (
                QuantileLossError([0.1, 0.3, 0.5]),
                ((213, 50, 3), (213, 50, 3)),
                ValueError,
                "Quantiles in losses and outputs are not the same",
            ),
            (
                [QuantileLossError([0.1, 0.3, 0.5])],
                ((213, 50, 3), (213, 50, 3)),
                ValueError,
                "Quantiles in losses and outputs are not the same",
            ),
            (
                [QuantileLossError([0.1, 0.3, 0.5]), "mse"],
                ((213, 50, 3), (213, 50)),
                None,
                None,
            ),
            (
                [
                    QuantileLossError([0.1, 0.3, 0.5]),
                    QuantileLossError([0.1, 0.3, 0.5]),
                ],
                ((213, 50, 3), (213, 50, 3)),
                None,
                None,
            ),
            (
                [QuantileLossError([0.1, 0.3, 0.5]), QuantileLossError([0.1, 0.5])],
                ((213, 50, 3), (213, 50, 2)),
                None,
                None,
            ),
            ("mse", ((213, 50), (213, 50)), None, None),
            (
                [
                    QuantileLossError([0.1, 0.3, 0.5]),
                    "mse",
                    QuantileLossError([0.1, 0.3, 0.5]),
                    "mse",
                ],
                ((213, 50, 3), (213, 50, 3)),
                None,
                None,
            ),
        ]
    )
    def test_predict_multi_output_shape(self, loss, shape, error, msg):

        if not error:
            model_test(
                DoubleDenseModel,
                shape,
                loss,
                self.w_uni.train,
                self.w_uni.valid,
                expected_output_shape=(None, 50),
            )
        else:
            with self.assertRaisesRegexp(error, msg):
                model_test(
                    DoubleDenseModel,
                    shape,
                    loss,
                    self.w_uni.train,
                    self.w_uni.valid,
                    expected_output_shape=(None, 50),
                )

    @parameterized.parameters(
        [
            (
                QuantileLossError([0.1, 0.3, 0.5]),
                ((213, 50, 3), (213, 50, 3)),
                ValueError,
                "Quantiles in losses and outputs are not the same.",
            ),
            (
                [QuantileLossError([0.1, 0.3, 0.5]), "mse"],
                ((213, 50, 3), (213, 50)),
                None,
                None,
            ),
            ("mse", ((213, 50), (213, 50)), None, None),
        ]
    )
    def test_predict_multi_output2_shape(self, loss, shape, error, msg):

        if not error:
            model_test(
                DoubleDenseModel2,
                shape,
                loss,
                self.w_uni.train,
                self.w_uni.valid,
                expected_output_shape=(None, 50),
            )
        else:
            with self.assertRaisesRegexp(error, msg):
                model_test(
                    DoubleDenseModel2,
                    shape,
                    loss,
                    self.w_uni.train,
                    self.w_uni.valid,
                    expected_output_shape=(None, 50),
                )

    @parameterized.parameters(
        [
            (QuantileLossError([0.1, 0.3, 0.5]), (213, 50, 2, 3), None, None),
            ([QuantileLossError([0.1, 0.3, 0.5])], (213, 50, 2, 3), None, None),
            ([QuantileLossError([0.1, 0.3, 0.5]), "mse"], (213, 50, 2, 3), None, None),
            (
                [
                    QuantileLossError([0.1, 0.3, 0.5]),
                    QuantileLossError([0.1, 0.3, 0.5]),
                ],
                (213, 50, 2, 3),
                None,
                None,
            ),
            (
                [QuantileLossError([0.1, 0.3, 0.5]), QuantileLossError([0.1, 0.5])],
                (213, 50, 2, 3),
                ValueError,
                "`quantiles` has to be identical through losses",
            ),
            ("mse", (213, 50, 2), None, None),
            (
                [
                    QuantileLossError([0.1, 0.3, 0.5]),
                    "mse",
                    QuantileLossError([0.1, 0.3, 0.5]),
                    "mse",
                ],
                (213, 50, 2, 3),
                None,
                None,
            ),
        ]
    )
    def test_predict_one_output_multivariate_shape(self, loss, shape, error, msg):
        if not error:
            model_test(
                DenseUnivariateModel,
                (shape,),
                loss,
                self.w_multi.train,
                self.w_multi.valid,
                expected_output_shape=(None, 50, 2),
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
                )
