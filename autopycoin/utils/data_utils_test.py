# pylint: skip-file

"""
Unit test for data utils functions.
"""

import numpy as np
import pandas as pd
import pytest

import tensorflow as tf
from tensorflow import test

from .data_utils import quantiles_handler, example_handler
from ..data import random_ts
from ..dataset import WindowGenerator


class CheckQuantileTest(test.TestCase):
    def test_quantile_50(self):
        quantiles = [0.1]
        self.assertIn(0.5, quantiles_handler(quantiles))

    def test_symetry(self):
        quantiles = [0.1]
        self.assertIn(0.9, quantiles_handler(quantiles))
        quantiles = [0.9]
        self.assertIn(0.1, quantiles_handler(quantiles))

    def test_instance(self):
        quantiles = [0.1]
        self.assertIsInstance(quantiles_handler(quantiles), list)

    def test_type(self):
        quantiles = [0.1]
        self.assertDTypeEqual(quantiles_handler(quantiles), "float")


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

    request.cls.data = pd.DataFrame(data[0].numpy(), columns=["test"])

    request.cls.w = WindowGenerator(
        data=request.cls.data,
        input_width=50,
        label_width=20,
        shift=20,
        test_size=10,
        valid_size=10,
        strategy="one_shot",
        batch_size=32,
        input_columns=["test"],
        known_columns=None,
        label_columns=["test"],
        date_columns=None,
    )


@pytest.mark.usefixtures("prepare_data")
class ExampleHandlerTest(test.TestCase):
    def test_shape(self):
        """
        Dataset and tuple shapes are not handled.
        """

        outputs = example_handler(self.w.train)
        self.assertEqual(
            ([output.shape for output in outputs[0]] + [outputs[1].shape]),
            [
                tf.TensorShape([32, 50]),
                tf.TensorShape([32, 0]),
                (32, 50),
                (32, 20),
                tf.TensorShape([32, 20]),
            ],
        )

        outputs = example_handler(self.w.test)
        self.assertEqual(
            ([output.shape for output in outputs[0]] + [outputs[1].shape]),
            [
                tf.TensorShape([10, 50]),
                tf.TensorShape([10, 0]),
                (10, 50),
                (10, 20),
                tf.TensorShape([10, 20]),
            ],
        )

        outputs = example_handler(self.w.valid)
        self.assertEqual(
            ([output.shape for output in outputs[0]] + [outputs[1].shape]),
            [
                tf.TensorShape([32, 50]),
                tf.TensorShape([32, 0]),
                (32, 50),
                (32, 20),
                tf.TensorShape([32, 20]),
            ],
        )

        outputs = example_handler(self.w.forecast(self.data, None))
        self.assertEqual(
            ([output.shape for output in outputs[0]] + [outputs[1].shape]),
            [
                tf.TensorShape([331, 50]),
                tf.TensorShape([331, 0]),
                (331, 50),
                (331, 20),
                tf.TensorShape([331, 20]),
            ],
        )

        # example_handler doesn't affect tuple inputs with good shape
        tensor = (
            tf.constant([0.0]),
            tf.constant([0.0]),
            np.array(["0"], dtype="<U2"),
            np.array(["0"], dtype="<U2"),
        ), tf.constant([0.0])
        outputs = example_handler(tensor)
        self.assertEqual(
            ([output.shape for output in outputs[0]] + [outputs[1].shape]),
            [tens.shape for tens in tensor[0]] + [tensor[1].shape],
        )

    def test_dtype(self):
        """
        Dataset dtypes are effectively turned into desired dtypes.
        tuple with good types is not handled.
        """

        outputs = example_handler(self.w.train)
        self.assertEqual(
            ([output.dtype for output in outputs[0]] + [outputs[1].dtype]),
            [tf.float32, tf.float32, np.dtype("<U2"), np.dtype("<U3"), tf.float32],
        )

        outputs = example_handler(self.w.test)
        self.assertEqual(
            ([output.dtype for output in outputs[0]] + [outputs[1].dtype]),
            [tf.float32, tf.float32, np.dtype("<U3"), np.dtype("<U3"), tf.float32],
        )

        outputs = example_handler(self.w.valid)
        self.assertEqual(
            ([output.dtype for output in outputs[0]] + [outputs[1].dtype]),
            [tf.float32, tf.float32, np.dtype("<U3"), np.dtype("<U3"), tf.float32],
        )

        outputs = example_handler(self.w.forecast(self.data, None))
        self.assertEqual(
            ([output.dtype for output in outputs[0]] + [outputs[1].dtype]),
            [tf.float32, tf.float32, np.dtype("<U3"), np.dtype("<U3"), tf.float32],
        )

        # example_handler doesn't affect tuple inputs with good -types
        tensor = (
            tf.constant([0.0]),
            tf.constant([0.0]),
            np.array(["0"], dtype="<U2"),
            np.array(["0"], dtype="<U2"),
        ), tf.constant([0.0])
        outputs = example_handler(tensor)
        self.assertEqual(
            ([output.dtype for output in outputs[0]] + [outputs[1].dtype]),
            [tens.dtype for tens in tensor[0]] + [tensor[1].dtype],
        )

    def test_raise(self):
        """
        inputs shape or type is not respected.
        """

        # wrong shape
        with self.assertRaises(ValueError):
            tensor = (tf.constant([0]), tf.constant([0]))
            example_handler(tensor)

        # wrong instance
        with self.assertRaises(ValueError):
            tensor = [tf.constant([0]), tf.constant([0])]
            example_handler(tensor)

        # good shape and instance but wrong dtypes
        with self.assertRaises(ValueError):
            tensor = (
                tf.constant([0]),
                tf.constant([0]),
                tf.constant([0]),
                tf.constant([0]),
            ), tf.constant([0])
            example_handler(tensor)
