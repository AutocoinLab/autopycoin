# pylint: skip-file

"""
Unit test for data utils functions.
"""

import numpy as np
import pandas as pd
import pytest

import tensorflow as tf
from tensorflow import test

from .data_utils import quantiles_handler, example_handler, fill_none
from ..data import random_ts
from ..dataset import WindowGenerator


class CheckQuantileTest(test.TestCase):
    def test_nested_quantiles(self):
        quantiles = [[0.1, 0.1], [3.0, 0.7]]
        self.assertEqual([[0.1], [0.03, 0.7]], quantiles_handler(quantiles))

        quantiles = [0.1, 0.1]
        self.assertEqual([[0.1]], quantiles_handler(quantiles))

    def test_float_quantile(self):
        quantiles = 0.1
        self.assertEqual([[0.1]], quantiles_handler(quantiles))

    def test_int_quantile(self):
        quantiles = 1
        self.assertEqual([[0.01]], quantiles_handler(quantiles))

    def test_none_quantile(self):
        quantiles = [1, None]
        with self.assertRaisesRegexp(
            ValueError, "None value or empty list are not supported"
        ):
            self.assertEqual([[0.1]], quantiles_handler(quantiles))


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
        input_width=50,
        label_width=20,
        shift=20,
        test_size=10,
        valid_size=10,
        flat=True,
        batch_size=32,
    )

    request.cls.w = request.cls.w.from_array(
        data=request.cls.data, input_columns=["test"], label_columns=["test"],
    )

    request.cls.w2 = WindowGenerator(
        input_width=50,
        label_width=20,
        shift=20,
        test_size=10,
        valid_size=10,
        flat=True,
        batch_size=32,
    )

    request.cls.w2 = request.cls.w2.from_array(
        data=request.cls.data,
        input_columns=["test"],
        label_columns=["test"],
        date_columns=["test"],
    )


@pytest.mark.usefixtures("prepare_data")
class ExampleHandlerTest(test.TestCase):
    def test_shape(self):
        """
        Dataset and tuple shapes are not handled.
        """
        outputs = example_handler(self.w.train, self.w)
        self.assertEqual(
            (
                [output if output is None else output.shape for output in outputs[0]]
                + [outputs[1].shape]
            ),
            [tf.TensorShape([32, 50]), None, None, None, tf.TensorShape([32, 20]),],
        )

        outputs = example_handler(self.w.test, self.w)
        self.assertEqual(
            (
                [output if output is None else output.shape for output in outputs[0]]
                + [outputs[1].shape]
            ),
            [tf.TensorShape([10, 50]), None, None, None, tf.TensorShape([10, 20]),],
        )

        outputs = example_handler(self.w.valid, self.w)
        self.assertEqual(
            (
                [output if output is None else output.shape for output in outputs[0]]
                + [outputs[1].shape]
            ),
            [tf.TensorShape([10, 50]), None, None, None, tf.TensorShape([10, 20]),],
        )

        outputs = example_handler(self.w.production(self.data, None), self.w)
        self.assertEqual(
            (
                [output if output is None else output.shape for output in outputs[0]]
                + [outputs[1].shape]
            ),
            [tf.TensorShape([1, 50]), None, None, None, tf.TensorShape([1, 20]),],
        )

        # example_handler doesn't affect tuple inputs with good shape
        tensor = (tf.constant([0.0]), tf.constant([0.0]))
        outputs = example_handler(tensor, self.w)
        self.assertEqual(
            (
                [output if output is None else output.shape for output in outputs[0]]
                + [outputs[1].shape]
            ),
            [tf.TensorShape([1]), None, None, None, tf.TensorShape([1])],
        )

    def test_dtype(self):
        """
        Dataset dtypes are effectively turned into desired dtypes.
        tuple with good types is not handled.
        """

        outputs = example_handler(self.w2.train, self.w2)
        self.assertEqual(
            (
                [output if output is None else output.dtype for output in outputs[0]]
                + [outputs[1].dtype]
            ),
            [tf.float32, None, np.dtype("<U3"), np.dtype("<U3"), tf.float32],
        )

        outputs = example_handler(self.w2.test, self.w2)
        self.assertEqual(
            (
                [output if output is None else output.dtype for output in outputs[0]]
                + [outputs[1].dtype]
            ),
            [tf.float32, None, np.dtype("<U3"), np.dtype("<U3"), tf.float32],
        )

        outputs = example_handler(self.w2.valid, self.w2)
        self.assertEqual(
            (
                [output if output is None else output.dtype for output in outputs[0]]
                + [outputs[1].dtype]
            ),
            [tf.float32, None, np.dtype("<U3"), np.dtype("<U3"), tf.float32],
        )

        outputs = example_handler(self.w2.production(self.data, None), self.w2)
        self.assertEqual(
            (
                [output if output is None else output.dtype for output in outputs[0]]
                + [outputs[1].dtype]
            ),
            [tf.float32, None, np.dtype("<U3"), np.dtype("<U3"), tf.float32],
        )

        # example_handler doesn't affect tuple inputs with good -types
        tensor = (
            (
                tf.constant([0.0]),
                np.array(["0"], dtype="<U2"),
                np.array(["0"], dtype="<U2"),
            ),
            tf.constant([0.0]),
        )
        outputs = example_handler(tensor, self.w2)
        self.assertEqual(
            (
                [output if output is None else output.dtype for output in outputs[0]]
                + [outputs[1].dtype]
            ),
            [tf.float32, None, np.dtype("<U2"), np.dtype("<U2"), tf.float32],
        )

    def test_raise(self):
        """
        inputs shape or type is not respected.
        """

        # list and not tuple
        with self.assertRaises(ValueError):
            tensor = [tf.constant([0.0]), tf.constant([0.0])]
            example_handler(tensor, self.w)

        # bad inputs shape
        with self.assertRaises(ValueError):
            tensor = (
                (
                    tf.constant([0]),
                    tf.constant([0]),
                    tf.constant([0]),
                    tf.constant([0]),
                ),
                tf.constant([0]),
            )
            example_handler(tensor, self.w)

        # good inputs shape but bad dtypes
        with self.assertRaises(ValueError):
            tensor = (tf.constant(["0"]), tf.constant([0]))
            example_handler(tensor, self.w)

    def test_none(self):
        """Test if None value fill the input of example_handler."""

        w = WindowGenerator(
            input_width=50,
            label_width=20,
            shift=20,
            test_size=10,
            valid_size=10,
            flat=True,
            batch_size=32,
        )

        w = w.from_array(data=self.data, input_columns=["test"], label_columns=["test"])

        tensor = (tf.constant([0.0]), tf.constant([0.0]))
        np.testing.assert_equal(
            example_handler(tensor, w),
            ((tf.constant([0.0]), None, None, None), tf.constant([0.0])),
        )


@pytest.mark.usefixtures("prepare_data")
class FillNoneTest(test.TestCase):
    def test_max_value(self):
        tensor = (tf.constant([0.0]), tf.constant([0.0]))
        np.testing.assert_equal(
            fill_none(tensor, max_value=4),
            (tf.constant([0.0]), tf.constant([0.0]), None, None),
        )

        np.testing.assert_equal(
            fill_none(tensor, max_value=5),
            (tf.constant([0.0]), tf.constant([0.0]), None, None, None),
        )
