# pylint: skip-file

"""
Unit tests for generator class
"""

import pandas as pd
import numpy as np
import pytest
from absl.testing import parameterized

import tensorflow as tf
from tensorflow.python.keras import keras_parameterized
from tensorflow.keras.backend import floatx

from .generator import WindowGenerator
from ..models.nbeats import create_interpretable_nbeats
from ..losses.losses import QuantileLossError


def test_dataframe():
    columns = ["data1", "data2", "data3", "data4", "known", "year", "label"]
    return pd.DataFrame(
        [[i for i in range(j, len(columns) * 100, 100)] for j in range(100)],
        columns=columns,
    )


def test_array():
    return np.array([[i for i in range(j, 7 * 100, 100)] for j in range(100)])


def test_serie():
    return pd.Series(range(100))


@pytest.fixture(scope="class")
def prepare_generator(request):
    request.cls.df = test_dataframe()


@keras_parameterized.run_all_keras_modes
@pytest.mark.usefixtures("prepare_generator")
class TestGenerator(tf.test.TestCase, parameterized.TestCase):
    @parameterized.parameters(
        [  # Test data with all arguments assigned
            (
                2,
                3,
                3,
                3,
                2,
                True,
                2,
                ["data1"],
                ["label"],
                ["data2"],
                ["year"],
                1,
                [2, 3, 3, 3, 2, True, 2, [0], [6], [1], [5], 1],
                True,
            ),
            # Test data with their default values
            (
                2,
                3,
                3,
                3,
                2,
                True,
                1,
                ["data1"],
                ["label"],
                [],
                [],
                None,
                [2, 3, 3, 3, 2, True, 1, [0], [6], None, None, None],
                True,
            ),
            # Test data with others values
            (
                2,
                3,
                3,
                0,
                0,
                False,
                1,
                ["data1", "label"],
                ["label", "label"],
                [],
                [],
                None,
                [2, 3, 3, 0, 0, False, 1, [0, 6], [6, 6], None, None, None],
                True,
            ),
            # Test data with all arguments assigned
            (
                2,
                3,
                3,
                3,
                2,
                True,
                1,
                [0],
                [6],
                [1],
                [5],
                1,
                [2, 3, 3, 3, 2, True, 1, [0], [6], [1], [5], 1],
                False,
                test_dataframe().values,
            ),
            # Test data with their default values
            (
                2,
                3,
                3,
                3,
                2,
                True,
                1,
                [0],
                [6],
                [],
                [],
                None,
                [2, 3, 3, 3, 2, True, 1, [0], [6], None, None, None],
                False,
                test_dataframe().values,
            ),
            # Test data with others values
            (
                2,
                3,
                3,
                0,
                0,
                False,
                1,
                [0, 6],
                [6, 6],
                [],
                [],
                None,
                [2, 3, 3, 0, 0, False, 1, [0, 6], [6, 6], None, None, None],
                False,
                test_dataframe().values,
            ),
            # Test data with float valid and test size
            (
                2,
                3,
                3,
                0.25,
                0.20,
                False,
                1,
                [0, 6],
                [6, 6],
                [],
                [],
                None,
                [2, 3, 3, 0.25, 0.20, False, 1, [0, 6], [6, 6], None, None, None],
                False,
                test_dataframe().values,
            ),
            (
                2,
                3,
                3,
                0.25,
                0.20,
                False,
                1,
                None,
                None,
                None,
                None,
                None,
                [
                    2,
                    3,
                    3,
                    0.25,
                    0.20,
                    False,
                    1,
                    [0, 1, 2, 3, 4, 5, 6],
                    [0, 1, 2, 3, 4, 5, 6],
                    None,
                    None,
                    None,
                ],
                False,
                test_array(),
            ),
            (
                2,
                3,
                3,
                0.25,
                0.20,
                False,
                1,
                None,
                None,
                None,
                None,
                None,
                [2, 3, 3, 0.25, 0.20, False, 1, [0], [0], None, None, None],
                False,
                test_serie(),
            ),
            (
                2,
                3,
                3,
                0.25,
                0.20,
                False,
                1,
                None,
                [],
                None,
                None,
                None,
                [2, 3, 3, 0.25, 0.20, False, 1, [0], None, None, None, None],
                False,
                test_serie(),
            ),
        ]
    )
    def test_attributes(
        self,
        input_width,
        label_width,
        shift,
        valid_size,
        test_size,
        flat,
        sequence_stride,
        input_columns,
        label_columns,
        known_columns,
        date_columns,
        batch_size,
        expected_values,
        df,
        df_subsitute=None,
    ):
        """Attributes testing."""
        w = WindowGenerator(
            input_width=input_width,
            label_width=label_width,
            shift=shift,
            valid_size=valid_size,
            test_size=test_size,
            flat=flat,
            sequence_stride=sequence_stride,
            batch_size=batch_size,
        )

        attributes = [
            "input_width",
            "label_width",
            "shift",
            "valid_size",
            "test_size",
            "flat",
            "sequence_stride",
            "input_columns",
            "label_columns",
            "known_columns",
            "date_columns",
            "batch_size",
        ]

        if df:
            w = w.from_array(
                self.df, input_columns, label_columns, known_columns, date_columns,
            )

            for attr, expected_value in zip(attributes, expected_values):
                value = getattr(w, attr)
                if isinstance(expected_value, dict):
                    self.assertDictEqual(
                        value,
                        expected_value,
                        f"{attr} not equal to the expected value, got {value} != {expected_value}",
                    )
                else:
                    self.assertEqual(
                        value,
                        expected_value,
                        f"{attr} not equal to the expected value, got {value} != {expected_value}",
                    )
        else:
            w = w.from_array(
                df_subsitute, input_columns, label_columns, known_columns, date_columns,
            )

            for attr, expected_value in zip(attributes, expected_values):
                value = getattr(w, attr)
                if isinstance(expected_value, dict):
                    self.assertDictEqual(
                        value,
                        expected_value,
                        f"{attr} not equal to the expected value, got {value} != {expected_value}",
                    )
                else:
                    self.assertEqual(
                        value,
                        expected_value,
                        f"{attr} not equal to the expected value, got {value} != {expected_value}",
                    )

    @parameterized.parameters(
        [
            (
                -2,
                2,
                2,
                2,
                2,
                True,
                ["data1"],
                ["label"],
                ["data1"],
                ["year"],
                1,
                AssertionError,
                "The input width has to be strictly positive, got -2.",
            ),
            (
                2,
                -2,
                2,
                2,
                2,
                True,
                ["data1"],
                ["label"],
                ["data1"],
                ["year"],
                1,
                AssertionError,
                "The label width has to be strictly positive, got -2.",
            ),
            (
                2,
                2,
                -2,
                2,
                2,
                True,
                ["data1"],
                ["label"],
                ["data1"],
                ["year"],
                1,
                AssertionError,
                "The shift has to be strictly positive, got -2.",
            ),
            (
                2,
                2,
                2,
                -2,
                2,
                True,
                ["data1"],
                ["label"],
                ["data1"],
                ["year"],
                1,
                AssertionError,
                "The valid size has to be positive or null, got -2.",
            ),
            (
                2,
                2,
                2,
                2,
                -2,
                True,
                ["data1"],
                ["label"],
                ["data1"],
                ["year"],
                1,
                AssertionError,
                "The test size has to be positive or null, got -2.",
            ),
            (
                2,
                2,
                2,
                2,
                2,
                True,
                ["data1"],
                ["label"],
                ["data1"],
                ["year"],
                -1,
                AssertionError,
                "The batch size has to be strictly positive, got -1.",
            ),
            (
                2,
                2,
                2,
                2,
                2,
                True,
                ["data15"],
                ["label"],
                ["data1"],
                ["year"],
                None,
                KeyError,
                "Columns are not found inside data,",
            ),
            (
                2,
                2,
                2,
                2,
                2,
                True,
                ["data1"],
                ["data15"],
                ["data1"],
                ["year"],
                1,
                KeyError,
                "Columns are not found inside data,",
            ),
            (
                2,
                2,
                2,
                2,
                2,
                True,
                ["data1"],
                ["label"],
                ["data15"],
                ["year"],
                1,
                KeyError,
                "Columns are not found inside data,",
            ),
            (
                2,
                2,
                2,
                2,
                2,
                True,
                ["data1"],
                ["label"],
                ["data1"],
                ["data15"],
                1,
                KeyError,
                "Columns are not found inside data,",
            ),
            (
                2,
                2,
                2,
                2,
                2,
                True,
                [],
                ["data1"],
                [],
                [],
                1,
                AssertionError,
                "The input columns list is empty",
            ),
            (
                100,
                2,
                2,
                2,
                2,
                True,
                ["data1"],
                ["data1"],
                [],
                [],
                1,
                AssertionError,
                "The training dataset is empty, please redefine the test size or valid size",
            ),
            (
                2,
                100,
                2,
                2,
                2,
                True,
                ["data1"],
                ["data1"],
                [],
                [],
                1,
                AssertionError,
                "The label width has to be equal or lower than 4, got 100",
            ),
            (
                2,
                2,
                100,
                2,
                2,
                True,
                ["data1"],
                ["data1"],
                [],
                [],
                1,
                AssertionError,
                "The training dataset is empty, please redefine the test size or valid size",
            ),
            (
                2,
                2,
                2,
                1000,
                2,
                True,
                ["data1"],
                ["data1"],
                [],
                [],
                1,
                AssertionError,
                "The training dataset is empty, please redefine the test size or valid size.",
            ),
            (
                2,
                2,
                2,
                2,
                1000,
                True,
                ["data1"],
                ["data1"],
                [],
                [],
                1,
                AssertionError,
                "The training dataset is empty, please redefine the test size or valid size.",
            ),
            (
                2,
                2,
                2,
                1.0,
                0.1,
                True,
                ["data1"],
                ["data1"],
                [],
                [],
                1,
                AssertionError,
                "The training dataset is empty, please redefine the test size or valid size.",
            ),
            (
                2,
                2,
                2,
                0.1,
                1.0,
                True,
                ["data1"],
                ["data1"],
                [],
                [],
                1,
                AssertionError,
                "The training dataset is empty, please redefine the test size or valid size.",
            ),
        ]
    )
    def test_raise_error_attributes(
        self,
        input_width,
        label_width,
        shift,
        valid_size,
        test_size,
        flat,
        input_columns,
        label_columns,
        known_columns,
        date_columns,
        batch_size,
        error,
        msg_error,
    ):
        """Test attributes assertions."""
        with self.assertRaisesRegex(error, msg_error):
            w = WindowGenerator(
                input_width=input_width,
                label_width=label_width,
                shift=shift,
                valid_size=valid_size,
                test_size=test_size,
                flat=flat,
                batch_size=batch_size,
            )

            w.from_array(
                test_dataframe(),
                input_columns,
                label_columns,
                known_columns,
                date_columns,
            )

            w.train
            w.valid
            w.test

    @parameterized.parameters(
        [
            (
                2,
                2,
                2,
                4,
                2,
                True,
                ["data1"],
                ["label"],
                ["data1"],
                ["year"],
                1,
                ["train", "valid", "test", "data",],
                (1, 2),
                (1, 2),
                (1, 2),
                (1, 2),
                4,
            ),
            (
                2,
                2,
                2,
                2,
                2,
                True,
                ["data1"],
                ["label"],
                ["data1"],
                ["year"],
                1,
                ["train", "valid", "test"],
                (1, 2),
                (1, 2),
                (1, 2),
                (1, 2),
                4,
            ),
            (
                2,
                2,
                2,
                1,
                1,
                True,
                ["data1"],
                ["label"],
                ["data1"],
                ["year"],
                1,
                ["train", "valid", "test"],
                (1, 2),
                (1, 2),
                (1, 2),
                (1, 2),
                4,
            ),
            (
                2,
                2,
                2,
                1,
                2,
                True,
                ["data1"],
                [],
                ["data1"],
                ["year"],
                1,
                ["train", "valid", "test"],
                (1, 2),
                None,
                (1, 2),
                (1, 2),
                4,
            ),
            (
                2,
                2,
                2,
                2,
                2,
                True,
                ["data1"],
                ["label"],
                [],
                [],
                1,
                ["train", "valid", "test"],
                (1, 2),
                (1, 2),
                None,
                None,
                1,
            ),
            (
                2,
                2,
                2,
                2,
                2,
                False,
                ["data1"],
                ["label"],
                ["data1"],
                [],
                1,
                ["train", "valid", "test"],
                (1, 2, 1),
                (1, 2, 1),
                (1, 2, 1),
                None,
                2,
            ),
            (
                2,
                2,
                2,
                2,
                2,
                True,
                ["data1", "data1"],
                ["label"],
                ["data1"],
                [],
                1,
                ["train", "valid", "test"],
                (1, 4),
                (1, 2),
                (1, 2),
                None,
                2,
            ),
            (
                2,
                2,
                2,
                2,
                2,
                False,
                ["data1", "data1"],
                ["label"],
                ["data1"],
                [],
                1,
                ["train", "valid", "test"],
                (1, 2, 2),
                (1, 2, 1),
                (1, 2, 1),
                None,
                2,
            ),
        ]
    )
    def test_properties(
        self,
        input_width,
        label_width,
        shift,
        valid_size,
        test_size,
        flat,
        input_columns,
        label_columns,
        known_columns,
        date_columns,
        batch_size,
        properties,
        expected_input_shape,
        expected_label_shape,
        expected_known_shape,
        expected_date_shape,
        expected_x_shape,
    ):
        w = WindowGenerator(
            input_width=input_width,
            label_width=label_width,
            shift=shift,
            valid_size=valid_size,
            test_size=test_size,
            flat=flat,
            batch_size=batch_size,
        )

        w = w.from_array(
            self.df, input_columns, label_columns, known_columns, date_columns,
        )

        df = pd.DataFrame()
        for attr in properties:
            property = getattr(w, attr)
            # Test if dataset
            if attr == "data":
                self.assertIsInstance(property, np.ndarray)
                np.testing.assert_equal(w.data, self.df)
            elif property is not None:
                self.assertIsInstance(property, tf.data.Dataset)
                expected_cardinality = valid_size if "valid" in attr else test_size
                if "valid" in attr or "test" in attr:
                    self.assertEqual(
                        property.reduce(np.int64(0), lambda x, _: x + 1),
                        expected_cardinality,
                        f"Error in {attr}, cardinality not equals.",
                    )
                # Test shape, handle if columns are not defined
                if expected_label_shape:
                    x, y = iter(property).get_next()
                else:
                    x = iter(property).get_next()
                if expected_x_shape > 1:
                    self.assertIsInstance(x, tuple)
                else:
                    self.assertIsInstance(x, tf.Tensor)
                self.assertEqual(len(x), expected_x_shape)
                if expected_input_shape:
                    if expected_x_shape > 1:
                        self.assertEqual(x[0].shape, expected_input_shape)
                        self.assertDTypeEqual(x[0], floatx())
                    else:
                        self.assertEqual(x.shape, expected_input_shape)
                        self.assertDTypeEqual(x, floatx())
                if expected_date_shape:
                    self.assertEqual(x[-2].shape, expected_date_shape)
                    self.assertDTypeEqual(x[-2], "O")
                    self.assertEqual(x[-1].shape, expected_date_shape)
                    self.assertDTypeEqual(x[-1], "O")
                if expected_label_shape:
                    self.assertIsInstance(y, tf.Tensor)
                    self.assertEqual(y.shape, expected_label_shape)
                    self.assertDTypeEqual(y, floatx())
                if expected_known_shape:
                    self.assertEqual(x[1].shape, expected_known_shape)
                    self.assertDTypeEqual(x[1], floatx())

                # values
                if "train" in attr and expected_x_shape == 4 and expected_label_shape:
                    for values, col, s in zip(
                        [x[0], x[1], x[2], x[3], y],
                        [
                            w.input_columns,
                            w.known_columns,
                            w.date_columns,
                            w.date_columns,
                            w.label_columns,
                        ],
                        [
                            slice(None, 2),
                            slice(2, 4),
                            slice(None, 2),
                            slice(2, 4),
                            slice(2, 4),
                        ],
                    ):
                        # Not testing index or values type.
                        if flat == True:
                            variables = len(col)
                            values = tf.reshape(
                                values.numpy().astype(floatx()), (-1, variables)
                            )
                        elif flat == False:
                            values = values[0].numpy().astype(floatx())
                        np.testing.assert_equal(self.df.values[s, col], values)
            else:
                self.assertIsInstance(property, type(None))

    @parameterized.parameters(
        [
            (
                2,
                2,
                2,
                2,
                2,
                True,
                ["data1"],
                ["data1"],
                ["label"],
                ["year"],
                1,
                (1, 2),
                (1, 2),
                (1, 2),
                (1, 2),
            ),
            (
                2,
                2,
                2,
                2,
                2,
                True,
                ["data1"],
                ["data1"],
                ["label"],
                ["year"],
                None,
                (1, 2),
                (1, 2),
                (1, 2),
                (1, 2),
            ),
            (
                2,
                2,
                2,
                2,
                2,
                False,
                ["data1"],
                ["data1"],
                ["label"],
                ["year"],
                1,
                (1, 2, 1),
                (1, 2, 1),
                (1, 2, 1),
                (1, 2, 1),
            ),
            (
                2,
                2,
                2,
                2,
                2,
                False,
                ["data1"],
                ["data1"],
                ["label"],
                ["year"],
                None,
                (1, 2, 1),
                (1, 2, 1),
                (1, 2, 1),
                (1, 2, 1),
            ),
        ]
    )
    def test_methods(
        self,
        input_width,
        label_width,
        shift,
        valid_size,
        test_size,
        flat,
        input_columns,
        label_columns,
        known_columns,
        date_columns,
        batch_size,
        expected_input_shape,
        expected_label_shape,
        expected_known_shape,
        expected_date_shape,
    ):

        w = WindowGenerator(
            input_width=input_width,
            label_width=label_width,
            shift=shift,
            valid_size=valid_size,
            test_size=test_size,
            flat=flat,
            batch_size=batch_size,
        )

        w = w.from_array(
            self.df, input_columns, label_columns, known_columns, date_columns,
        )

        dataset = w.production(self.df, batch_size)

        # shape
        self.assertIsInstance(dataset, tf.data.Dataset)
        expected_cardinality = (
            self.df.shape[0] + 1 - input_width - shift
            if batch_size is None
            else int((self.df.shape[0] + 1 - input_width - shift) / batch_size)
        )
        self.assertEqual(
            dataset.reduce(np.int64(0), lambda x, _: x + 1).numpy(),
            expected_cardinality,
            f"Error in cardinality.",
        )

        # Test shape
        x, y = iter(dataset).get_next()
        self.assertEqual(x[0].shape, expected_input_shape)
        self.assertEqual(x[2].shape, expected_date_shape)
        self.assertEqual(x[3].shape, expected_date_shape)
        self.assertEqual(y.shape, expected_label_shape)
        if len(known_columns) == 0:
            self.assertEqual(x[1].shape, (expected_known_shape[0], 0))
        else:
            self.assertEqual(x[1].shape, expected_known_shape)

        # Test type
        self.assertDTypeEqual(x[0], floatx())
        self.assertDTypeEqual(x[1], floatx())
        self.assertDTypeEqual(x[2], "O")
        self.assertDTypeEqual(x[3], "O")
        self.assertDTypeEqual(y, floatx())

    @parameterized.parameters(
        [
            (
                2,
                2,
                2,
                2,
                2,
                True,
                ["data1"],
                ["data1"],
                ["label"],
                ["year"],
                1,
                pd.DataFrame([1], columns=["test"]),
                "The given dataframe doesn't contain enough values",
            ),
            (
                2,
                2,
                2,
                2,
                2,
                True,
                ["data1"],
                ["data1"],
                ["label"],
                ["year"],
                1,
                pd.DataFrame([[1, 1, 1]], columns=["data1", "label", "year"]),
                "The given dataframe doesn't contain enough values",
            ),
        ]
    )
    def test_raise_error_production(
        self,
        input_width,
        label_width,
        shift,
        valid_size,
        test_size,
        flat,
        input_columns,
        label_columns,
        known_columns,
        date_columns,
        batch_size,
        data,
        error,
    ):

        w = WindowGenerator(
            input_width=input_width,
            label_width=label_width,
            shift=shift,
            valid_size=valid_size,
            test_size=test_size,
            flat=flat,
            batch_size=batch_size,
        )

        w = w.from_array(
            self.df, input_columns, label_columns, known_columns, date_columns,
        )

        # Raise an error if columns of data are not inside self.data
        with self.assertRaisesRegexp(AssertionError, error):
            w.production(data, None)

    @parameterized.parameters(
        [(2, 2, 2, 2, 2, True, ["data1"], ["data1"], ["label"], ["year"], 1),]
    )
    def test_with_model(
        self,
        input_width,
        label_width,
        shift,
        valid_size,
        test_size,
        flat,
        input_columns,
        label_columns,
        known_columns,
        date_columns,
        batch_size,
    ):
        model = create_interpretable_nbeats(
            label_width=2,
            forecast_periods=[2],
            backcast_periods=[2],
            forecast_fourier_order=[2],
            backcast_fourier_order=[2],
            p_degree=1,
            trend_n_neurons=16,
            seasonality_n_neurons=16,
            drop_rate=0.0,
            share=True,
        )

        w = WindowGenerator(
            input_width=input_width,
            label_width=label_width,
            shift=shift,
            valid_size=valid_size,
            test_size=test_size,
            flat=flat,
            batch_size=batch_size,
        )

        w = w.from_array(
            self.df, input_columns, label_columns, known_columns, date_columns,
        )

        model.compile(loss=QuantileLossError(quantiles=[0.5]))
        model.fit(w.train, validation_data=w.valid)
        model.evaluate(w.test)
        prod = w.production(self.df)
        model.predict(prod)
