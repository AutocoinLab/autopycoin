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

from .generator import WindowGenerator, STRATEGIES
from ..models.nbeats import create_interpretable_nbeats
from ..losses.losses import QuantileLossError


def test_dataframe():
    columns = ["data1", "data2", "data3", "data4", "known", "year", "label"]
    return pd.DataFrame(
        [[i for i in range(j, len(columns) * 100, 100)] for j in range(100)],
        columns=columns,
    )


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
                "one_shot",
                ["data1"],
                ["label"],
                ["data2"],
                ["year"],
                1,
                [2, 3, 3, 3, 2, "one_shot", [0], [6], [1], [5], 1],
                True,
            ),
            # Test data with their default values
            (
                2,
                3,
                3,
                3,
                2,
                "one_shot",
                ["data1"],
                ["label"],
                [],
                [],
                None,
                [2, 3, 3, 3, 2, "one_shot", [0], [6], [], [7], None],
                True,
            ),
            # Test data with others values
            (
                2,
                3,
                3,
                0,
                0,
                "auto_regressive",
                ["data1", "label"],
                ["label", "label"],
                [],
                [],
                None,
                [2, 3, 3, 0, 0, "auto_regressive", [0, 6], [6, 6], [], [7], None],
                True,
            ),
            # Test data with all arguments assigned
            (
                2,
                3,
                3,
                3,
                2,
                "one_shot",
                [0],
                [6],
                [1],
                [5],
                1,
                [2, 3, 3, 3, 2, "one_shot", [0], [6], [1], [5], 1],
                False,
            ),
            # Test data with their default values
            (
                2,
                3,
                3,
                3,
                2,
                "one_shot",
                [0],
                [6],
                [],
                [],
                None,
                [2, 3, 3, 3, 2, "one_shot", [0], [6], [], [7], None],
                False,
            ),
            # Test data with others values
            (
                2,
                3,
                3,
                0,
                0,
                "auto_regressive",
                [0, 6],
                [6, 6],
                [],
                [],
                None,
                [2, 3, 3, 0, 0, "auto_regressive", [0, 6], [6, 6], [], [7], None],
                False,
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
        strategy,
        input_columns,
        label_columns,
        known_columns,
        date_columns,
        batch_size,
        expected_values,
        df,
    ):
        """Attributes testing."""
        w = WindowGenerator(
            input_width,
            label_width,
            shift,
            valid_size,
            test_size,
            strategy,
            batch_size,
        )

        attributes = [
            "input_width",
            "label_width",
            "shift",
            "valid_size",
            "test_size",
            "strategy",
            "input_columns",
            "label_columns",
            "known_columns",
            "date_columns",
            "batch_size",
        ]

        if df:
            w = w.from_dataframe(
                self.df,
                input_columns,
                label_columns,
                known_columns,
                date_columns,
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
                test_dataframe().values,
                input_columns,
                label_columns,
                known_columns,
                date_columns,
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
                "one_shot",
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
                "one_shot",
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
                "one_shot",
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
                "one_shot",
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
                "one_shot",
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
                "regressive",
                ["data1"],
                ["label"],
                ["data1"],
                ["year"],
                1,
                AssertionError,
                "Invalid strategy",
            ),
            (
                2,
                2,
                2,
                2,
                2,
                "one_shot",
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
                "one_shot",
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
                "one_shot",
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
                "one_shot",
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
                "one_shot",
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
                "one_shot",
                [],
                ["data1"],
                [],
                [],
                1,
                AssertionError,
                "The input columns list is empty",
            ),
            (
                2,
                2,
                2,
                2,
                2,
                "one_shot",
                ["data1"],
                [],
                [],
                [],
                1,
                AssertionError,
                "The label columns list is empty",
            ),
            (
                100,
                2,
                2,
                2,
                2,
                "one_shot",
                ["data1"],
                ["data1"],
                [],
                [],
                1,
                AssertionError,
                "The input width and shift has to be equal or lower than",
            ),
            (
                2,
                100,
                2,
                2,
                2,
                "one_shot",
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
                "one_shot",
                ["data1"],
                ["data1"],
                [],
                [],
                1,
                AssertionError,
                "The input width and shift has to be equal or lower than",
            ),
            (
                2,
                2,
                2,
                1000,
                2,
                "one_shot",
                ["data1"],
                ["data1"],
                [],
                [],
                1,
                AssertionError,
                "The training dataframe is empty, please redefine the test size or valid size.",
            ),
            (
                2,
                2,
                2,
                2,
                1000,
                "one_shot",
                ["data1"],
                ["data1"],
                [],
                [],
                1,
                AssertionError,
                "The training dataframe is empty, please redefine the test size or valid size.",
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
        strategy,
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
                input_width,
                label_width,
                shift,
                valid_size,
                test_size,
                strategy,
                batch_size,
            )

            w.from_dataframe(
                test_dataframe(),
                input_columns,
                label_columns,
                known_columns,
                date_columns,
            )

    @parameterized.parameters(
        [
            (
                2,
                2,
                2,
                4,
                2,
                "one_shot",
                ["data1"],
                ["data1"],
                ["label"],
                ["year"],
                1,
                [
                    "train_data",
                    "valid_data",
                    "test_data",
                    "train",
                    "valid",
                    "test",
                    "data",
                ],
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
                0,
                "one_shot",
                ["data1"],
                ["data1"],
                ["label"],
                ["year"],
                1,
                ["train", "valid", "test"],
                (1, 2),
                (1, 2),
                (1, 2),
                (1, 2),
            ),
            (
                2,
                2,
                2,
                0,
                0,
                "one_shot",
                ["data1"],
                ["data1"],
                ["label"],
                ["year"],
                1,
                ["train", "valid", "test"],
                (1, 2),
                (1, 2),
                (1, 2),
                (1, 2),
            ),
            (
                2,
                2,
                2,
                0,
                2,
                "one_shot",
                ["data1"],
                ["data1"],
                ["label"],
                ["year"],
                1,
                ["train", "valid", "test"],
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
                "one_shot",
                ["data1"],
                ["data1"],
                ["label"],
                [],
                1,
                ["train", "valid", "test"],
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
                "auto_regressive",
                ["data1"],
                ["data1"],
                ["label"],
                [],
                1,
                ["train", "valid", "test"],
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
                "one_shot",
                ["data1", "data1"],
                ["data1"],
                ["label"],
                [],
                1,
                ["train", "valid", "test"],
                (1, 4),
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
                "auto_regressive",
                ["data1", "data1"],
                ["data1"],
                ["label"],
                [],
                1,
                ["train", "valid", "test"],
                (1, 2, 2),
                (1, 2, 1),
                (1, 2, 1),
                (1, 2, 1),
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
        strategy,
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
    ):
        w = WindowGenerator(
            input_width,
            label_width,
            shift,
            valid_size,
            test_size,
            strategy,
            batch_size,
        )

        w = w.from_dataframe(
            self.df,
            input_columns,
            label_columns,
            known_columns,
            date_columns,
        )

        df = pd.DataFrame()
        for attr in properties:
            property = getattr(w, attr)
            # Test if dataset
            if attr == "data":
                self.assertIsInstance(property, np.ndarray)
                np.testing.assert_equal(w.data, self.df)
            elif "_data" in attr:
                self.assertIsInstance(property, np.ndarray)
                if np.size(df):
                    df = np.concatenate((df, property))
                else:
                    df = property
                if "test" in attr:
                    self.assertEqual(
                        df.shape[0],
                        self.df.shape[0] + 2 * input_width,
                        f"array duplicates",
                    )
                    # shapes
                    self.assertEqual(
                        property.shape,
                        (test_size - 1 + (input_width + shift), 7),
                        "error in test_data shape",
                    )
                if "valid" in attr:
                    self.assertEqual(
                        property.shape,
                        (valid_size - 1 + (input_width + shift), 7),
                        "error in valid_data shape",
                    )
                # Types
                self.assertAllInSet(property.dtype, ["int64"])
            elif property:
                self.assertIsInstance(property, tf.data.Dataset)
                expected_cardinality = valid_size if "valid" in attr else test_size
                if "valid" in attr or "test" in attr:
                    self.assertEqual(
                        property.cardinality(),
                        expected_cardinality,
                        f"Error in {attr}, cardinality not equals.",
                    )
                # Test shape
                x, y = iter(property).get_next()
                self.assertIsInstance(x, tuple)
                self.assertEqual(len(x), 4)
                self.assertIsInstance(y, tf.Tensor)
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

                # values
                if "train" in attr:
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
                        if strategy == "one_shot":
                            variables = len(col)
                            values = tf.reshape(
                                values.numpy().astype(floatx()), (-1, variables)
                            )
                        elif strategy == "auto_regressive":
                            values = values[0].numpy().astype(floatx())
                        np.testing.assert_equal(w.train_data[s, col], values)
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
                "one_shot",
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
                "one_shot",
                ["data1"],
                ["data1"],
                ["label"],
                ["year"],
                None,
                (97, 2),
                (97, 2),
                (97, 2),
                (97, 2),
            ),
            (
                2,
                2,
                2,
                2,
                2,
                "auto_regressive",
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
                "auto_regressive",
                ["data1"],
                ["data1"],
                ["label"],
                ["year"],
                None,
                (97, 2, 1),
                (97, 2, 1),
                (97, 2, 1),
                (97, 2, 1),
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
        strategy,
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
            input_width,
            label_width,
            shift,
            valid_size,
            test_size,
            strategy,
            batch_size,
        )

        w = w.from_dataframe(
            self.df,
            input_columns,
            label_columns,
            known_columns,
            date_columns,
        )

        dataset = w.production(self.df, batch_size)

        # shape
        self.assertIsInstance(dataset, tf.data.Dataset)
        expected_cardinality = (
            1
            if batch_size is None
            else int((self.df.shape[0] + 1 - input_width - shift) / batch_size)
        )
        self.assertEqual(
            dataset.cardinality(), expected_cardinality, f"Error in cardinality."
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
                "one_shot",
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
                "one_shot",
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
        strategy,
        input_columns,
        label_columns,
        known_columns,
        date_columns,
        batch_size,
        data,
        error,
    ):

        w = WindowGenerator(
            input_width,
            label_width,
            shift,
            valid_size,
            test_size,
            strategy,
            batch_size,
        )

        w = w.from_dataframe(
            self.df,
            input_columns,
            label_columns,
            known_columns,
            date_columns,
        )

        # Raise an error if columns of data are not inside self.data
        with self.assertRaisesRegexp(AssertionError, error):
            w.production(data, None)

    @parameterized.parameters(
        [
            (2, 2, 2, 2, 2, "one_shot", ["data1"], ["data1"], ["label"], ["year"], 1),
        ]
    )
    def test_with_model(
        self,
        input_width,
        label_width,
        shift,
        valid_size,
        test_size,
        strategy,
        input_columns,
        label_columns,
        known_columns,
        date_columns,
        batch_size,
    ):
        model = create_interpretable_nbeats(
            input_width=2,
            label_width=2,
            periods=[2],
            back_periods=[2],
            forecast_fourier_order=[2],
            backcast_fourier_order=[2],
            p_degree=1,
            trend_n_neurons=16,
            seasonality_n_neurons=16,
            drop_rate=0,
            share=True,
        )

        w = WindowGenerator(
            input_width,
            label_width,
            shift,
            valid_size,
            test_size,
            strategy,
            batch_size,
        )

        w = w.from_dataframe(
            self.df,
            input_columns,
            label_columns,
            known_columns,
            date_columns,
        )

        model.compile(loss=QuantileLossError(quantiles=[0.5]))
        model.fit(w.train, validation_data=w.valid)
        model.evaluate(w.test)
        prod = w.production(self.df)
        model.predict(prod)
