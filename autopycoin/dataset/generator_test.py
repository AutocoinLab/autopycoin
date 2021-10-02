"""
Unit tests for generator class
"""

import pandas as pd
import numpy as np
import pytest
import logging

import tensorflow as tf
from tensorflow.python.keras import keras_parameterized

from .generator import WindowGenerator
from ..models.nbeats import create_interpretable_nbeats
from ..losses.losses import QuantileLossError


@pytest.fixture(scope="class")
def prepare_generator(request):
    request.cls.data = pd.DataFrame(
        [
            [10, 11, 12, 13, 100, 2000, 0],
            [21, 22, 23, 24, 101, 2001, 6],
            [31, 32, 33, 34, 102, 2002, 2],
            [41, 42, 43, 44, 103, 2003, 3],
            [51, 52, 53, 54, 104, 2004, 4],
            [61, 62, 63, 64, 105, 2005, 5],
            [71, 72, 73, 74, 106, 2006, 6],
            [81, 82, 83, 84, 107, 2007, 17],
            [91, 92, 93, 94, 108, 2008, 8],
            [101, 102, 103, 104, 109, 2009, 9],
            [10, 11, 12, 13, 100, 2010, 7],
            [21, 22, 23, 24, 101, 2011, 10],
            [31, 32, 33, 34, 102, 2012, 15],
            [41, 42, 43, 44, 103, 2013, 17],
            [51, 52, 53, 54, 104, 2014, 19],
        ],
        columns=["data1", "data2", "data3", "data4", "known", "annee", "label"],
    )
    request.cls.prod_data = pd.DataFrame(
        [
            [10, 11, 12, 13, 100, 2000, 0],
            [21, 22, 23, 24, 101, 2001, 0],
            [31, 32, 33, 34, 102, 2002, 0],
            [41, 42, 43, 44, 103, 2003, 0],
        ],
        columns=["data1", "data2", "data3", "data4", "known", "annee", "label"],
    )

    forecast = 3  # number of predict step
    test_size = 2  # number of test instances
    valid_size = 2  # number of validation instances
    strategy = "auto_regressive"
    
    input_columns = [
        "data1",
        "data2",
        "data3",
        "data4",
        "known",
        "annee",
        "label",
    ]

    window = 2

    known_columns = ["annee", "known"]
    label_columns = ["label"]
    date_columns = ["annee"]

    request.cls.w_autoreg = WindowGenerator(
        data=request.cls.data,
        input_width=window,
        label_width=forecast,
        shift=forecast,
        test_size=test_size,
        valid_size=valid_size,
        strategy=strategy,
        batch_size=None,
        input_columns=input_columns,
        known_columns=known_columns,
        label_columns=label_columns,
        date_columns=date_columns,
    )

    request.cls.x, request.cls.y = iter(request.cls.w_autoreg.train).get_next()

    input_columns = [
        "label",
    ]

    request.cls.w_oneshot = WindowGenerator(
        data=request.cls.data,
        input_width=window,
        label_width=forecast,
        shift=forecast,
        test_size=test_size,
        valid_size=valid_size,
        strategy='one_shot',
        batch_size=None,
        input_columns=input_columns,
        known_columns=None,
        label_columns=label_columns,
        date_columns=None,
    )

    request.cls.x_oneshot, request.cls.y_oneshot = iter(request.cls.w_oneshot.train).get_next()

    logging.info("Execute generator fixture")


@keras_parameterized.run_all_keras_modes
@pytest.mark.usefixtures("prepare_generator")
class TestGenerator(keras_parameterized.TestCase):
    def test_frames_train_valid_test_ds(self):
        ds = pd.concat([self.w_autoreg.train_ds, self.w_autoreg.valid_ds])
        ds = pd.concat([ds, self.w_autoreg.test_ds])
        ds = ds.drop_duplicates()

        pd.testing.assert_frame_equal(self.data, ds, check_exact=True, check_like=True)

    def test_n_common_lines(self):
        common_lines = len(
            pd.merge(
                self.w_autoreg.train_ds,
                self.w_autoreg.valid_ds,
                how="inner",
                right_index=True,
                left_index=True,
            ).index
        )
        common_lines = common_lines + len(
            pd.merge(
                self.w_autoreg.test_ds,
                self.w_autoreg.valid_ds,
                how="inner",
                right_index=True,
                left_index=True,
            ).index
        )

        assert common_lines == 2 * self.w_autoreg.input_width

    def test_loc_common_lines_train_valid(self):
        common_lines_indices = pd.merge(
            self.w_autoreg.train_ds,
            self.w_autoreg.valid_ds,
            how="inner",
            right_index=True,
            left_index=True,
            suffixes=("", "_y"),
        )[self.w_autoreg.train_ds.columns]
        train_lines = self.w_autoreg.train_ds.iloc[-len(common_lines_indices) :]
        pd.testing.assert_frame_equal(
            train_lines, common_lines_indices, check_exact=True, check_like=True
        )

    def test_loc_common_lines_test_valid(self):
        common_lines_indices = pd.merge(
            self.w_autoreg.test_ds,
            self.w_autoreg.valid_ds,
            how="inner",
            right_index=True,
            left_index=True,
            suffixes=("", "_y"),
        )[self.w_autoreg.valid_ds.columns]
        valid_lines = self.w_autoreg.valid_ds.iloc[-len(common_lines_indices) :]
        pd.testing.assert_frame_equal(
            valid_lines, common_lines_indices, check_exact=True, check_like=True
        )

    def test_no_common_lines(self):
        common_lines = pd.merge(
            self.w_autoreg.train_ds,
            self.w_autoreg.test_ds,
            how="inner",
            right_index=True,
            left_index=True,
        )
        assert len(common_lines) == 0

    def test_label_columns_indices(self):
        assert self.w_autoreg.label_columns_indices == {
            "label": 0,
        }

    def test_inputs_columns_indices(self):
        assert self.w_autoreg.inputs_columns_indices == {
            "data1": 0,
            "data2": 1,
            "data3": 2,
            "data4": 3,
            "known": 4,
            "annee": 5,
            "label": 6,
        }

    def test_labels_in_inputs_indices(self):
        assert self.w_autoreg.labels_in_inputs_indices == {"label": 6}

    def test_columns_indices(self):
        assert self.w_autoreg.inputs_columns_indices == {
            "data1": 0,
            "data2": 1,
            "data3": 2,
            "data4": 3,
            "known": 4,
            "annee": 5,
            "label": 6,
        }

    def test_y_train(self):

        y_true = np.array([[[2.0], [3.0], [4.0]], [[3.0], [4.0], [5.0]]])

        np.testing.assert_array_equal(self.y, y_true)

    def test_x0_train(self):

        x_true = np.array(
            [
                [
                    [
                        1.000e01,
                        1.100e01,
                        1.200e01,
                        1.300e01,
                        1.000e02,
                        2.000e03,
                        0.000e00,
                    ],
                    [
                        2.100e01,
                        2.200e01,
                        2.300e01,
                        2.400e01,
                        1.010e02,
                        2.001e03,
                        6.000e00,
                    ],
                ],
                [
                    [
                        2.100e01,
                        2.200e01,
                        2.300e01,
                        2.400e01,
                        1.010e02,
                        2.001e03,
                        6.000e00,
                    ],
                    [
                        3.100e01,
                        3.200e01,
                        3.300e01,
                        3.400e01,
                        1.020e02,
                        2.002e03,
                        2.000e00,
                    ],
                ],
            ]
        )

        np.testing.assert_array_equal(self.x[0], x_true)

    def test_x1_train(self):

        x_true = np.array(
            [
                [[2002.0, 102.0], [2003.0, 103.0], [2004.0, 104.0]],
                [[2003.0, 103.0], [2004.0, 104.0], [2005.0, 105.0]],
            ]
        )

        np.testing.assert_array_equal(self.x[1], x_true)

    def test_x2_train(self):

        x_true = np.array([[b"2000", b"2001"], [b"2001", b"2002"]])

        np.testing.assert_array_equal(self.x[2], x_true)

    def test_x3_train(self):

        x_true = np.array([[b"2002", b"2003", b"2004"], [b"2003", b"2004", b"2005"]])

        np.testing.assert_array_equal(self.x[3], x_true)

    def test_oneshot_vs_autoreg(self):

        np.testing.assert_array_equal(self.x_oneshot[0], [[0., 6.], [6., 2.]])
        np.testing.assert_array_equal(self.y_oneshot, [[2., 3., 4.], [3., 4., 5.]])
        
    def test_with_model(self):
        model = create_interpretable_nbeats(
            horizon=3,
            back_horizon=2,
            periods=[3],
            back_periods=[2],
            forecast_fourier_order=[3],
            backcast_fourier_order=[2],
            p_degree=1,
            trend_n_neurons=16,
            seasonality_n_neurons=16,
            quantiles=1,
            drop_rate=0,
            share=True
        )

        model.compile(loss=QuantileLossError(quantiles=[0.5]))
        model.fit(self.w_oneshot.train, validation_data=self.w_oneshot.valid)
        model.evaluate(self.w_oneshot.test)