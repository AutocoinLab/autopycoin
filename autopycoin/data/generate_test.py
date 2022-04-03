# pylint: skip-file

"""
Test for generate functions.
"""

import pytest
import numpy as np
from absl.testing import parameterized

import tensorflow as tf
from keras.backend import floatx

from .generate import random_ts


@pytest.fixture(scope="class")
def prepare_ts(request):
    request.cls.time_serie = random_ts(
        n_steps=100,
        trend_degree=1,
        periods=[10],
        fourier_orders=[10],
        trend_mean=0,
        trend_std=1,
        seasonality_mean=0,
        seasonality_std=1,
        batch_size=1,
        n_variables=1,
        noise=False,
        seed=42,
    )

    request.cls.time_serie_noise = random_ts(
        n_steps=100,
        trend_degree=1,
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

    request.cls.time_serie_periods = random_ts(
        n_steps=100,
        trend_degree=1,
        periods=[10, 5],
        fourier_orders=[10, 5],
        trend_mean=0,
        trend_std=1,
        seasonality_mean=0,
        seasonality_std=1,
        batch_size=1,
        n_variables=1,
        noise=False,
        seed=42,
    )


@pytest.mark.usefixtures("prepare_ts")
class TestGenerate(tf.test.TestCase, parameterized.TestCase):
    @parameterized.parameters(
        [
            (100, 1, [10], [10], 0, 1, 0, 1, 1, 1, False),
            (50, 1, [10], [10], 0, 1, 0, 1, 1, 3, False),
            (50, 1, [10, 10], [10, 10], 0, 1, 0, 1, 1, 3, False),
            (50, 1, [10], [10], 0, 1, 0, 1, 10, 3, False),
            (50, 1, [10], [10], 0, 1, 0, 1, [10, 10], 3, False),
            (50, 1, [10], [10], 0, 1, 0, 1, None, 3, False),
        ]
    )
    def test_output_shape(
        self,
        n_steps,
        trend_degree,
        periods,
        fourier_orders,
        trend_mean,
        trend_std,
        seasonality_mean,
        seasonality_std,
        batch_size,
        n_variables,
        noise,
    ):

        time_serie = random_ts(
            n_steps=n_steps,
            trend_degree=trend_degree,
            periods=periods,
            fourier_orders=fourier_orders,
            trend_mean=trend_mean,
            trend_std=trend_std,
            seasonality_mean=seasonality_mean,
            seasonality_std=seasonality_std,
            batch_size=batch_size,
            n_variables=n_variables,
            noise=noise,
            seed=42,
        )

        if isinstance(batch_size, int):
            batch_size = [batch_size]
        elif batch_size is None:
            batch_size = []

        expected_shape = batch_size + [n_steps, n_variables]
        self.assertEqual(time_serie.shape, expected_shape)

    @parameterized.parameters(
        [
            (100, 1, [10], [10], 0, 1, 0, 1, 1, 1, False),
            (50, 1, [10], [10], 0, 1, 0, 1, 1, 3, False),
            (50, 1, [10, 10], [10, 10], 0, 1, 0, 1, 1, 3, False),
            (50, 1, [10], [10], 0, 1, 0, 1, 10, 3, False),
            (50, 1, [10], [10], 0, 1, 0, 1, [10, 10], 3, False),
            (50, 1, [10], [10], 0, 1, 0, 1, None, 3, False),
        ]
    )
    def test_output_type(
        self,
        n_steps,
        trend_degree,
        periods,
        fourier_orders,
        trend_mean,
        trend_std,
        seasonality_mean,
        seasonality_std,
        batch_size,
        n_variables,
        noise,
    ):

        time_serie = random_ts(
            n_steps=n_steps,
            trend_degree=trend_degree,
            periods=periods,
            fourier_orders=fourier_orders,
            trend_mean=trend_mean,
            trend_std=trend_std,
            seasonality_std=seasonality_std,
            seasonality_mean=seasonality_mean,
            batch_size=batch_size,
            n_variables=n_variables,
            noise=noise,
            seed=42,
        )

        self.assertEqual(time_serie.dtype, floatx())

    def test_noise(self):
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(self.time_serie, self.time_serie_noise)

    def test_multi_periods(self):
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_equal,
            self.time_serie_periods,
            self.time_serie,
        )
