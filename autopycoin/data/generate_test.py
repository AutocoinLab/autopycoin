# pylint: skip-file

"""
Test for generate functions.
"""

import pytest
import numpy as np

import tensorflow as tf
from tensorflow.python.keras import keras_parameterized

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

    request.cls.time_serie_batch = random_ts(
        n_steps=100,
        trend_degree=1,
        periods=[10],
        fourier_orders=[10],
        trend_mean=0,
        trend_std=1,
        seasonality_mean=0,
        seasonality_std=1,
        batch_size=2,
        n_variables=1,
        noise=True,
        seed=42,
    )

    request.cls.time_serie_variables = random_ts(
        n_steps=100,
        trend_degree=1,
        periods=[10],
        fourier_orders=[10],
        trend_mean=0,
        trend_std=1,
        seasonality_mean=0,
        seasonality_std=1,
        batch_size=2,
        n_variables=2,
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
class TestGenerate(keras_parameterized.TestCase):
    def test_output_shape(self):
        self.assertEqual(self.time_serie.shape, (1, 100, 1))

    def test_noise(self):
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_equal,
            self.time_serie,
            self.time_serie_noise,
        )

    def test_batch(self):
        self.assertEqual(self.time_serie_batch.shape, (2, 100, 1))

    def test_columns(self):
        self.assertEqual(self.time_serie_variables.shape, (2, 100, 2))

    def test_multi_periods(self):
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_equal,
            self.time_serie_periods,
            self.time_serie,
        )
