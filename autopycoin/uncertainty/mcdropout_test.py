# pylint: skip-file

"""
Unit test for MCDropout method.
"""

import pytest
import pandas as pd
import tensorflow as tf

from ..losses import QuantileLossError
from ..dataset import WindowGenerator
from ..data import random_ts
from ..models import create_interpretable_nbeats
from .mcdropout import MCDropoutEstimator


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

    w = WindowGenerator(
        input_width=50,
        label_width=20,
        shift=20,
        test_size=10,
        valid_size=10,
        strategy="one_shot",
        batch_size=32,
    )

    w = w.from_dataframe(
        data=data,
        input_columns=["test"],
        known_columns=[],
        label_columns=["test"],
        date_columns=[],
    )

    model_params = {
        "input_width": 50,
        "label_width": 20,
        "periods": [10],
        "back_periods": [10],
        "forecast_fourier_order": [10],
        "backcast_fourier_order": [10],
        "p_degree": 1,
        "trend_n_neurons": 32,
        "seasonality_n_neurons": 32,
        "drop_rate": 0.1,
        "share": True,
    }

    model_ql = create_interpretable_nbeats(**model_params)

    model_mse = create_interpretable_nbeats(**model_params)

    model_ql.compile(
        tf.keras.optimizers.Adam(
            learning_rate=0.015,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False,
            name="Adam",
        ),
        loss=QuantileLossError([0.1, 0.3, 0.5]),
        metrics=["mae"],
    )
    model_ql.fit(w.train, validation_data=w.valid, epochs=10)

    request.cls.inputs = w.train
    request.cls.model_ql = model_ql
    request.cls.estimator_ql = MCDropoutEstimator(n_preds=10, quantile=0.99)

    model_mse.compile(
        tf.keras.optimizers.Adam(
            learning_rate=0.015,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False,
            name="Adam",
        ),
        loss="mse",
        metrics=["mae"],
    )

    model_mse.fit(w.train, validation_data=w.valid, epochs=10)

    request.cls.model_mse = model_mse
    request.cls.estimator_mse = MCDropoutEstimator(n_preds=10, quantile=0.98)


@pytest.mark.usefixtures("prepare_data")
class MCDropoutEstimatorTest(tf.test.TestCase):
    """
    Unit tests for MC dropout estimator.
    """

    def test_attributes_ql(self):
        self.assertEqual(self.estimator_ql.n_preds, 10)
        self.assertEqual(self.estimator_ql.quantile, 0.99)

    def test_mc_dropout_estimation_ql(self):
        outputs = self.estimator_ql.mc_dropout_estimation(
            self.inputs, self.model_ql, self.estimator_ql.n_preds
        )
        self.assertEqual(outputs.shape, (10, 5, 273, 20))

    def test_call_ql(self):
        outputs = self.estimator_ql(self.inputs, self.model_ql)
        self.assertEqual(outputs[0].shape, (5, 273, 20))
        self.assertEqual(outputs[1].shape, (5, 273, 20))
        self.assertEqual(outputs[2].shape, (5, 273, 20))

    def test_attributes_mse(self):
        self.assertEqual(self.estimator_mse.n_preds, 10)
        self.assertEqual(self.estimator_mse.quantile, 0.98)

    def test_mc_dropout_estimation_mse(self):
        outputs = self.estimator_mse.mc_dropout_estimation(
            self.inputs, self.model_mse, self.estimator_mse.n_preds
        )
        self.assertEqual(outputs.shape, (10, 273, 20))

    def test_call_mse(self):
        outputs = self.estimator_mse(self.inputs, self.model_mse)
        self.assertEqual(outputs[0].shape, (273, 20))
        self.assertEqual(outputs[1].shape, (273, 20))
        self.assertEqual(outputs[2].shape, (273, 20))
