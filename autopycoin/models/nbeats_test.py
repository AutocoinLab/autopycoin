"""
Test for nbeats model
"""

import numpy as np

from tensorflow.python.keras import keras_parameterized
import tensorflow as tf

from ..utils.testing_utils import layer_test
from . import nbeats


@keras_parameterized.run_all_keras_modes
class NBEATSLayersTest(keras_parameterized.TestCase):
    """
    Unit tests for the nbeats model.
    """

    def test_trendblock(self):
        """
        Define multiple test for `TrendBlock`.
        """
        n_neurons = 3
        horizon = 1
        back_horizon = 2
        p_degree = 2
        n_neurons = 3
        drop_rate = 0
        quantiles = 1

        weights = [
            np.zeros(shape=(back_horizon, n_neurons)),
            np.ones(shape=(n_neurons,)),
            np.zeros(shape=(n_neurons, n_neurons)),
            np.ones(shape=(n_neurons,)),
            np.zeros(shape=(n_neurons, n_neurons)),
            np.ones(shape=(n_neurons,)),
            np.zeros(shape=(n_neurons, n_neurons)),
            np.ones(shape=(n_neurons,)),
            np.ones(shape=(n_neurons, p_degree)),
            np.ones(shape=(n_neurons, p_degree)),
            np.array([[1.0], [0.0]]),
            np.array([[1.0, 1.0], [0.0, 0.5]]),
        ]

        layer_test(
            nbeats.TrendBlock,
            kwargs={
                "horizon": horizon,
                "back_horizon": back_horizon,
                "p_degree": p_degree,
                "n_neurons": n_neurons,
                "quantiles": quantiles,
                "drop_rate": drop_rate,
                "weights": weights,
            },
            input_dtype="float",
            input_shape=(2, 2),
            expected_output_shape=((None, 1), (None, 2)),
            expected_output_dtype=["float32", "float32"],
            expected_output=[
                tf.constant(3.0, shape=(2, 1)),
                tf.constant([3.0, 4.5, 3.0, 4.5], shape=(2, 2)),
            ],
            custom_objects={"TrendBlock": nbeats.TrendBlock},
        )

        quantiles = 2

        weights[-4] = np.ones(shape=(quantiles, n_neurons, p_degree))

        layer_test(
            nbeats.TrendBlock,
            kwargs={
                "horizon": horizon,
                "back_horizon": back_horizon,
                "p_degree": p_degree,
                "n_neurons": n_neurons,
                "quantiles": quantiles,
                "drop_rate": drop_rate,
                "weights": weights,
            },
            input_dtype="float",
            input_shape=(2, 2),
            expected_output_shape=((quantiles, None, 1), (None, 2)),
            expected_output_dtype=["float32", "float32"],
            expected_output=[
                tf.constant(3.0, shape=(quantiles, 2, 1)),
                tf.constant([3.0, 4.5, 3.0, 4.5], shape=(2, 2)),
            ],
            custom_objects={"TrendBlock": nbeats.TrendBlock},
        )

        model = nbeats.TrendBlock(
            n_neurons=30,
            horizon=1,
            back_horizon=2,
            p_degree=2,
            drop_rate=0.01,
            quantiles=1,
        )

        actual_1 = model(tf.constant([[1.0, 8.0], [1.0, 2.0]]))
        actual_2 = model(tf.constant([[1.0, 8.0], [1.0, 2.0]]))
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal, actual_1[1], actual_2[1]
        )

    def test_seasonalityblock(self):
        """
        Define multiple test for `SeasonalityBlock`.
        """
        n_neurons = 3
        horizon = 2
        back_horizon = 3
        periods = [2]
        back_periods = [3]
        forecast_fourier_order = [2]
        backcast_fourier_order = [3]
        n_neurons = 3
        quantiles = 1
        drop_rate = 0

        forecast_neurons = tf.reduce_sum(2 * periods)
        backcast_neurons = tf.reduce_sum(2 * back_periods)

        weights = [
            np.zeros(shape=(back_horizon, n_neurons)),
            np.ones(shape=(n_neurons,)),
            np.zeros(shape=(n_neurons, n_neurons)),
            np.ones(shape=(n_neurons,)),
            np.zeros(shape=(n_neurons, n_neurons)),
            np.ones(shape=(n_neurons,)),
            np.zeros(shape=(n_neurons, n_neurons)),
            np.ones(shape=(n_neurons,)),
            np.ones(shape=(n_neurons, forecast_neurons)),
            np.ones(shape=(n_neurons, backcast_neurons)),
            np.array([[1, 1],
            [1, tf.cos(np.pi)],
            [0, 0],
            [0, tf.sin(np.pi)]]),
            np.array(
            [[1, 1, 1],
            [1, tf.cos(1 * (1/3) * 2 * np.pi), tf.cos(1 * (2/3) * 2 * np.pi)],
            [1, tf.cos(2 * (1/3) * 2 * np.pi), tf.cos(2 * (2/3) * 2 * np.pi)],
            [0, 0, 0],
            [0, tf.sin(1 * (1/3) * 2 * np.pi), tf.sin(1 * (2/3) * 2 * np.pi)],
            [0, tf.sin(2 * (1/3) * 2 * np.pi), tf.sin(2 * (2/3) * 2 * np.pi)]]
        )]

        layer_test(
            nbeats.SeasonalityBlock,
            kwargs={
                "horizon" : horizon,
                "back_horizon" : back_horizon,
                "n_neurons" : n_neurons,
                "periods" : periods,
                "back_periods" : back_periods,
                "forecast_fourier_order" : forecast_fourier_order,
                "backcast_fourier_order" : backcast_fourier_order,
                "quantiles" : quantiles,
                "drop_rate" : drop_rate,
                "weights": weights
            },
            input_dtype="float",
            input_shape=(2, 3),
            expected_output_shape=((None, 2), (None, 3)),
            expected_output_dtype=["float32", "float32"],
            expected_output=[
                tf.constant([6., 0., 6., 0.], shape=(2, 2)),
                tf.constant([9., 0., 0., 9., 0., 0.], shape=(2, 3)),
            ],
            custom_objects={"SeasonalityBlock": nbeats.SeasonalityBlock},
        )
