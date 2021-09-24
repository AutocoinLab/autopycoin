"""Test for nbeats model
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
            np.zeros(shape=(2, n_neurons)),
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

        weights[-3] = np.ones(shape=(quantiles, n_neurons, p_degree))

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
