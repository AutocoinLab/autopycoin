"""
Test for nbeats model
"""

import pytest
import numpy as np

from tensorflow.python.keras import keras_parameterized
import tensorflow as tf

from ..utils.testing_utils import layer_test
from . import nbeats


@pytest.fixture(scope="class")
def prepare_data(request):

    request.cls.n_neurons = 3
    request.cls.horizon = 1
    request.cls.back_horizon = 2
    request.cls.p_degree = 2
    request.cls.n_neurons = 3
    request.cls.drop_rate = 0
    request.cls.quantiles = 1

    request.cls.trend_weights = [
        np.zeros(shape=(request.cls.back_horizon, request.cls.n_neurons)),
        np.ones(shape=(request.cls.n_neurons,)),
        np.zeros(shape=(request.cls.n_neurons, request.cls.n_neurons)),
        np.ones(shape=(request.cls.n_neurons,)),
        np.zeros(shape=(request.cls.n_neurons, request.cls.n_neurons)),
        np.ones(shape=(request.cls.n_neurons,)),
        np.zeros(shape=(request.cls.n_neurons, request.cls.n_neurons)),
        np.ones(shape=(request.cls.n_neurons,)),
        np.ones(shape=(request.cls.n_neurons, request.cls.p_degree)),
        np.ones(shape=(request.cls.n_neurons, request.cls.p_degree)),
        np.array([[1.0], [0.0]]),
        np.array([[1.0, 1.0], [0.0, 0.5]]),
    ]

    request.cls.seasonality_horizon = 2
    request.cls.seasonality_back_horizon = 3
    request.cls.periods = [2]
    request.cls.back_periods = [3]
    request.cls.forecast_fourier_order = [2]
    request.cls.backcast_fourier_order = [3]

    request.cls.forecast_neurons = tf.reduce_sum(2 * request.cls.periods)
    backcast_neurons = tf.reduce_sum(2 * request.cls.back_periods)

    request.cls.seasonality_weights = [
        np.zeros(shape=(request.cls.seasonality_back_horizon, request.cls.n_neurons)),
        np.ones(shape=(request.cls.n_neurons,)),
        np.zeros(shape=(request.cls.n_neurons, request.cls.n_neurons)),
        np.ones(shape=(request.cls.n_neurons,)),
        np.zeros(shape=(request.cls.n_neurons, request.cls.n_neurons)),
        np.ones(shape=(request.cls.n_neurons,)),
        np.zeros(shape=(request.cls.n_neurons, request.cls.n_neurons)),
        np.ones(shape=(request.cls.n_neurons,)),
        np.ones(shape=(request.cls.n_neurons, request.cls.forecast_neurons)),
        np.ones(shape=(request.cls.n_neurons, backcast_neurons)),
        np.array([[1, 1], [1, tf.cos(np.pi)], [0, 0], [0, tf.sin(np.pi)]]),
        np.array(
            [
                [1, 1, 1],
                [1, tf.cos(1 * (1 / 3) * 2 * np.pi), tf.cos(1 * (2 / 3) * 2 * np.pi)],
                [1, tf.cos(2 * (1 / 3) * 2 * np.pi), tf.cos(2 * (2 / 3) * 2 * np.pi)],
                [0, 0, 0],
                [0, tf.sin(1 * (1 / 3) * 2 * np.pi), tf.sin(1 * (2 / 3) * 2 * np.pi)],
                [0, tf.sin(2 * (1 / 3) * 2 * np.pi), tf.sin(2 * (2 / 3) * 2 * np.pi)],
            ]
        ),
    ]

    request.cls.y1 = (
        3
        * (
            1
            + tf.cos(1 * (1 / 3) * 2 * np.pi)
            + tf.cos(2 * (1 / 3) * 2 * np.pi)
            + tf.sin(1 * (1 / 3) * 2 * np.pi)
            + tf.sin(2 * (1 / 3) * 2 * np.pi)
        ).numpy()
    )

    request.cls.y2 = (
        3
        * (
            1
            + tf.cos(1 * (2 / 3) * 2 * np.pi)
            + tf.cos(2 * (2 / 3) * 2 * np.pi)
            + tf.sin(1 * (2 / 3) * 2 * np.pi)
            + tf.sin(2 * (2 / 3) * 2 * np.pi)
        ).numpy()
    )

    request.cls.trend_neurons = 5
    request.cls.seasonality_neurons = 5


@keras_parameterized.run_all_keras_modes
@pytest.mark.usefixtures("prepare_data")
class NBEATSLayersTest(keras_parameterized.TestCase):
    """
    Unit tests for the nbeats model.
    """

    def test_trendblock(self):

        layer_test(
            nbeats.TrendBlock,
            kwargs={
                "horizon": self.horizon,
                "back_horizon": self.back_horizon,
                "p_degree": self.p_degree,
                "n_neurons": self.n_neurons,
                "quantiles": self.quantiles,
                "drop_rate": self.drop_rate,
                "weights": self.trend_weights,
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

    def test_multi_quantiles_trendblock(self):
        quantiles = 2

        trend_weights = self.trend_weights.copy()
        trend_weights[-4] = np.ones(shape=(quantiles, self.n_neurons, self.p_degree))

        layer_test(
            nbeats.TrendBlock,
            kwargs={
                "horizon": self.horizon,
                "back_horizon": self.back_horizon,
                "p_degree": self.p_degree,
                "n_neurons": self.n_neurons,
                "quantiles": quantiles,
                "drop_rate": self.drop_rate,
                "weights": trend_weights,
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

    def test_dropout(self):
        drop_rate = 0.01

        model = nbeats.TrendBlock(
            n_neurons=30,
            horizon=self.horizon,
            back_horizon=self.back_horizon,
            p_degree=self.p_degree,
            drop_rate=drop_rate,
            quantiles=self.quantiles,
        )

        actual_1 = model(tf.constant([[1.0, 8.0], [1.0, 2.0]]))
        actual_2 = model(tf.constant([[1.0, 8.0], [1.0, 2.0]]))
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal, actual_1[1], actual_2[1]
        )

    def test_seasonalityblock(self):

        layer_test(
            nbeats.SeasonalityBlock,
            kwargs={
                "horizon": self.seasonality_horizon,
                "back_horizon": self.seasonality_back_horizon,
                "n_neurons": self.n_neurons,
                "periods": self.periods,
                "back_periods": self.back_periods,
                "forecast_fourier_order": self.forecast_fourier_order,
                "backcast_fourier_order": self.backcast_fourier_order,
                "quantiles": self.quantiles,
                "drop_rate": self.drop_rate,
                "weights": self.seasonality_weights,
            },
            input_dtype="float",
            input_shape=(2, 3),
            expected_output_shape=((None, 2), (None, 3)),
            expected_output_dtype=["float32", "float32"],
            expected_output=[
                tf.constant([6.0, 0.0, 6.0, 0.0], shape=(2, 2)),
                tf.constant(
                    [9.0, self.y1, self.y2, 9.0, self.y1, self.y2], shape=(2, 3)
                ),
            ],
            custom_objects={"SeasonalityBlock": nbeats.SeasonalityBlock},
        )

    def test_multi_quantiles_seasonalityblock(self):
        quantiles = 2

        seasonality_weights = self.seasonality_weights.copy()
        seasonality_weights[-4] = np.ones(
            shape=(quantiles, self.n_neurons, self.forecast_neurons)
        )

        layer_test(
            nbeats.SeasonalityBlock,
            kwargs={
                "horizon": self.seasonality_horizon,
                "back_horizon": self.seasonality_back_horizon,
                "n_neurons": self.n_neurons,
                "periods": self.periods,
                "back_periods": self.back_periods,
                "forecast_fourier_order": self.forecast_fourier_order,
                "backcast_fourier_order": self.backcast_fourier_order,
                "quantiles": quantiles,
                "drop_rate": self.drop_rate,
                "weights": seasonality_weights,
            },
            input_dtype="float",
            input_shape=(2, 3),
            expected_output_shape=((quantiles, None, 2), (None, 3)),
            expected_output_dtype=["float32", "float32"],
            expected_output=[
                tf.constant(
                    [6.0, 0.0, 6.0, 0.0, 6.0, 0.0, 6.0, 0.0],
                    shape=(quantiles, 2, 2),
                ),
                tf.constant(
                    [9.0, self.y1, self.y2, 9.0, self.y1, self.y2], shape=(2, 3)
                ),
            ],
            custom_objects={"SeasonalityBlock": nbeats.SeasonalityBlock},
        )

    def test_multi_periods_seasonalityblock(self):
        periods = [1, 2]
        back_periods = [2, 3]

        forecast_fourier_order = [1, 2]
        backcast_fourier_order = [2, 3]

        forecast_neurons = tf.reduce_sum(2 * periods)
        backcast_neurons = tf.reduce_sum(2 * back_periods)

        seasonality_weights = self.seasonality_weights.copy()
        seasonality_weights[-4:] = [
            np.ones(shape=(self.n_neurons, forecast_neurons)),
            np.ones(shape=(self.n_neurons, backcast_neurons)),
            np.array(
                [[1, 1], [0, 0], [1, 1], [1, tf.cos(np.pi)], [0, 0], [0, tf.sin(np.pi)]]
            ),
            np.array(
                [
                    [1, 1, 1],
                    [
                        1,
                        tf.cos(1 * (1 / 2) * 2 * np.pi),
                        tf.cos(1 * (2 / 2) * 2 * np.pi),
                    ],
                    [0, 0, 0],
                    [
                        0,
                        tf.sin(1 * (1 / 2) * 2 * np.pi),
                        tf.sin(1 * (2 / 2) * 2 * np.pi),
                    ],
                    [1, 1, 1],
                    [
                        1,
                        tf.cos(1 * (1 / 3) * 2 * np.pi),
                        tf.cos(1 * (2 / 3) * 2 * np.pi),
                    ],
                    [
                        1,
                        tf.cos(2 * (1 / 3) * 2 * np.pi),
                        tf.cos(2 * (2 / 3) * 2 * np.pi),
                    ],
                    [0, 0, 0],
                    [
                        0,
                        tf.sin(1 * (1 / 3) * 2 * np.pi),
                        tf.sin(1 * (2 / 3) * 2 * np.pi),
                    ],
                    [
                        0,
                        tf.sin(2 * (1 / 3) * 2 * np.pi),
                        tf.sin(2 * (2 / 3) * 2 * np.pi),
                    ],
                ]
            ),
        ]

        y1 = (
            3
            * (
                2
                + tf.cos(1 * (1 / 2) * 2 * np.pi)
                + tf.sin(1 * (1 / 2) * 2 * np.pi)
                + tf.cos(1 * (1 / 3) * 2 * np.pi)
                + tf.cos(2 * (1 / 3) * 2 * np.pi)
                + tf.sin(1 * (1 / 3) * 2 * np.pi)
                + tf.sin(2 * (1 / 3) * 2 * np.pi)
            ).numpy()
        )

        y2 = (
            3
            * (
                2
                + tf.cos(1 * (2 / 2) * 2 * np.pi)
                + tf.sin(1 * (2 / 2) * 2 * np.pi)
                + tf.cos(1 * (2 / 3) * 2 * np.pi)
                + tf.cos(2 * (2 / 3) * 2 * np.pi)
                + tf.sin(1 * (2 / 3) * 2 * np.pi)
                + tf.sin(2 * (2 / 3) * 2 * np.pi)
            ).numpy()
        )

        layer_test(
            nbeats.SeasonalityBlock,
            kwargs={
                "horizon": self.seasonality_horizon,
                "back_horizon": self.seasonality_back_horizon,
                "n_neurons": self.n_neurons,
                "periods": periods,
                "back_periods": back_periods,
                "forecast_fourier_order": forecast_fourier_order,
                "backcast_fourier_order": backcast_fourier_order,
                "quantiles": self.quantiles,
                "drop_rate": self.drop_rate,
                "weights": seasonality_weights,
            },
            input_dtype="float",
            input_shape=(2, 3),
            expected_output_shape=((None, 2), (None, 3)),
            expected_output_dtype=["float32", "float32"],
            expected_output=[
                tf.constant([9.0, 3.0, 9.0, 3.0], shape=(2, 2)),
                tf.constant([15.0, y1, y2, 15.0, y1, y2], shape=(2, 3)),
            ],
            custom_objects={"SeasonalityBlock": nbeats.SeasonalityBlock},
        )

    def test_genericblock(self):

        layer_test(
            nbeats.GenericBlock,
            kwargs={
                "horizon": self.horizon,
                "back_horizon": self.back_horizon,
                "n_neurons": self.n_neurons,
                "forecast_neurons": self.trend_neurons,
                "backcast_neurons": self.seasonality_neurons,
                "quantiles": self.quantiles,
                "drop_rate": self.drop_rate,
            },
            input_dtype="float",
            input_shape=(2, 2),
            expected_output_shape=((None, 1), (None, 2)),
            expected_output_dtype=["float32", "float32"],
            expected_output=None,
            custom_objects={"GenericBlock": nbeats.GenericBlock},
        )

    def test_stack_generic(self):

        kwargs = {
            "horizon": self.horizon,
            "back_horizon": self.back_horizon,
            "n_neurons": self.n_neurons,
            "forecast_neurons": self.trend_neurons,
            "backcast_neurons": self.seasonality_neurons,
            "quantiles": self.quantiles,
            "drop_rate": self.drop_rate,
        }
        blocks = [nbeats.GenericBlock(**kwargs), nbeats.GenericBlock(**kwargs)]

        layer_test(
            nbeats.Stack,
            kwargs={"blocks": blocks},
            input_dtype="float",
            input_shape=(2, 2),
            expected_output_shape=((None, 1), (None, 2)),
            expected_output_dtype=["float32", "float32"],
            expected_output=None,
            custom_objects={"Stack": nbeats.Stack},
        )

    def test_stack_interpretable(self):

        trend_weights = self.trend_weights.copy()
        trend_weights[0] = np.zeros(shape=(3, 3))
        trend_weights[-2:] = [np.array([[1.0, 1.0], [0.0, 0.5]]),
                             np.array([[1.0, 1.0, 1.0], [0.0, 1/3, 2/3]])]
        kwargs_1 = {
            "horizon": self.seasonality_horizon,
            "back_horizon": self.seasonality_back_horizon,
            "p_degree": self.p_degree,
            "n_neurons": self.n_neurons,
            "quantiles": self.quantiles,
            "drop_rate": self.drop_rate,
            "weights": trend_weights
        }

        kwargs_2 = {
            "horizon": self.seasonality_horizon,
            "back_horizon": self.seasonality_back_horizon,
            "n_neurons": self.n_neurons,
            "periods": self.periods,
            "back_periods": self.back_periods,
            "forecast_fourier_order": self.forecast_fourier_order,
            "backcast_fourier_order": self.backcast_fourier_order,
            "quantiles": self.quantiles,
            "drop_rate": self.drop_rate,
            "weights": self.seasonality_weights
        }

        blocks = [nbeats.TrendBlock(**kwargs_1), nbeats.SeasonalityBlock(**kwargs_2)]

        layer_test(
            nbeats.Stack,
            kwargs={"blocks": blocks},
            input_dtype="float",
            input_shape=(2, 3),
            expected_output_shape=((None, 2), (None, 3)),
            expected_output_dtype=["float32", "float32"],
            expected_output=[
                tf.constant([9.0, 4.5, 9.0, 4.5], shape=(2, 2)),
                -1 * tf.constant([12.0, self.y1 + 3 + 1, self.y2 + 3 + 2, 12.0, self.y1 + 3 + 1, self.y2 + 3 + 2], shape=(2, 3)),
            ],
            custom_objects={"Stack": nbeats.Stack},
        )

    def test_nbeats(self):

        trend_weights = self.trend_weights.copy()
        trend_weights[0] = np.zeros(shape=(3, 3))
        trend_weights[-2:] = [np.array([[1.0, 1.0], [0.0, 0.5]]),
                             np.array([[1.0, 1.0, 1.0], [0.0, 1/3, 2/3]])]
        kwargs_1 = {
            "horizon": self.seasonality_horizon,
            "back_horizon": self.seasonality_back_horizon,
            "p_degree": self.p_degree,
            "n_neurons": self.n_neurons,
            "quantiles": self.quantiles,
            "drop_rate": self.drop_rate,
            "weights": trend_weights
        }

        kwargs_2 = {
            "horizon": self.seasonality_horizon,
            "back_horizon": self.seasonality_back_horizon,
            "n_neurons": self.n_neurons,
            "periods": self.periods,
            "back_periods": self.back_periods,
            "forecast_fourier_order": self.forecast_fourier_order,
            "backcast_fourier_order": self.backcast_fourier_order,
            "quantiles": self.quantiles,
            "drop_rate": self.drop_rate,
            "weights": self.seasonality_weights
        }

        blocks_1 = nbeats.Stack([nbeats.TrendBlock(**kwargs_1), nbeats.SeasonalityBlock(**kwargs_2)])
        blocks_2 = nbeats.Stack([nbeats.TrendBlock(**kwargs_1), nbeats.SeasonalityBlock(**kwargs_2)])
        stacks = [blocks_1, blocks_2]

        layer_test(
            nbeats.NBEATS,
            kwargs={"stacks": stacks},
            input_dtype="float",
            input_shape=(2, 3),
            expected_output_shape=((None, 2), (None, 3)),
            expected_output_dtype=["float32", "float32"],
            expected_output=tf.constant([18.0, 9.0, 18.0, 9.0], shape=(2, 2)),
        )

    def test_nbeats_attributes(self):

        trend_weights = self.trend_weights.copy()
        trend_weights[0] = np.zeros(shape=(3, 3))
        trend_weights[-2:] = [np.array([[1.0, 1.0], [0.0, 0.5]]),
                             np.array([[1.0, 1.0, 1.0], [0.0, 1/3, 2/3]])]
        kwargs_1 = {
            "horizon": self.seasonality_horizon,
            "back_horizon": self.seasonality_back_horizon,
            "p_degree": self.p_degree,
            "n_neurons": self.n_neurons,
            "quantiles": self.quantiles,
            "drop_rate": self.drop_rate,
            "weights": trend_weights
        }

        kwargs_2 = {
            "horizon": self.seasonality_horizon,
            "back_horizon": self.seasonality_back_horizon,
            "n_neurons": self.n_neurons,
            "periods": self.periods,
            "back_periods": self.back_periods,
            "forecast_fourier_order": self.forecast_fourier_order,
            "backcast_fourier_order": self.backcast_fourier_order,
            "quantiles": self.quantiles,
            "drop_rate": self.drop_rate,
            "weights": self.seasonality_weights
        }

        blocks_1 = nbeats.Stack([nbeats.TrendBlock(**kwargs_1), nbeats.TrendBlock(**kwargs_1)])
        blocks_2 = nbeats.Stack([nbeats.SeasonalityBlock(**kwargs_2), nbeats.SeasonalityBlock(**kwargs_2)])
        stacks = [blocks_1, blocks_2]

        model = nbeats.NBEATS(stacks)
        inputs = np.array([[0., 0., 0.],
                           [0., 0., 0.]])

        model(inputs)
        self.assertNotEmpty(model.seasonality)
        self.assertNotEmpty(model.trend)

        kwargs_3 = {
            "horizon": self.horizon,
            "back_horizon": self.back_horizon,
            "n_neurons": self.n_neurons,
            "forecast_neurons": self.trend_neurons,
            "backcast_neurons": self.seasonality_neurons,
            "quantiles": self.quantiles,
            "drop_rate": self.drop_rate,
        }

        blocks_3 = nbeats.Stack([nbeats.GenericBlock(**kwargs_3), nbeats.GenericBlock(**kwargs_3)])

        stacks = [blocks_3, blocks_3]

        model = nbeats.NBEATS(stacks)
        inputs = np.array([[0., 0.],
                           [0., 0.]])

        model(inputs)

        with self.assertRaises(AttributeError):
            model.seasonality
            model.trend

    def test_create_interpretable_nbeats(self):

        model = nbeats.create_interpretable_nbeats(
            horizon = self.seasonality_horizon,
            back_horizon = self.seasonality_back_horizon,
            periods = self.periods,
            back_periods = self.back_periods,
            forecast_fourier_order = self.forecast_fourier_order,
            backcast_fourier_order = self.backcast_fourier_order,
            p_degree=self.p_degree,
            trend_n_neurons=self.n_neurons,
            seasonality_n_neurons=self.n_neurons,
            quantiles=self.quantiles,
            drop_rate=0,
            share=True)

        self.assertIsInstance(model, nbeats.NBEATS)
        trend_stack = model.stacks[0]
        seasonality_stack = model.stacks[1]
        self.assertIs(trend_stack.name, 'trend_stack')
        self.assertIs(seasonality_stack.name, 'seasonality_stack')
        
        # Test if interpretable model is composed by trend and seasonality blocks only
        for block in trend_stack.blocks:
            self.assertIsInstance(block, nbeats.TrendBlock)
        for block in seasonality_stack.blocks:
            self.assertIsInstance(block, nbeats.SeasonalityBlock)

        # Compare weights values, expected to be equals over blocks because share=True
        self.assertEquals(trend_stack.blocks[0].get_weights(),
        trend_stack.blocks[1].get_weights(),
        trend_stack.blocks[2].get_weights())

        self.assertEquals(seasonality_stack.blocks[0].get_weights(),
        seasonality_stack.blocks[1].get_weights(),
        seasonality_stack.blocks[2].get_weights())

        # Compare output shape with expected
        outputs = model.predict(np.array([[1.0, 2.0, 3.0]]))
        self.assertEqual(outputs.shape, (1, 2))

        # Compare output shape with expected when quantiles = 2
        model = nbeats.create_interpretable_nbeats(
            horizon = self.seasonality_horizon,
            back_horizon = self.seasonality_back_horizon,
            periods = self.periods,
            back_periods = self.back_periods,
            forecast_fourier_order = self.forecast_fourier_order,
            backcast_fourier_order = self.backcast_fourier_order,
            p_degree=self.p_degree,
            trend_n_neurons=self.n_neurons,
            seasonality_n_neurons=self.n_neurons,
            quantiles=2,
            drop_rate=0,
            share=True)

        outputs = model.predict(np.array([[1.0, 2.0, 3.0]]))
        self.assertEqual(outputs.shape, (2, 1, 2))
