# pylint: skip-file

import pytest
import numpy as np
import pandas as pd
from absl.testing import parameterized

from tensorflow.python.keras import keras_parameterized
from tensorflow.keras.backend import floatx
import tensorflow as tf

from ..utils import layer_test, check_attributes
from . import NBEATS, Stack, GenericBlock, TrendBlock, SeasonalityBlock, BaseBlock
from ..data import random_ts
from ..dataset import WindowGenerator


class ExampleBlock(BaseBlock):
                def __init__(self,
                            input_width: int,
                            label_width: int,
                            output_first_dim_forecast: int,
                            output_first_dim_backcast: int,
                            n_neurons: int,
                            drop_rate: float,
                            g_trainable: bool=False,
                            interpretable: bool=False,
                            block_type: str='BaseBlock'):
                    super().__init__(
                            input_width=input_width,
                            label_width=label_width,
                            output_first_dim_forecast=output_first_dim_forecast,
                            output_first_dim_backcast=output_first_dim_backcast,
                            n_neurons=n_neurons,
                            drop_rate=drop_rate,
                            g_trainable=g_trainable,
                            interpretable=interpretable,
                            block_type=block_type)

                def coefficient_factory(self, *args: list, **kwargs: dict):
                    pass

                def _get_backcast_coefficients(self):
                    return tf.constant([0])
                def _get_forecast_coefficients(self):
                    return tf.constant([0])


@keras_parameterized.run_all_keras_modes
class BlocksLayersTest(tf.test.TestCase, parameterized.TestCase):
    """
    Unit tests for the nbeats model.
    """

    @parameterized.parameters(
        [# BaseBlock attributes test
         ((10, 10, 10, 10, 16, 0.5), BaseBlock, ['input_width', 'label_width', 'drop_rate', 'is_interpretable', 'is_g_trainable', 'block_type'],
         [10, 10, 0.5, False, False, 'BaseBlock']),
         # TrendBlock attributes test
         ((1, 2, 2, 3, 0.), TrendBlock, ['input_width', 'label_width', 'p_degree', 'drop_rate', 'is_interpretable', 'is_g_trainable', 'block_type'],
         [1, 2, 2, 0., True, False, 'TrendBlock']),
         # SeasonalityBlock attributes test
         ((10, 10, [10], [10], [10], [10], 16, 0.5), SeasonalityBlock, 
         ['input_width', 'label_width', 'periods', 'back_periods', 'forecast_fourier_order', 'backcast_fourier_order', 'drop_rate', 'is_interpretable', 'is_g_trainable', 'block_type'],
         [10, 10, [10], [10], [10], [10], 0.5, True, False, 'SeasonalityBlock']),
         # GenericBlock attributes test
         ((10, 10, 10, 10, 16, 0.5), GenericBlock, ['input_width', 'label_width', 'drop_rate', 'is_interpretable', 'is_g_trainable', 'block_type'],
         [10, 10, 0.5, False, True, "GenericBlock"]),]
    )
    def test_attributes_and_property(self,
        args,
        cls,
        attributes,
        expected_values):

        if cls == BaseBlock:
            block = ExampleBlock(*args)
        else:
            block = cls(*args)

        check_attributes(self, block, attributes, expected_values)

    @parameterized.parameters(
        [
         ((-10, 10, 10, 10, 16, 0.5), BaseBlock, 'Received an invalid values for `input_width` or `label_width`'),
         ((10, -10, 10, 10, 16, 0.5), BaseBlock, 'Received an invalid values for `input_width` or `label_width`'),
         ((10, 10, -10, 10, 16, 0.5), GenericBlock, 'Received an invalid value for `forecast_neurons` or `backcast_neurons`'),
         ((10, 10, 10, -10, 16, 0.5), GenericBlock, 'Received an invalid value for `forecast_neurons` or `backcast_neurons`'),
         ((10, 10, 10, 10, -16, 0.5), BaseBlock, 'Received an invalid value for `n_neurons`'),
         ((10, 10, 10, 10, 16, -0.5), BaseBlock, 'Received an invalid value for `drop_rate`'),
         ((10, 10, 10, 10, 16, 0.5, False, False, 'base'), BaseBlock, '`name` has to contain `Block`'),
         ((10, 10, 10, 10, 16, 0.5), BaseBlock, "layer doesn't match the desired shape"),
         ((10, 10, -10, 16, 0.5), TrendBlock, "Received an invalid value for `p_degree`, expected"),
         ((10, 10, [10, 10], [10], [10], [10], 16, 0.5), SeasonalityBlock, "`periods` and `forecast_fourier_order` are expected"),
         ((10, 10, [10], [10, 10], [10], [10], 16, 0.5), SeasonalityBlock, "`back_periods` and `backcast_fourier_order` are expected"),
         ((10, 10, [], [10], [10], [10], 16, 0.5), SeasonalityBlock, "`periods` have to be a non-empty list and all elements have to be strictly positives values."),
         ((10, 10, [10], [], [10], [10], 16, 0.5), SeasonalityBlock, "`back_periods` have to be a non-empty list"),
         ((10, 10, [-10], [10], [10], [10], 16, 0.5), SeasonalityBlock, "`periods` have to be a non-empty list and all elements have to be strictly positives values."),
         ((10, 10, [10], [-10], [10], [10], 16, 0.5), SeasonalityBlock, "`back_periods` have to be a non-empty list"),
        ]
    )
    def test_raises_error(self,
        args,
        cls,
        error):

        with self.assertRaisesRegexp(ValueError, error):
            if cls == BaseBlock:
                obj = ExampleBlock(*args)
            else:
                obj = cls(*args)
            
            with self.assertRaisesRegexp(AssertionError, error):
                obj.build(tf.TensorShape((None, args[0])))

            raise ValueError(error)

    @parameterized.parameters(
        [
         (1, 2, 2, 3, 0.),
         (1, 2, 2, 3, 0.01),
        ]
    )
    def test_trendblock(self, 
    input_width, label_width, p_degree, n_neurons, drop_rate):

        trend_weights = [
            np.zeros(shape=(label_width, n_neurons)),
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

        if drop_rate == 0.:
            layer_test(
                TrendBlock,
                kwargs={
                    "input_width": input_width,
                    "label_width": label_width,
                    "p_degree": label_width,
                    "n_neurons": n_neurons,
                    "drop_rate": drop_rate,
                    "weights": trend_weights,
                },
                input_dtype=floatx(),
                input_shape=(2, 2),
                expected_output_shape=((None, 1), (None, 2)),
                expected_output_dtype=[floatx(), floatx()],
                expected_output=[
                    tf.constant(3.0, shape=(2, 1)),
                    tf.constant([3.0, 4.5, 3.0, 4.5], shape=(2, 2)),
                ],
                custom_objects={"TrendBlock": TrendBlock},
            )

        elif drop_rate > 0:
            model = TrendBlock(
                n_neurons=30,
                input_width=input_width,
                label_width=label_width,
                p_degree=p_degree,
                drop_rate=drop_rate,
            )
            actual_1 = model(tf.constant([[1.0, 8.0], [1.0, 2.0]]))
            actual_2 = model(tf.constant([[1.0, 8.0], [1.0, 2.0]]))
            np.testing.assert_raises(
                AssertionError, np.testing.assert_array_equal, actual_1[1], actual_2[1]
            )

    @parameterized.parameters(
        [
         (2, 3, [2], [3], [2], [3], 3, 0.)
        ]
    )
    def test_seasonalityblock(
        self,
        input_width,
        label_width,
        periods,
        back_periods,
        forecast_fourier_order,
        backcast_fourier_order,
        n_neurons,
        drop_rate,
        ):

        forecast_neurons = tf.reduce_sum(2 * periods)
        backcast_neurons = tf.reduce_sum(2 * back_periods)

        seasonality_weights = [
        np.zeros(shape=(label_width, n_neurons)),
        np.ones(shape=(n_neurons,)),
        np.zeros(shape=(n_neurons, n_neurons)),
        np.ones(shape=(n_neurons,)),
        np.zeros(shape=(n_neurons, n_neurons)),
        np.ones(shape=(n_neurons,)),
        np.zeros(shape=(n_neurons, n_neurons)),
        np.ones(shape=(n_neurons,)),
        np.ones(shape=(n_neurons, forecast_neurons)),
        np.ones(shape=(n_neurons, backcast_neurons)),
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
        y1 = (
        3
        * (
            1
            + tf.cos(1 * (1 / 3) * 2 * np.pi)
            + tf.cos(2 * (1 / 3) * 2 * np.pi)
            + tf.sin(1 * (1 / 3) * 2 * np.pi)
            + tf.sin(2 * (1 / 3) * 2 * np.pi)
        ).numpy()
    )

        y2 = (
        3
        * (
            1
            + tf.cos(1 * (2 / 3) * 2 * np.pi)
            + tf.cos(2 * (2 / 3) * 2 * np.pi)
            + tf.sin(1 * (2 / 3) * 2 * np.pi)
            + tf.sin(2 * (2 / 3) * 2 * np.pi)
        ).numpy()
    )

        layer_test(
            SeasonalityBlock,
            kwargs={
                "input_width": input_width,
                "label_width": label_width,
                "n_neurons": n_neurons,
                "periods": periods,
                "back_periods": back_periods,
                "forecast_fourier_order": forecast_fourier_order,
                "backcast_fourier_order": backcast_fourier_order,
                "drop_rate": drop_rate,
                "weights": seasonality_weights,
            },
            input_dtype=floatx(),
            input_shape=(2, 3),
            expected_output_shape=((None, 2), (None, 3)),
            expected_output_dtype=[floatx(), floatx()],
            expected_output=[
                tf.constant([6.0, 0.0, 6.0, 0.0], shape=(2, 2)),
                tf.constant(
                    [9.0, y1, y2, 9.0, y1, y2], shape=(2, 3)
                ),
            ],
            custom_objects={"SeasonalityBlock": SeasonalityBlock},
        )

        periods = [1, 2]
        back_periods = [2, 3]

        forecast_fourier_order = [1, 2]
        backcast_fourier_order = [2, 3]

        forecast_neurons = tf.reduce_sum(2 * periods)
        backcast_neurons = tf.reduce_sum(2 * back_periods)

        seasonality_weights[-4:] = [
            np.ones(shape=(n_neurons, forecast_neurons)),
            np.ones(shape=(n_neurons, backcast_neurons)),
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
            SeasonalityBlock,
            kwargs={
                "input_width": input_width,
                "label_width": label_width,
                "n_neurons": n_neurons,
                "periods": periods,
                "back_periods": back_periods,
                "forecast_fourier_order": forecast_fourier_order,
                "backcast_fourier_order": backcast_fourier_order,
                "drop_rate": drop_rate,
                "weights": seasonality_weights,
            },
            input_dtype=floatx(),
            input_shape=(2, 3),
            expected_output_shape=((None, 2), (None, 3)),
            expected_output_dtype=[floatx(), floatx()],
            expected_output=[
                tf.constant([9.0, 3.0, 9.0, 3.0], shape=(2, 2)),
                tf.constant([15.0, y1, y2, 15.0, y1, y2], shape=(2, 3)),
            ],
            custom_objects={"SeasonalityBlock": SeasonalityBlock},
        )

    @parameterized.parameters(
        [
         (1, 2, 5, 5, 3, 0.)
        ]
    )
    def test_genericblock(
        self,
        input_width,
        label_width,
        forecast_neurons,
        backcast_neurons,
        n_neurons,
        drop_rate):

        layer_test(
            GenericBlock,
            kwargs={
                "input_width": input_width,
                "label_width": label_width,
                "n_neurons": n_neurons,
                "forecast_neurons": forecast_neurons,
                "backcast_neurons": backcast_neurons,
                "drop_rate": drop_rate,
            },
            input_dtype=floatx(),
            input_shape=(2, 2),
            expected_output_shape=((None, 1), (None, 2)),
            expected_output_dtype=[floatx(), floatx()],
            expected_output=None,
            custom_objects={"GenericBlock": GenericBlock},
        )