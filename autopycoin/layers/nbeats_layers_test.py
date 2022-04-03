# pylint: skip-file

import numpy as np
from absl.testing import parameterized

from tensorflow.python.keras import keras_parameterized
from tensorflow.keras.backend import floatx
import tensorflow as tf

from ..test_utils import layer_test, check_attributes
from . import GenericBlock, TrendBlock, SeasonalityBlock, BaseBlock


class ExampleBlock(BaseBlock):
    def __init__(
        self,
        label_width: int,
        n_neurons: int,
        drop_rate: float,
        g_trainable: bool = False,
        interpretable: bool = False,
        block_type: str = "BaseBlock",
    ):
        super().__init__(
            label_width=label_width,
            n_neurons=n_neurons,
            drop_rate=drop_rate,
            g_trainable=g_trainable,
            interpretable=interpretable,
            block_type=block_type,
        )

    def coefficient_factory(self, *args: list, **kwargs: dict):
        pass

    def get_coefficients(self, output_last_dim: int, branch_name: str):
        return tf.constant([0, 0, 0])


@keras_parameterized.run_all_keras_modes
class BlocksLayersTest(tf.test.TestCase, parameterized.TestCase):
    """
    Unit tests for the nbeats model.
    """

    @parameterized.parameters(
        [  # BaseBlock attributes test
            (
                (10, 16, 0.5),
                BaseBlock,
                [
                    "label_width",
                    "drop_rate",
                    "is_interpretable",
                    "is_g_trainable",
                    "block_type",
                ],
                [10, 0.5, False, False, "BaseBlock"],
            ),
            # TrendBlock attributes test
            (
                (1, 2, 3, 0.0),
                TrendBlock,
                [
                    "label_width",
                    "p_degree",
                    "drop_rate",
                    "is_interpretable",
                    "is_g_trainable",
                    "block_type",
                ],
                [1, 2, 0.0, True, False, "TrendBlock"],
            ),
            # SeasonalityBlock attributes test
            (
                (10, [10], [10], [10], [10], 16, 0.5),
                SeasonalityBlock,
                [
                    "label_width",
                    "forecast_periods",
                    "backcast_periods",
                    "forecast_fourier_order",
                    "backcast_fourier_order",
                    "drop_rate",
                    "is_interpretable",
                    "is_g_trainable",
                    "block_type",
                ],
                [10, [10], [10], [10], [10], 0.5, True, False, "SeasonalityBlock"],
            ),
            # GenericBlock attributes test
            (
                (10, 10, 10, 16, 0.5),
                GenericBlock,
                [
                    "label_width",
                    "g_forecast_neurons",
                    "g_backcast_neurons",
                    "drop_rate",
                    "is_interpretable",
                    "is_g_trainable",
                    "block_type",
                ],
                [10, 10, 10, 0.5, False, True, "GenericBlock"],
            ),
        ]
    )
    def test_attributes_and_property(self, args, cls, attributes, expected_values):

        if cls == BaseBlock:
            block = ExampleBlock(*args)
        else:
            block = cls(*args)

        check_attributes(self, block, attributes, expected_values)

    @parameterized.parameters(
        [  # BaseBlock attributes test
            (
                (10, None, None, None, None, 16, 0.5),
                SeasonalityBlock,
                [
                    "label_width",
                    "forecast_periods",
                    "backcast_periods",
                    "forecast_fourier_order",
                    "backcast_fourier_order",
                    "drop_rate",
                    "is_interpretable",
                    "is_g_trainable",
                    "block_type",
                ],
                [10, 5, 2, 5, 2, 0.5, True, False, "SeasonalityBlock"],
            ),
        ]
    )
    def test_attributes_and_property_after_build(
        self, args, cls, attributes, expected_values
    ):

        block = cls(*args)
        block(tf.constant([[0.0, 0.0, 0.0, 0.0]]))
        check_attributes(self, block, attributes, expected_values)

    @parameterized.parameters(
        [
            (
                (-10, 16, 0.5),
                BaseBlock,
                "`label_width` or its elements has to be greater or equal",
            ),
            (
                (10, -10, 10, 16, 0.5),
                GenericBlock,
                "`g_forecast_neurons` or its elements has to be greater or equal",
            ),
            (
                (10, 10, -10, 16, 0.5),
                GenericBlock,
                "`g_backcast_neurons` or its elements has to be greater or equal",
            ),
            (
                (10, -16, 0.5),
                BaseBlock,
                "`n_neurons` or its elements has to be greater or equal",
            ),
            (
                (10, 16, -0.5),
                BaseBlock,
                "`drop_rate` or its elements has to be between",
            ),
            (
                (10, 16, 0.5, False, False, "base"),
                BaseBlock,
                "`name` has to contain `Block`",
            ),
            (
                (10, -10, 16, 0.5),
                TrendBlock,
                "`p_degree` or its elements has to be greater or equal",
            ),
            (
                (10, [10, 10], [10], [10], [10], 16, 0.5),
                SeasonalityBlock,
                "`forecast_periods` and `forecast_fourier_order` are expected to have the same length",
            ),
            (
                (10, [10], [10, 10], [10], [10], 16, 0.5),
                SeasonalityBlock,
                "`backcast_periods` and `backcast_fourier_order` are expected to have the same length",
            ),
            (
                (10, [-10], [10], [10], [10], 16, 0.5),
                SeasonalityBlock,
                "`forecast_periods` or its elements has to be greater or equal",
            ),
            (
                (10, [10], [-10], [10], [10], 16, 0.5),
                SeasonalityBlock,
                "`backcast_periods` or its elements has to be greater or equal",
            ),
        ]
    )
    def test_raises_error(self, args, cls, error):

        with self.assertRaisesRegexp(ValueError, error):

            with self.assertRaisesRegexp(AssertionError, error):
                if cls == BaseBlock:
                    obj = ExampleBlock(*args)
                else:
                    obj = cls(*args)
                obj.build(tf.TensorShape((None, args[0])))

            raise ValueError(error)

    @parameterized.parameters(
        [(1, 2, 1, 3, 0.0), (1, 2, 1, 3, 0.01),]
    )
    def test_trendblock(self, label_width, input_width, p_degree, n_neurons, drop_rate):

        trend_weights = [
            np.zeros(shape=(input_width, n_neurons)),
            np.ones(shape=(1, n_neurons)),
            np.zeros(shape=(n_neurons, n_neurons)),
            np.ones(shape=(1, n_neurons,)),
            np.zeros(shape=(n_neurons, n_neurons)),
            np.ones(shape=(1, n_neurons,)),
            np.zeros(shape=(n_neurons, n_neurons)),
            np.ones(shape=(1, n_neurons,)),
            np.ones(shape=(n_neurons, p_degree + 1)),
            np.ones(shape=(n_neurons, p_degree + 1)),
            np.array([[1.0], [0.0]]),
            np.array([[1.0, 1.0], [0.0, 0.5]]),
        ]

        if drop_rate == 0.0:
            layer_test(
                TrendBlock,
                kwargs={
                    "label_width": label_width,
                    "p_degree": p_degree,
                    "n_neurons": n_neurons,
                    "drop_rate": drop_rate,
                    "weights": trend_weights,
                },
                input_dtype=floatx(),
                input_shape=(2, 2),
                expected_output_shape=(
                    tf.TensorShape((None, 2)),
                    tf.TensorShape((None, 1)),
                ),
                expected_output_dtype=[floatx(), floatx()],
                expected_output=[
                    tf.constant([3.0, 4.5, 3.0, 4.5], shape=(2, 2)),
                    tf.constant(3.0, shape=(2, 1)),
                ],
                custom_objects={"TrendBlock": TrendBlock},
            )

        elif drop_rate > 0:
            model = TrendBlock(
                label_width=label_width,
                p_degree=p_degree,
                n_neurons=30,
                drop_rate=drop_rate,
            )
            actual_1 = model(tf.constant([[1.0, 8.0], [1.0, 2.0]]))
            actual_2 = model(tf.constant([[1.0, 8.0], [1.0, 2.0]]))
            np.testing.assert_raises(
                AssertionError, np.testing.assert_array_equal, actual_1[1], actual_2[1]
            )

    @parameterized.parameters([(2, 3, [2], [3], [2], [3], 3, 0.0)])
    def test_seasonalityblock(
        self,
        label_width,
        input_width,
        forecast_periods,
        backcast_periods,
        forecast_fourier_order,
        backcast_fourier_order,
        n_neurons,
        drop_rate,
    ):

        g_forecast_neurons = tf.reduce_sum(2 * forecast_periods)
        g_backcast_neurons = tf.reduce_sum(2 * backcast_periods)

        seasonality_weights = [
            np.zeros(shape=(input_width, n_neurons)),
            np.ones(shape=(1, n_neurons,)),
            np.zeros(shape=(n_neurons, n_neurons)),
            np.ones(shape=(1, n_neurons,)),
            np.zeros(shape=(n_neurons, n_neurons)),
            np.ones(shape=(1, n_neurons,)),
            np.zeros(shape=(n_neurons, n_neurons)),
            np.ones(shape=(1, n_neurons,)),
            np.ones(shape=(n_neurons, g_forecast_neurons)),
            np.ones(shape=(n_neurons, g_backcast_neurons)),
            np.array([[1, 1], [1, tf.cos(np.pi)], [0, 0], [0, tf.sin(np.pi)]]),
            np.array(
                [
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
                "label_width": label_width,
                "forecast_periods": forecast_periods,
                "backcast_periods": backcast_periods,
                "forecast_fourier_order": forecast_fourier_order,
                "backcast_fourier_order": backcast_fourier_order,
                "n_neurons": n_neurons,
                "drop_rate": drop_rate,
                "weights": seasonality_weights,
            },
            input_dtype=floatx(),
            input_shape=(2, 3),
            expected_output_shape=(
                tf.TensorShape((None, 3)),
                tf.TensorShape((None, 2)),
            ),
            expected_output_dtype=[floatx(), floatx()],
            expected_output=[
                tf.constant([9.0, y1, y2, 9.0, y1, y2], shape=(2, 3)),
                tf.constant([6.0, 0.0, 6.0, 0.0], shape=(2, 2)),
            ],
            custom_objects={"SeasonalityBlock": SeasonalityBlock},
        )

        forecast_periods = [1, 2]
        backcast_periods = [2, 3]

        forecast_fourier_order = [1, 2]
        backcast_fourier_order = [2, 3]

        g_forecast_neurons = tf.reduce_sum(2 * forecast_periods)
        g_backcast_neurons = tf.reduce_sum(2 * backcast_periods)

        seasonality_weights[-4:] = [
            np.ones(shape=(n_neurons, g_forecast_neurons)),
            np.ones(shape=(n_neurons, g_backcast_neurons)),
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
                "label_width": label_width,
                "forecast_periods": forecast_periods,
                "backcast_periods": backcast_periods,
                "forecast_fourier_order": forecast_fourier_order,
                "backcast_fourier_order": backcast_fourier_order,
                "n_neurons": n_neurons,
                "drop_rate": drop_rate,
                "weights": seasonality_weights,
            },
            input_dtype=floatx(),
            input_shape=(2, 3),
            expected_output_shape=(
                tf.TensorShape((None, 3)),
                tf.TensorShape((None, 2)),
            ),
            expected_output_dtype=[floatx(), floatx()],
            expected_output=[
                tf.constant([15.0, y1, y2, 15.0, y1, y2], shape=(2, 3)),
                tf.constant([9.0, 3.0, 9.0, 3.0], shape=(2, 2)),
            ],
            custom_objects={"SeasonalityBlock": SeasonalityBlock},
        )

    @parameterized.parameters([(1, 5, 5, 3, 0.0)])
    def test_genericblock(
        self, label_width, g_forecast_neurons, g_backcast_neurons, n_neurons, drop_rate,
    ):

        layer_test(
            GenericBlock,
            kwargs={
                "label_width": label_width,
                "g_forecast_neurons": g_forecast_neurons,
                "g_backcast_neurons": g_backcast_neurons,
                "n_neurons": n_neurons,
                "drop_rate": drop_rate,
            },
            input_dtype=floatx(),
            input_shape=(2, 2),
            expected_output_shape=(
                tf.TensorShape((None, 2)),
                tf.TensorShape((None, 1)),
            ),
            expected_output_dtype=[floatx(), floatx()],
            expected_output=None,
            custom_objects={"GenericBlock": GenericBlock},
        )
