# pylint: skip-file

"""
Test for nbeats model
"""

import numpy as np
from absl.testing import parameterized

from tensorflow.python.keras import keras_parameterized
from tensorflow.keras.backend import floatx
import tensorflow as tf

from ..utils import layer_test, check_attributes
from ..losses import QuantileLossError
from . import (
    create_interpretable_nbeats,
    create_generic_nbeats,
    NBEATS,
    Stack,
    GenericBlock,
    TrendBlock,
    SeasonalityBlock,
)


def trend_weights(n_neurons, p_degree):
    return [
        np.zeros(shape=(3, 3)),
        np.ones(shape=(n_neurons,)),
        np.zeros(shape=(n_neurons, n_neurons)),
        np.ones(shape=(n_neurons,)),
        np.zeros(shape=(n_neurons, n_neurons)),
        np.ones(shape=(n_neurons,)),
        np.zeros(shape=(n_neurons, n_neurons)),
        np.ones(shape=(n_neurons,)),
        np.ones(shape=(n_neurons, p_degree)),
        np.ones(shape=(n_neurons, p_degree)),
        np.array([[1.0, 1.0], [0.0, 0.5]]),
        np.array([[1.0, 1.0, 1.0], [0.0, 1 / 3, 2 / 3]]),
    ]


def seasonality_weights(input_width, n_neurons, forecast_neurons, backcast_neurons):
    return [
        np.zeros(shape=(input_width, n_neurons)),
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


@keras_parameterized.run_all_keras_modes
class NBEATSLayersTest(tf.test.TestCase, parameterized.TestCase):
    """
    Unit tests for the nbeats model.
    """

    @parameterized.parameters(
        [  # Stack attributes test
            (
                [(TrendBlock(*(1, 2, 2, 3, 0.0)), TrendBlock(*(1, 2, 2, 3, 0.0)))],
                Stack,
                ["blocks", "stack_type", "is_interpretable"],
                [
                    (
                        TrendBlock(*(1, 2, 2, 3, 0.0)),
                        TrendBlock(*(1, 2, 2, 3, 0.0)),
                    ),
                    "TrendStack",
                    True,
                ],
            ),
            # Stack attributes test
            (
                [(GenericBlock(*(1, 2, 2, 2, 3, 0.0)), TrendBlock(*(1, 2, 2, 3, 0.0)))],
                Stack,
                ["blocks", "stack_type", "is_interpretable"],
                [
                    (
                        GenericBlock(*(1, 2, 2, 2, 3, 0.0)),
                        TrendBlock(*(1, 2, 2, 3, 0.0)),
                    ),
                    "CustomStack",
                    False,
                ],
            ),
            # NBEATS attributes test
            (
                [
                    [
                        Stack(
                            [
                                TrendBlock(*(1, 2, 2, 3, 0.0)),
                                TrendBlock(*(1, 2, 2, 3, 0.0)),
                            ]
                        )
                    ]
                ],
                NBEATS,
                ["stacks", "is_interpretable", "nbeats_type"],
                [
                    [
                        Stack(
                            [
                                TrendBlock(*(1, 2, 2, 3, 0.0)),
                                TrendBlock(*(1, 2, 2, 3, 0.0)),
                            ]
                        )
                    ],
                    True,
                    "InterpretableNbeats",
                ],
            ),
            (
                [
                    [
                        Stack(
                            [
                                GenericBlock(*(1, 2, 2, 2, 3, 0.0)),
                                TrendBlock(*(1, 2, 2, 3, 0.0)),
                            ]
                        )
                    ]
                ],
                NBEATS,
                ["stacks", "is_interpretable", "nbeats_type"],
                [
                    [
                        Stack(
                            [
                                GenericBlock(*(1, 2, 2, 2, 3, 0.0)),
                                TrendBlock(*(1, 2, 2, 3, 0.0)),
                            ]
                        )
                    ],
                    False,
                    "Nbeats",
                ],
            ),
        ]
    )
    def test_attributes_and_property(self, args, cls, attributes, expected_values):

        block = cls(*args)

        check_attributes(self, block, attributes, expected_values)

    @parameterized.parameters(
        [
            (
                2,
                3,
                3,
                2,
                [2],
                [3],
                [2],
                [3],
                0.0,
            )
        ]
    )
    def test_stack(
        self,
        label_width,
        input_width,
        n_neurons,
        p_degree,
        periods,
        back_periods,
        forecast_fourier_order,
        backcast_fourier_order,
        drop_rate,
    ):

        forecast_neurons = tf.reduce_sum(2 * periods)
        backcast_neurons = tf.reduce_sum(2 * back_periods)

        s_weights = seasonality_weights(
            input_width, n_neurons, forecast_neurons, backcast_neurons
        )

        t_weights = trend_weights(n_neurons, p_degree)

        kwargs_1 = {
            "label_width": label_width,
            "input_width": input_width,
            "p_degree": p_degree,
            "n_neurons": n_neurons,
            "drop_rate": drop_rate,
            "weights": t_weights,
        }

        kwargs_2 = {
            "label_width": label_width,
            "input_width": input_width,
            "n_neurons": n_neurons,
            "periods": periods,
            "back_periods": back_periods,
            "forecast_fourier_order": forecast_fourier_order,
            "backcast_fourier_order": backcast_fourier_order,
            "drop_rate": drop_rate,
            "weights": s_weights,
        }

        blocks = [TrendBlock(**kwargs_1), SeasonalityBlock(**kwargs_2)]

        layer_test(
            Stack,
            kwargs={"blocks": blocks},
            input_dtype=floatx(),
            input_shape=(2, 3),
            expected_output_shape=((None, 2), (None, 3)),
            expected_output_dtype=[floatx(), floatx()],
            expected_output=[
                tf.constant([9.0, 4.5, 9.0, 4.5], shape=(2, 2)),
                -1
                * tf.constant(
                    [
                        12.0,
                        y1 + 3 + 1,
                        y2 + 3 + 2,
                        12.0,
                        y1 + 3 + 1,
                        y2 + 3 + 2,
                    ],
                    shape=(2, 3),
                ),
            ],
            custom_objects={"Stack": Stack},
        )

    @parameterized.parameters(
        [
            (
                2,
                3,
                3,
                2,
                [2],
                [3],
                [2],
                [3],
                0.0,
            )
        ]
    )
    def test_nbeats(
        self,
        label_width,
        input_width,
        n_neurons,
        p_degree,
        periods,
        back_periods,
        forecast_fourier_order,
        backcast_fourier_order,
        drop_rate,
    ):

        forecast_neurons = tf.reduce_sum(2 * periods)
        backcast_neurons = tf.reduce_sum(2 * back_periods)

        s_weights = seasonality_weights(
            input_width, n_neurons, forecast_neurons, backcast_neurons
        )

        t_weights = trend_weights(n_neurons, p_degree)

        kwargs_1 = {
            "label_width": label_width,
            "input_width": input_width,
            "p_degree": p_degree,
            "n_neurons": n_neurons,
            "drop_rate": drop_rate,
            "weights": t_weights,
        }

        kwargs_2 = {
            "label_width": label_width,
            "input_width": input_width,
            "n_neurons": n_neurons,
            "periods": periods,
            "back_periods": back_periods,
            "forecast_fourier_order": forecast_fourier_order,
            "backcast_fourier_order": backcast_fourier_order,
            "drop_rate": drop_rate,
            "weights": s_weights,
        }

        blocks_1 = Stack([TrendBlock(**kwargs_1), SeasonalityBlock(**kwargs_2)])
        blocks_2 = Stack([TrendBlock(**kwargs_1), SeasonalityBlock(**kwargs_2)])
        stacks = [blocks_1, blocks_2]

        layer_test(
            NBEATS,
            kwargs={"stacks": stacks},
            input_dtype=floatx(),
            input_shape=(2, 3),
            expected_output_shape=((None, 2), (None, 3)),
            expected_output_dtype=[floatx(), floatx()],
            expected_output=tf.constant([18.0, 9.0, 18.0, 9.0], shape=(2, 2)),
        )

    @parameterized.parameters(
        [
            (
                2,
                3,
                3,
                2,
                [2],
                [3],
                [2],
                [3],
                0.0,
            )
        ]
    )
    def test_nbeats_attributes(
        self,
        label_width,
        input_width,
        n_neurons,
        p_degree,
        periods,
        back_periods,
        forecast_fourier_order,
        backcast_fourier_order,
        drop_rate,
    ):

        forecast_neurons = tf.reduce_sum(2 * periods)
        backcast_neurons = tf.reduce_sum(2 * back_periods)

        s_weights = seasonality_weights(
            input_width, n_neurons, forecast_neurons, backcast_neurons
        )

        t_weights = trend_weights(n_neurons, p_degree)

        kwargs_1 = {
            "label_width": label_width,
            "input_width": input_width,
            "p_degree": p_degree,
            "n_neurons": n_neurons,
            "drop_rate": drop_rate,
            "weights": t_weights,
        }

        kwargs_2 = {
            "label_width": label_width,
            "input_width": input_width,
            "n_neurons": n_neurons,
            "periods": periods,
            "back_periods": back_periods,
            "forecast_fourier_order": forecast_fourier_order,
            "backcast_fourier_order": backcast_fourier_order,
            "drop_rate": drop_rate,
            "weights": s_weights,
        }

        blocks_1 = Stack([TrendBlock(**kwargs_1), TrendBlock(**kwargs_1)])
        blocks_2 = Stack([SeasonalityBlock(**kwargs_2), SeasonalityBlock(**kwargs_2)])
        stacks = [blocks_1, blocks_2]

        model = NBEATS(stacks)
        inputs = tf.constant([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        self.assertNotEmpty(model.seasonality(inputs))
        self.assertNotEmpty(model.trend(inputs))

        kwargs_3 = {
            "label_width": label_width,
            "input_width": 2,
            "n_neurons": n_neurons,
            "forecast_neurons": 5,
            "backcast_neurons": 5,
            "drop_rate": drop_rate,
        }

        blocks_3 = Stack([GenericBlock(**kwargs_3), GenericBlock(**kwargs_3)])

        stacks = [blocks_3, blocks_3]

        model = NBEATS(stacks)
        inputs = tf.constant([[0.0, 0.0], [0.0, 0.0]])

        with self.assertRaises(AttributeError):
            model.seasonality(inputs)
            model.trend(inputs)

    @parameterized.parameters([(2, 3, [2], [3], [2], [3], 2, 5, 5, 0.0, True)])
    def test_create_interpretable_nbeats(
        self,
        label_width,
        input_width,
        periods,
        back_periods,
        forecast_fourier_order,
        backcast_fourier_order,
        p_degree,
        trend_n_neurons,
        seasonality_n_neurons,
        drop_rate,
        share,
    ):

        model = create_interpretable_nbeats(
            label_width=label_width,
            input_width=input_width,
            periods=periods,
            back_periods=back_periods,
            forecast_fourier_order=forecast_fourier_order,
            backcast_fourier_order=backcast_fourier_order,
            p_degree=p_degree,
            trend_n_neurons=trend_n_neurons,
            seasonality_n_neurons=seasonality_n_neurons,
            drop_rate=drop_rate,
            share=share,
        )

        self.assertIsInstance(model, NBEATS)
        trend_stack = model.stacks[0]
        seasonality_stack = model.stacks[1]
        self.assertEqual(trend_stack.stack_type, "TrendStack")
        self.assertEqual(seasonality_stack.stack_type, "SeasonalityStack")

        # Test if interpretable model is composed by trend and seasonality blocks only
        for block in trend_stack.blocks:
            self.assertIsInstance(block, TrendBlock)
        for block in seasonality_stack.blocks:
            self.assertIsInstance(block, SeasonalityBlock)

        # Compare weights values, expected to be equals over blocks because share=True
        self.assertEquals(
            trend_stack.blocks[0].get_weights(),
            trend_stack.blocks[1].get_weights(),
            trend_stack.blocks[2].get_weights(),
        )

        self.assertEquals(
            seasonality_stack.blocks[0].get_weights(),
            seasonality_stack.blocks[1].get_weights(),
            seasonality_stack.blocks[2].get_weights(),
        )

        # Compare output shape with expected shape
        outputs = model.predict(np.array([[1.0, 2.0, 3.0]]))
        self.assertEqual(outputs.shape, (1, 2))

        # Compare output shape with expected shape when quantiles = 3
        model.compile(loss=QuantileLossError([0.1, 0.5, 0.9]))
        outputs = model.predict(np.array([[1.0, 2.0, 3.0]]))
        self.assertEqual(outputs.shape, (3, 1, 2))

    @parameterized.parameters([(2, 3, 5, 5, 5, 3, 2, 0.0, True)])
    def test_create_generic_nbeats(
        self,
        label_width,
        input_width,
        forecast_neurons,
        backcast_neurons,
        n_neurons,
        n_blocks,
        n_stacks,
        drop_rate,
        share,
    ):

        model = create_generic_nbeats(
            label_width=label_width,
            input_width=input_width,
            forecast_neurons=forecast_neurons,
            backcast_neurons=backcast_neurons,
            n_neurons=n_neurons,
            n_blocks=n_blocks,
            n_stacks=n_stacks,
            drop_rate=drop_rate,
            share=share,
        )

        self.assertIsInstance(model, NBEATS)
        generic_stack_1 = model.stacks[0]
        generic_stack_2 = model.stacks[1]
        self.assertEqual(generic_stack_1.stack_type, "GenericStack")
        self.assertEqual(generic_stack_2.stack_type, "GenericStack")

        # Test if interpretable model is composed by trend and seasonality blocks only
        for block in generic_stack_1.blocks:
            self.assertIsInstance(block, GenericBlock)
        for block in generic_stack_2.blocks:
            self.assertIsInstance(block, GenericBlock)

        # Compare weights values, expected to be equals over blocks because share=True
        self.assertEquals(
            generic_stack_1.blocks[0].get_weights(),
            generic_stack_1.blocks[1].get_weights(),
            generic_stack_1.blocks[2].get_weights(),
        )

        self.assertEquals(
            generic_stack_2.blocks[0].get_weights(),
            generic_stack_2.blocks[1].get_weights(),
            generic_stack_2.blocks[2].get_weights(),
        )

        # Compare output shape with expected
        outputs = model.predict(np.array([[1.0, 2.0, 3.0]]))
        self.assertEqual(outputs.shape, (1, 2))

        # Compare output shape with expected when quantiles = 2
        model.compile(loss=QuantileLossError([0.1, 0.5, 0.9]))
        outputs = model.predict(np.array([[1.0, 2.0, 3.0]]))
        self.assertEqual(outputs.shape, (3, 1, 2))
