# pylint: skip-file

"""
Test for nbeats model
"""

import numpy as np
from absl.testing import parameterized

from tensorflow.python.keras import keras_parameterized
from tensorflow.keras.backend import floatx
import tensorflow as tf

from autopycoin.dataset.generator import WindowGenerator

from ..models import PoolNBEATS

from ..test_utils import layer_test, check_attributes
from ..losses import QuantileLossError
from ..layers.nbeats_layers import GenericBlock, TrendBlock, SeasonalityBlock
from . import (
    create_interpretable_nbeats,
    create_generic_nbeats,
    NBEATS,
    Stack,
)
from ..data import random_ts


def check_shape(model, inputs, shape1, shape2):
    outputs_predict = model.predict(inputs)
    outputs_call = model(inputs)

    for output in [outputs_predict, outputs_call]:
        assert output[0].shape == shape1
        assert output[1].shape == shape2


def trend_weights(n_neurons, p_degree):
    return [
        np.zeros(shape=(3, 3)),
        np.ones(shape=(1, n_neurons,)),
        np.zeros(shape=(n_neurons, n_neurons)),
        np.ones(shape=(1, n_neurons,)),
        np.zeros(shape=(n_neurons, n_neurons)),
        np.ones(shape=(1, n_neurons,)),
        np.zeros(shape=(n_neurons, n_neurons)),
        np.ones(shape=(1, n_neurons,)),
        np.ones(shape=(n_neurons, p_degree + 1)),
        np.ones(shape=(n_neurons, p_degree + 1)),
        np.array([[1.0, 1.0], [0.0, 0.5]]),
        np.array([[1.0, 1.0, 1.0], [0.0, 1 / 3, 2 / 3]]),
    ]


def seasonality_weights(input_width, n_neurons, forecast_neurons, backcast_neurons):
    return [
        np.zeros(shape=(input_width, n_neurons)),
        np.ones(shape=(1, n_neurons,)),
        np.zeros(shape=(n_neurons, n_neurons)),
        np.ones(shape=(1, n_neurons,)),
        np.zeros(shape=(n_neurons, n_neurons)),
        np.ones(shape=(1, n_neurons,)),
        np.zeros(shape=(n_neurons, n_neurons)),
        np.ones(shape=(1, n_neurons,)),
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

trend_args = (1, 1, 3, 0.0)


@keras_parameterized.run_all_keras_modes
class NBEATSLayersTest(tf.test.TestCase, parameterized.TestCase):
    """
    Unit tests for the nbeats model.
    """

    @parameterized.parameters(
        [  # Stack attributes test
            (
                [(TrendBlock(*trend_args), TrendBlock(*trend_args))],
                Stack,
                ["blocks", "stack_type", "is_interpretable", "label_width"],
                [
                    (TrendBlock(*trend_args), TrendBlock(*trend_args),),
                    "TrendStack",
                    True,
                    1,
                ],
            ),
            # Stack attributes test
            (
                [(GenericBlock(*(1, 2, 2, 3, 0.0)), TrendBlock(*trend_args))],
                Stack,
                ["blocks", "stack_type", "is_interpretable", "label_width"],
                [
                    (GenericBlock(*(1, 2, 2, 3, 0.0)), TrendBlock(*trend_args),),
                    "CustomStack",
                    False,
                    1,
                ],
            ),
            # NBEATS attributes test
            (
                [[Stack([TrendBlock(*trend_args), TrendBlock(*trend_args),])]],
                NBEATS,
                ["stacks", "is_interpretable", "nbeats_type", "label_width"],
                [
                    [Stack([TrendBlock(*trend_args), TrendBlock(*trend_args),])],
                    True,
                    "InterpretableNbeats",
                    1,
                ],
            ),
            (
                [[Stack([GenericBlock(*(1, 2, 2, 3, 0.0)), TrendBlock(*trend_args),])]],
                NBEATS,
                ["stacks", "is_interpretable", "nbeats_type", "label_width"],
                [
                    [
                        Stack(
                            [GenericBlock(*(1, 2, 2, 3, 0.0)), TrendBlock(*trend_args),]
                        )
                    ],
                    False,
                    "Nbeats",
                    1,
                ],
            ),
            (
                [
                    1,
                    10,
                    lambda label_width: create_generic_nbeats(
                        label_width=label_width,
                        g_forecast_neurons=10,
                        g_backcast_neurons=10,
                        n_neurons=10,
                        n_blocks=2,
                        n_stacks=2,
                        drop_rate=0.0,
                        share=True,
                    ),
                ],
                PoolNBEATS,
                ["label_width", "n_models"],
                [1, 10],
            ),
        ]
    )
    def test_attributes_and_property(self, args, cls, attributes, expected_values):

        block = cls(*args)

        check_attributes(self, block, attributes, expected_values)

    @parameterized.parameters([(2, 3, 3, 1, [2], [3], [2], [3], 0.0,)])
    def test_stack(
        self,
        label_width,
        input_width,
        n_neurons,
        p_degree,
        forecast_periods,
        backcast_periods,
        forecast_fourier_order,
        backcast_fourier_order,
        drop_rate,
    ):

        forecast_neurons = tf.reduce_sum(2 * forecast_periods)
        backcast_neurons = tf.reduce_sum(2 * backcast_periods)

        s_weights = seasonality_weights(
            input_width, n_neurons, forecast_neurons, backcast_neurons
        )

        t_weights = trend_weights(n_neurons, p_degree)

        kwargs_1 = {
            "label_width": label_width,
            "p_degree": p_degree,
            "n_neurons": n_neurons,
            "drop_rate": drop_rate,
            "weights": t_weights,
        }

        kwargs_2 = {
            "label_width": label_width,
            "forecast_periods": forecast_periods,
            "backcast_periods": backcast_periods,
            "forecast_fourier_order": forecast_fourier_order,
            "backcast_fourier_order": backcast_fourier_order,
            "n_neurons": n_neurons,
            "drop_rate": drop_rate,
            "weights": s_weights,
        }

        blocks = [TrendBlock(**kwargs_1), SeasonalityBlock(**kwargs_2)]

        layer_test(
            Stack,
            kwargs={"blocks": blocks},
            input_dtype=floatx(),
            input_shape=(2, 3),
            expected_output_shape=(
                tf.TensorShape((None, 3)),
                tf.TensorShape((None, 2)),
            ),
            expected_output_dtype=[floatx(), floatx()],
            expected_output=[
                -1 * tf.constant([12.5, 4.5, 5.5, 12.5, 4.5, 5.5,], shape=(2, 3),),
                tf.constant([9.0, 4.5, 9.0, 4.5], shape=(2, 2)),
            ],
            custom_objects={"Stack": Stack},
        )

    @parameterized.parameters([(2, 3, 3, 1, [2], [3], [2], [3], 0.0,)])
    def test_nbeats(
        self,
        label_width,
        input_width,
        n_neurons,
        p_degree,
        forecast_periods,
        backcast_periods,
        forecast_fourier_order,
        backcast_fourier_order,
        drop_rate,
    ):

        forecast_neurons = tf.reduce_sum(2 * forecast_periods)
        backcast_neurons = tf.reduce_sum(2 * backcast_periods)

        s_weights = seasonality_weights(
            input_width, n_neurons, forecast_neurons, backcast_neurons
        )

        t_weights = trend_weights(n_neurons, p_degree)

        kwargs_1 = {
            "label_width": label_width,
            "p_degree": p_degree,
            "n_neurons": n_neurons,
            "drop_rate": drop_rate,
            "weights": t_weights,
        }

        kwargs_2 = {
            "label_width": label_width,
            "forecast_periods": forecast_periods,
            "backcast_periods": backcast_periods,
            "forecast_fourier_order": forecast_fourier_order,
            "backcast_fourier_order": backcast_fourier_order,
            "n_neurons": n_neurons,
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
            expected_output_shape=(
                tf.TensorShape((None, 3)),
                tf.TensorShape((None, 2)),
            ),
            expected_output_dtype=[floatx(), floatx()],
            expected_output=[
                tf.constant([24.0, 8.0, 10.0, 24.0, 8.0, 10.0], shape=(2, 3)),
                tf.constant([18.0, 9.0, 18.0, 9.0], shape=(2, 2)),
            ],
        )

    @parameterized.parameters([(2, 3, 3, 1, [2], [3], [2], [3], 0.0,)])
    def test_nbeats_methods(
        self,
        label_width,
        input_width,
        n_neurons,
        p_degree,
        forecast_periods,
        backcast_periods,
        forecast_fourier_order,
        backcast_fourier_order,
        drop_rate,
    ):

        forecast_neurons = tf.reduce_sum(2 * forecast_periods)
        backcast_neurons = tf.reduce_sum(2 * backcast_periods)

        s_weights = seasonality_weights(
            input_width, n_neurons, forecast_neurons, backcast_neurons
        )

        t_weights = trend_weights(n_neurons, p_degree)

        kwargs_1 = {
            "label_width": label_width,
            "p_degree": p_degree,
            "n_neurons": n_neurons,
            "drop_rate": drop_rate,
            "weights": t_weights,
        }

        kwargs_2 = {
            "label_width": label_width,
            "forecast_periods": forecast_periods,
            "backcast_periods": backcast_periods,
            "forecast_fourier_order": forecast_fourier_order,
            "backcast_fourier_order": backcast_fourier_order,
            "n_neurons": n_neurons,
            "drop_rate": drop_rate,
            "weights": s_weights,
        }

        blocks_1 = Stack([TrendBlock(**kwargs_1), TrendBlock(**kwargs_1)])
        blocks_2 = Stack([SeasonalityBlock(**kwargs_2), SeasonalityBlock(**kwargs_2)])
        stacks = [blocks_1, blocks_2]

        model = NBEATS(stacks)
        inputs = tf.constant([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        seasonality = model.seasonality(inputs)
        trend = model.trend(inputs)
        self.assertNotEmpty(seasonality[0])
        self.assertNotEmpty(seasonality[1])
        self.assertNotEmpty(trend[0])
        self.assertNotEmpty(trend[1])

    @parameterized.parameters([(2, 3, 3, 1, [2], [3], [2], [3], 0.0,)])
    def test_nbeats_raises_error(
        self,
        label_width,
        input_width,
        n_neurons,
        p_degree,
        forecast_periods,
        backcast_periods,
        forecast_fourier_order,
        backcast_fourier_order,
        drop_rate,
    ):

        forecast_neurons = tf.reduce_sum(2 * forecast_periods)
        backcast_neurons = tf.reduce_sum(2 * backcast_periods)

        s_weights = seasonality_weights(
            input_width, n_neurons, forecast_neurons, backcast_neurons
        )

        t_weights = trend_weights(n_neurons, p_degree)

        kwargs_1 = {
            "label_width": label_width,
            "p_degree": p_degree,
            "n_neurons": n_neurons,
            "drop_rate": drop_rate,
            "weights": t_weights,
        }

        kwargs_2 = {
            "label_width": label_width,
            "forecast_periods": forecast_periods,
            "backcast_periods": backcast_periods,
            "forecast_fourier_order": forecast_fourier_order,
            "backcast_fourier_order": backcast_fourier_order,
            "n_neurons": n_neurons,
            "drop_rate": drop_rate,
            "weights": s_weights,
        }

        blocks_1 = Stack([TrendBlock(**kwargs_1), SeasonalityBlock(**kwargs_2)])
        blocks_2 = Stack([SeasonalityBlock(**kwargs_2), SeasonalityBlock(**kwargs_2)])
        stacks = [blocks_1, blocks_2]

        model = NBEATS(stacks)
        inputs = tf.constant([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        with self.assertRaisesRegexp(
            AttributeError, f"""The first stack has to be a `TrendStack`"""
        ):
            model.seasonality(inputs)
        with self.assertRaisesRegexp(AttributeError, f"""No `TrendStack` defined."""):
            model.trend(inputs)

        blocks_1 = Stack([TrendBlock(**kwargs_1), TrendBlock(**kwargs_1)])
        stacks = [blocks_1]

        model = NBEATS(stacks)
        with self.assertRaisesRegexp(
            AttributeError, f"""No `SeasonalityStack` defined"""
        ):
            model.seasonality(inputs)

        kwargs_3 = {
            "label_width": label_width,
            "n_neurons": n_neurons,
            "g_forecast_neurons": 5,
            "g_backcast_neurons": 5,
            "drop_rate": drop_rate,
        }

        blocks_3 = Stack([GenericBlock(**kwargs_3), GenericBlock(**kwargs_3)])

        stacks = [blocks_3, blocks_3]

        model = NBEATS(stacks)
        inputs = tf.constant([[0.0, 0.0], [0.0, 0.0]])

        with self.assertRaisesRegexp(
            AttributeError, "The first stack has to be a `TrendStack`"
        ):
            model.seasonality(inputs)
        with self.assertRaises(AttributeError):
            model.trend(inputs)

    @parameterized.parameters([(2, [2], [3], [2], [3], 1, 5, 5, 0.0, True)])
    def test_create_interpretable_nbeats(
        self,
        label_width,
        forecast_periods,
        backcast_periods,
        forecast_fourier_order,
        backcast_fourier_order,
        p_degree,
        trend_n_neurons,
        seasonality_n_neurons,
        drop_rate,
        share,
    ):

        create_model = lambda: create_interpretable_nbeats(
            label_width=label_width,
            forecast_periods=forecast_periods,
            backcast_periods=backcast_periods,
            forecast_fourier_order=forecast_fourier_order,
            backcast_fourier_order=backcast_fourier_order,
            p_degree=p_degree,
            trend_n_neurons=trend_n_neurons,
            seasonality_n_neurons=seasonality_n_neurons,
            drop_rate=drop_rate,
            share=share,
        )

        model = create_model()

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

        # Compare output shape with expected
        check_shape(model, np.array([[1.0, 2.0, 3.0]]), (1, 3), (1, 2))

        model = create_model()

        check_shape(
            model,
            np.array([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]]),
            (1, 3, 2),
            (1, 2, 2),
        )

        model = create_model()

        # Compare output shape with expected when quantiles = 3
        model.compile(loss=["mse", QuantileLossError([0.1, 0.5, 0.9])])
        check_shape(model, np.array([[1.0, 2.0, 3.0]]), (1, 3), (1, 2, 3))

        model = create_model()

        model.compile(loss=["mse", QuantileLossError([0.1, 0.5, 0.9])])
        check_shape(
            model,
            np.array([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]]),
            (1, 3, 2),
            (1, 2, 2, 3),
        )

    @parameterized.parameters([(2, 5, 5, 5, 3, 2, 0.0, True)])
    def test_create_generic_nbeats(
        self,
        label_width,
        g_forecast_neurons,
        g_backcast_neurons,
        n_neurons,
        n_blocks,
        n_stacks,
        drop_rate,
        share,
    ):

        create_model = lambda: create_generic_nbeats(
            label_width=label_width,
            g_forecast_neurons=g_forecast_neurons,
            g_backcast_neurons=g_backcast_neurons,
            n_neurons=n_neurons,
            n_blocks=n_blocks,
            n_stacks=n_stacks,
            drop_rate=drop_rate,
            share=share,
        )

        model = create_model()

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
        check_shape(model, np.array([[1.0, 2.0, 3.0]]), (1, 3), (1, 2))

        model = create_model()

        check_shape(
            model,
            np.array([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]]),
            (1, 3, 2),
            (1, 2, 2),
        )

        model = create_model()

        # Compare output shape with expected when quantiles = 3
        model.compile(loss=["mse", QuantileLossError([0.1, 0.5, 0.9])])
        check_shape(model, np.array([[1.0, 2.0, 3.0]]), (1, 3), (1, 2, 3))

        model = create_model()

        model.compile(loss=["mse", QuantileLossError([0.1, 0.5, 0.9])])
        check_shape(
            model,
            np.array([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]]),
            (1, 3, 2),
            (1, 2, 2, 3),
        )

    @parameterized.parameters(
        [
            (
                2,
                tf.reduce_mean,
                [(1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2, 2), (1, 2, 2)],
                (1, 2),
            ),
            (
                2,
                lambda x, axis: tf.identity(x),
                [(1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2, 2), (1, 2, 2)],
                (10, 1, 2),
            ),
        ]
    )
    def test_pool_nbeats_as_model(
        self, label_width, fn_agg, shape, shape2,
    ):

        data = random_ts(n_steps=30, n_variables=2)

        w = WindowGenerator(
            input_width=6,
            label_width=label_width,
            shift=label_width,
            valid_size=0,
            test_size=0,
            flat=False,
            preprocessing=lambda x, y: (x, (x, y)),
        )
        w.from_array(data, input_columns=[0, 1], label_columns=[0, 1])

        model = [
            create_generic_nbeats(
                label_width=label_width,
                g_forecast_neurons=1,
                g_backcast_neurons=1,
                n_neurons=1,
                n_blocks=1,
                n_stacks=1,
            )
            for _ in range(5)
        ]

        # Check quantiles model
        qloss = QuantileLossError([0.2, 0.5])

        model = PoolNBEATS(n_models=10, nbeats_models=model, fn_agg=fn_agg, seed=5)

        model.compile(
            tf.keras.optimizers.Adam(
                learning_rate=0.02,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07,
                amsgrad=True,
                name="Adam",
            ),
            loss=["mse", "mae", "mape", qloss],
            metrics=["mae"],
        )

        # Issue 13
        model.fit(w.train, epochs=1)
        # Check validation_data
        model.fit(w.train, validation_data=w.valid, epochs=1)
        # Check tensor inputs
        for x, y, x_val, y_val in zip(w.train.take(1), w.valid.take(1)):
            model.fit(x=x, y=y, validation_data=(x_val, y_val), epochs=1)

        output = model.predict(
            np.array(
                [
                    [
                        [1.0, 1.0],
                        [2.0, 2.0],
                        [3.0, 3.0],
                        [5.0, 5.0],
                        [6.0, 6.0],
                        [7.0, 7.0],
                    ]
                ]
            )
        )

        for o, s in zip(output[1], shape):
            self.assertEqual(o.shape, s)

        for loss in ["mse", "mae", "mape", qloss]:
            self.assertIn(loss, model.loss)

        # callable model
        nbeats = [
            lambda label_width: create_generic_nbeats(
                label_width=label_width,
                g_forecast_neurons=1,
                g_backcast_neurons=1,
                n_neurons=1,
                n_blocks=1,
                n_stacks=1,
            ),
            lambda label_width: create_interpretable_nbeats(
                label_width=label_width, trend_n_neurons=1, seasonality_n_neurons=1
            ),
        ]

        # Check randomness
        tf.random.set_seed(5)

        for i in range(2):
            model2 = PoolNBEATS(
                label_width=label_width,
                n_models=10,
                nbeats_models=nbeats,
                fn_agg=fn_agg,
                seed=5,
            )

            model2.compile(
                tf.keras.optimizers.Adam(
                    learning_rate=0.02,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-07,
                    amsgrad=True,
                    name="Adam",
                ),
                loss=["mse", "mae", "mape"],
                metrics=["mae"],
            )

            model2.build(tf.TensorShape((None, 6)))

            if i == 0:
                mask_random_v1 = model2._mask
                pool_losses_random_v1 = model2.compiled_loss._losses
                types_random_v1 = [m.nbeats_type for m in model2.models]

        self.assertAllEqual(model2._mask, mask_random_v1)
        self.assertAllEqual(model2.compiled_loss._losses, pool_losses_random_v1)
        self.assertAllEqual([m.nbeats_type for m in model2.models], types_random_v1)

        output = model2.predict(np.array([[1.0, 2.0, 3.0, 5.0, 6.0, 7.0]]))

        self.assertEqual(output[1].shape, shape2)

    def test_pool_nbeats_as_layer(self):

        nbeats = [
            lambda label_width: create_generic_nbeats(
                label_width=label_width,
                g_forecast_neurons=1,
                g_backcast_neurons=1,
                n_neurons=1,
                n_blocks=1,
                n_stacks=1,
            ),
            lambda label_width: create_interpretable_nbeats(
                label_width=label_width, trend_n_neurons=1, seasonality_n_neurons=1
            ),
        ]

        kwargs_2 = {"label_width": 3, "n_models": 3, "nbeats_models": nbeats, "seed": 0}

        layer_test(
            PoolNBEATS,
            kwargs=kwargs_2,
            input_dtype=floatx(),
            input_shape=(2, 10),
            expected_output_shape=(
                (tf.TensorShape((None, 10)), tf.TensorShape((None, 3))),
                (tf.TensorShape((None, 10)), tf.TensorShape((None, 3))),
                (tf.TensorShape((None, 10)), tf.TensorShape((None, 3))),
            ),
            expected_output_dtype=[
                floatx(),
                floatx(),
                floatx(),
                floatx(),
                floatx(),
                floatx(),
            ],
        )

    def test_pool_nbeats_raises_error(self):
        with self.assertRaisesRegexp(
            AssertionError,
            f"""When `models` are callable, `label_width` and `n_models` have to be integers.""",
        ):
            PoolNBEATS(n_models=10,)
