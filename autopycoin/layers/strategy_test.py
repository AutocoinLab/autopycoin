# pylint: skip-file

"""
Unit test for training.
"""

from absl.testing import parameterized

import tensorflow as tf
from tensorflow.python.keras import keras_parameterized
from tensorflow.keras.backend import floatx

from autopycoin.layers.strategy import UniVariate

from ..utils import layer_test


@keras_parameterized.run_all_keras_modes
class StrategyTest(tf.test.TestCase, parameterized.TestCase):
    """
    Unit tests for the nbeats model.
    """

    @parameterized.parameters(
        [  # BaseBlock attributes test
            (True, True, (None, 5, 2), (2, None, 5)),
            (True, False, (None, 5, 2), (None, 5, 2)),
            (
                False,
                True,
                (2, None, 5),
                (
                    None,
                    5,
                    None,
                ),  # In functional API we don't take into account the first dim (TODO: need to handle that)
            ),
            (False, False, (None, 5, 2), (None, 5, 2)),
        ]
    )
    def test_univariate(
        self, last_to_first, is_multivariate, input_shape, expected_output_shape
    ):

        layer_test(
            UniVariate,
            kwargs={"last_to_first": last_to_first, "is_multivariate": is_multivariate},
            input_shape=input_shape,
            expected_output_shape=[expected_output_shape],
            custom_objects={"UniVariate": UniVariate},
            validate_training=False,
        )
