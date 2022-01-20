# pylint: skip-file

"""
Customized Tools from keras API.
"""

import numpy as np
import threading
import pandas as pd

import tensorflow as tf
from tensorflow.compat.v1 import Dimension
from tensorflow.keras import layers, models
from tensorflow.python.eager import context
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect

from .. import losses


def dtype(obj):
    return obj.dtype.base_dtype.name


def string_test(actual, expected):
    np.testing.assert_array_equal(actual, expected)


def numeric_test(actual, expected):
    np.testing.assert_allclose(
        tf.round(actual, 3), tf.round(expected, 3), rtol=1e-3, atol=1e-5
    )


def layer_test(
    layer_cls,
    kwargs=None,
    input_shape=None,
    input_dtype=None,
    input_data=None,
    expected_output=None,
    expected_output_dtype=None,
    expected_output_shape=None,
    validate_training=True,
    adapt_data=None,
    custom_objects=None,
    test_harness=None,
    supports_masking=None,
):
    """
    Test routine for a layer with a single input and multiple output.

    Sequential API is not tested as it doesn't allow multiple output.

    Parameters
    ----------
    layer_cls: Layer class object.
    kwargs: Optional dictionary of keyword arguments for instantiating the
        layer.
    input_shape: Input shape tuple.
    input_dtype: Data type of the input data.
    input_data: Numpy array of input data.
    expected_output: Numpy array of the expected output.
    expected_output_dtype: Data type expected for the output.
    expected_output_shape: Shape tuple for the expected shape of the output.
    validate_training: Whether to attempt to validate training on this layer.
        This might be set to False for non-differentiable layers that output
        string or integer values.
    adapt_data: Optional data for an 'adapt' call. If None, adapt() will not
        be tested for this layer. This is only relevant for PreprocessingLayers.
    custom_objects: Optional dictionary mapping name strings to custom objects
        in the layer class. This is helpful for testing custom layers.
        test_harness: The Tensorflow test, if any, that this function is being
        called in.
    supports_masking: Optional boolean to check the `supports_masking` property
        of the layer. If None, the check will not be performed.

    Returns
    -------
    The output data (Numpy array) returned by the layer, for additional
    checks to be done by the calling code.

    Raises
    ------
    ValueError: if `input_shape is None`.
    """

    if input_data is None:
        if input_shape is None:
            raise ValueError("input_shape is None")
        if not input_dtype:
            input_dtype = "float32"
        input_data_shape = list(input_shape)
        for i, e in enumerate(input_data_shape):
            if e is None:
                input_data_shape[i] = np.random.randint(1, 4)
        input_data = 10 * np.zeros(input_data_shape)
        if input_dtype[:5] == "float":
            input_data -= 0.5
        input_data = input_data.astype(input_dtype)
    elif input_shape is None:
        input_shape = input_data.shape
    if input_dtype is None:
        input_dtype = input_data.dtype

    # instantiation
    kwargs = kwargs or {}
    kwargs_no_weights = kwargs.copy()
    recover_weights = kwargs_no_weights.pop("weights", None)
    layer = layer_cls(**kwargs)
    layer_no_weights = layer_cls(**kwargs_no_weights)

    # Compare fixed weights with their expected values
    if recover_weights is not None:
        numeric_test(
            layer.non_trainable_weights, layer_no_weights.non_trainable_weights
        )

    if supports_masking is not None and layer.supports_masking != supports_masking:
        raise AssertionError(
            "When testing layer %s, the `supports_masking` property is %r"
            "but expected to be %r.\nFull kwargs: %s"
            % (layer_cls.__name__, layer.supports_masking, supports_masking, kwargs)
        )

    # Test adapt, if data was passed.
    if adapt_data is not None:
        layer.adapt(adapt_data)

    # test in functional API
    x = layers.Input(shape=input_shape[1:], dtype=input_dtype)
    y = layer(x)

    # test get_weights, set_weights at layer level
    weights = layer.get_weights()
    layer.set_weights(weights)

    # test and instantiation from weights
    if "weights" in tf_inspect.getargspec(layer_cls.__init__):
        kwargs["weights"] = weights
        layer = layer_cls(**kwargs)

    def create_list(tensor):
        return tensor if isinstance(tensor, (tuple, list)) else [tensor]

    # Allow multi-y test
    ys = create_list(y)

    expected_outputs = create_list(expected_output)

    expected_output_dtypes = (
        [input_dtype for _ in ys]
        if expected_output_dtype is None
        else expected_output_dtype
        if isinstance(expected_output_dtype, (tuple, list))
        else [expected_output_dtype]
    )

    expected_output_shapes = create_list(expected_output_shape)

    computed_output_shapes = layer.compute_output_shape(tf.TensorShape(input_shape))

    computed_output_shapes = create_list(computed_output_shapes)

    computed_output_signatures = layer.compute_output_signature(
        tf.TensorSpec(shape=input_shape, dtype=input_dtype)
    )
    computed_output_signatures = create_list(computed_output_signatures)

    for (
        idx,
        (
            y,
            expected_output,
            expected_output_dtype,
            expected_output_shape,
            computed_output_shape,
            computed_output_signature,
        ),
    ) in enumerate(
        zip(
            ys,
            expected_outputs,
            expected_output_dtypes,
            expected_output_shapes,
            computed_output_shapes,
            computed_output_signatures,
        )
    ):

        if tf.dtypes.as_dtype(expected_output_dtype) == tf.dtypes.string:
            if test_harness:
                assert_equal = test_harness.assertAllEqual
            else:
                assert_equal = string_test
        else:
            if test_harness:
                assert_equal = test_harness.assertAllClose
            else:
                assert_equal = numeric_test

        if dtype(y) != expected_output_dtype:
            raise AssertionError(
                "When testing layer %s, for input %s, found output "
                "dtype=%s but expected to find %s.\nFull kwargs: %s"
                % (layer_cls.__name__, x, dtype(y), expected_output_dtype, kwargs)
            )

        def assert_shapes_equal(expected, actual):
            """Asserts that the output shape from the layer matches the actual shape."""
            if len(expected) != len(actual):
                raise AssertionError(
                    "When testing layer %s, for input %s, found output_shape="
                    "%s but expected to find %s.\nFull kwargs: %s"
                    % (layer_cls.__name__, x, actual, expected, kwargs)
                )

            for expected_dim, actual_dim in zip(expected, actual):
                if isinstance(expected_dim, Dimension):
                    expected_dim = expected_dim.value
                if isinstance(actual_dim, Dimension):
                    actual_dim = actual_dim.value
                if expected_dim is not None and expected_dim != actual_dim:
                    raise AssertionError(
                        "When testing layer %s, for input %s, found output_shape="
                        "%s but expected to find %s.\nFull kwargs: %s"
                        % (layer_cls.__name__, x, actual, expected, kwargs)
                    )

        if expected_output_shape is not None:
            assert_shapes_equal(tf.TensorShape(expected_output_shape), y.shape)

        # check shape inference
        model = models.Model(x, y)
        if isinstance(layer, tf.keras.Model):
            actual_output = layer.predict(input_data)
        else:
            actual_output = model.predict(input_data)
        # Handle multiple outputs
        actual_output = (
            actual_output[idx] if isinstance(actual_output, tuple) else actual_output
        )
        actual_output_shape = actual_output.shape
        assert_shapes_equal(computed_output_shape, actual_output_shape)
        assert_shapes_equal(computed_output_signature.shape, actual_output_shape)
        if computed_output_signature.dtype != actual_output.dtype:
            raise AssertionError(
                "When testing layer %s, for input %s, found output_dtype="
                "%s but expected to find %s.\nFull kwargs: %s"
                % (
                    layer_cls.__name__,
                    x,
                    actual_output.dtype,
                    computed_output_signature.dtype,
                    kwargs,
                )
            )
        if expected_output is not None:
            assert_equal(actual_output, expected_output)

    # test serialization, weight setting at model level
    if not isinstance(layer, tf.keras.Model):
        model_config = model.get_config()
        recovered_model = models.Model.from_config(model_config, custom_objects)
        if model.weights:
            weights = model.get_weights()
            recovered_model.set_weights(weights)
            output = recovered_model.predict(input_data)
            assert_equal(output, actual_output)

    # test training mode (e.g. useful for dropout tests)
    # Rebuild the model to avoid the graph being reused between predict() and
    # See b/120160788 for more details. This should be mitigated after 2.0.
    if validate_training:
        model = models.Model(x, layer(x))
        if _thread_local_data.run_eagerly is not None:
            model.compile(
                "rmsprop",
                [
                    "mse",
                    losses.QuantileLossError([0.5]),
                    losses.SymetricMeanAbsolutePercentageError,
                ],
                weighted_metrics=["acc"],
                run_eagerly=should_run_eagerly(),
            )
        else:
            model.compile(
                "rmsprop",
                [
                    "mse",
                    losses.QuantileLossError([0.5]),
                    losses.SymetricMeanAbsolutePercentageError,
                ],
                weighted_metrics=["acc"],
            )
        model.train_on_batch(input_data, model.predict(input_data))

    if not isinstance(layer, tf.keras.Model):
        # test as first layer in Sequential API
        layer_config = layer.get_config()
        layer_config["batch_input_shape"] = input_shape
        layer = layer.__class__.from_config(layer_config)

        # Test adapt, if data was passed.
        if adapt_data is not None:
            layer.adapt(adapt_data)

    return actual_output


_thread_local_data = threading.local()
_thread_local_data.model_type = None
_thread_local_data.run_eagerly = None
_thread_local_data.saved_model_format = None
_thread_local_data.save_kwargs = None


@tf_contextlib.contextmanager
def run_eagerly_scope(value):
    """
    Provides a scope within which we compile models to run eagerly or not.

    The boolean gets restored to its original value upon exiting the scope.

    Parameters
    ----------
    value: Bool specifying if we should run models eagerly in the active test.
        Should be True or False.
    Yields:
        The provided value.
    """
    previous_value = _thread_local_data.run_eagerly
    try:
        _thread_local_data.run_eagerly = value
        yield value
    finally:
        # Restore model type to initial value.
        _thread_local_data.run_eagerly = previous_value


def should_run_eagerly():
    """Returns whether the models we are testing should be run eagerly."""
    if _thread_local_data.run_eagerly is None:
        raise ValueError(
            "Cannot call `should_run_eagerly()` outside of a "
            "`run_eagerly_scope()` or `run_all_keras_modes` "
            "decorator."
        )

    return _thread_local_data.run_eagerly and context.executing_eagerly()


def check_attributes(instance, cls, attributes, expected_values):
    for attr, expected_value in zip(attributes, expected_values):
        value = getattr(cls, attr)
        if isinstance(expected_value, dict):
            instance.assertDictEqual(
                value,
                expected_value,
                f"{attr} not equal to the expected value, got {value} != {expected_value}",
            )
        elif isinstance(expected_value, (list, tuple)):
            for v, el in zip(value, expected_value):
                instance.assertIsInstance(v, type(el))
        else:
            np.testing.assert_array_equal(
                value,
                expected_value,
                f"{attr} not equal to the expected value, got {value} != {expected_value}",
            )
