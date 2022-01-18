"""
Defines layers for time series analysis.
#TODO: doc and test unit
"""

import tensorflow as tf
from tensorflow.keras.layers import InputSpec

from . import Layer
from ..dataset import features, date_features

# TODO: Unit testing
class BaseStrategy(Layer):
    def build(self, input_shapes):
        """Build method from tensorflow."""

        input_shapes = (input_shapes,) if not isinstance(input_shapes, tuple) else input_shapes
        dtype = tf.as_dtype(self.dtype or tf.float32())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError(
                f"Unable to build `{self.name}` layer with "
                "non-floating point dtype %s" % (dtype,)
            )

        self.last_dims = []
        shapes = []
        self.input_rank = []
        self.time_steps = []
        for input_shape in input_shapes:
            shape = tf.TensorShape(input_shape)
            last_dim = tf.compat.dimension_value(input_shape[-1])
            if last_dim is None:
                raise ValueError(
                    f"The last dimension of the inputs"
                    f" should be defined. Found {last_dim}."
                )
            self.last_dims.append(last_dim)
            shapes.append(shape)
            self.input_rank.append(input_shape.rank)

            if input_shape.rank > 2:
                self.time_steps.append(shape[-2])
            else:
                self.time_steps.append(None)
        super().build(input_shapes)

class BaseStrategyMultivariate(BaseStrategy):
    def build(self, input_shape):
        """Build method from tensorflow."""

        # TODO: not filter on one tensor but verify every tensors
        if isinstance(input_shape, tuple):
            input_shape = input_shape[0]

        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if tf.rank(input_shape) < 3:
            raise ValueError(
                f"The shape of the inputs"
                f"should contains time steps. Found {input_shape}."
            )

        super().build(input_shape)
        self.input_spec = InputSpec(min_ndim=3, axes={-1: last_dim})


class UniVariate(BaseStrategy):
    def build(self, input_shape):
        super().build(input_shape)
        self._n_variates = 1
        if self.input_rank[0] > 2: # See BaseStrategy
            self._n_variates = self.last_dims[0]
            self._new_shape = [self.input_rank[0] - 1] + [i for i in range(self.input_rank[0] - 1)]
            self._leg_shape = [i for i in range(1, self.input_rank[0])] + [0]

    def call(self, inputs, begin, quantiles=None):
        if begin:
            inputs = super().call(inputs) # initialize attributes in BaseStrategy
            if isinstance(inputs, tuple):
                inputs = inputs[0]
            if self.input_rank[0] > 2:
                inputs = tf.transpose(inputs, perm=self._new_shape)
            return inputs
        else:
            if self.input_rank[0] > 2 and quantiles is None:
                return tf.transpose(inputs, perm=self._leg_shape)
            return inputs


class MultiVariate(BaseStrategyMultivariate):
    def __init__(self, n_series, start, stop, **kwargs):
        super().__init__(**kwargs)

        if isinstance(n_series, int):
            self._custom_slices = False
            self.slices = [slice(start + idx*stop,
                        stop + idx*stop) for idx in n_series]
        elif isinstance(n_series, list):
            self._custom_slices = True
            self.slices = n_series

    def call(self, inputs):
        inputs = super().call(inputs)
        if self._custom_slices:
            return tf.ragged.stack([inputs[..., slicing] for slicing in self.slices], axis=0)
        return tf.stack([inputs[..., slicing] for slicing in self.slices], axis=0)


class Features(BaseStrategy):
    def __init__(self, features_slice, columns_index, **kwargs):
        super().__init__(**kwargs)
        self.columns_index = columns_index
        self.features_slice = features_slice

    def call(self, inputs):
        return features(inputs, self.features_slice, self.columns_index)


class DateFeatures(Features):
    def call(self, inputs):
        return date_features(inputs, self.features_slice, self.columns_index)
