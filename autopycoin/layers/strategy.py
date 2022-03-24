"""
Defines layers for time series analysis.
"""

from typing import Union, Tuple
import tensorflow as tf

from . import UnivariateLayer
from ..utils import transpose_first_to_last, transpose_last_to_first

# from ..dataset import features, date_features


class UniVariate(UnivariateLayer):
    """
    Used inside univariate models as first and last layer.
    It performs transpose operation to put the variables as the first dimensions
    to avoid to use them during calculations and get them back as last dimensions to
    fit with tensorflow norms.

    Parameters
    ----------
    last_to_first : boolean
        Set it `True`if it is the first layer else `False`.
    is_multivariate : boolean
        Based on your inputs, if `True` then it will perform back transformation
        if it is the last layer.
    """

    def __init__(
        self,
        last_to_first: bool = True,
        first_to_last: bool = False,
        is_multivariate: bool = False,
        **kwargs: dict
    ) -> None:

        super().__init__(**kwargs)

        self.last_to_first = last_to_first
        self.first_to_last = first_to_last
        self.is_multivariate = is_multivariate
        
    def build(self, input_shape):
        """
        Build the transform function.
        """

        # This is here because `is_multivariate` can be set 
        # during build method of an higher model or layer.
        # Inside `__init__`, the transform function is the indentity. 

        if self.last_to_first and self.is_multivariate:
            self.transform = transpose_last_to_first
        elif self.first_to_last and self.is_multivariate:
            self.transform = transpose_first_to_last
        else:
            self.transform = tf.identity

        super().build(input_shape)

    def call(self, inputs: Union[tf.Tensor, Tuple[tf.Tensor, ...]]) -> tf.Tensor:
        """See tensorflow documentation."""
        if isinstance(inputs, tuple):
            inputs = inputs[0]

        return self.transform(inputs)

    def get_config(self):
        """See tensorflow documentation."""

        config = super().get_config()
        config.update(
            {
                "last_to_first": self.last_to_first,
                "is_multivariate": self.is_multivariate,
            }
        )

        return config


def add_quantile_dim(fn_transform, axis):
    """
    Add a dimension to inputs which makes it broadcastable.

    Parameters
    ----------
    fn_transform : callable
        function which be applied before `expand_dims`.

    Returns
    -------
    fn : callable
        Apply `expand_dims`
    """

    def fn(inputs):
        return tf.expand_dims(fn_transform(inputs), axis=axis)

    return fn


"""class MultiVariate():
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


class Features():
    def __init__(self, features_slice, columns_index, **kwargs):
        super().__init__(**kwargs)
        self.columns_index = columns_index
        self.features_slice = features_slice

    def call(self, inputs):
        return features(inputs, self.features_slice, self.columns_index)


class DateFeatures():
    def call(self, inputs):
        return date_features(inputs, self.features_slice, self.columns_index)"""
