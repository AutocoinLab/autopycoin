"""
Overloading Layers tensorflow object
"""

import tensorflow as tf


class QuantileLayer(tf.keras.layers.Layer):
    """
    Override tensorflow Layer class to integrate a `quantiles` attribute which is `None`
    if the Model is not computing confidence intervals.
    It checks during compiling if the loss function has quantiles attribute, hence it defines
    its internal `quantiles` attribute to fit the loss `quantiles` one.

    See tensorflow documentation for more details.
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self._has_quantiles = False
        self._quantiles = None
        self._n_quantiles = 0
        self._additional_shapes = [[]]

    def _set_quantiles(self, value, additional_shapes=None, n_quantiles=None):
        """Reset the `built` attribute to False and change the value of `quantiles`"""

        self._built = False
        self._quantiles = value
        self._additional_shapes = additional_shapes
        self._n_quantiles = n_quantiles

    def get_additional_shapes(self, index):
        """Return the shape to add to your layers."""

        try:
            return self._additional_shapes[index]
        except IndexError:
            return []

    @property
    def quantiles(self):
        """Return the `quantiles` attribute."""

        return self._quantiles

    @property
    def n_quantiles(self):
        """Return the number of quantiles."""

        return self._n_quantiles


class UnivariateLayer(QuantileLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._apply_multivariate_transpose = True
        self._init_multivariates_params = False

    def init_params(self, input_shape, n_variates, is_multivariate, additional_shapes):

        self._n_variates = n_variates
        self._additional_shapes = additional_shapes
        self._is_multivariate = is_multivariate
        self._init_multivariates_params = True

    def get_additional_shapes(self, index):
        if not self._init_multivariates_params:
            raise AssertionError('Please call `init_univariate_params` at the beginning of build.')
        return super().get_additional_shapes(index)

    @property
    def is_multivariate(self):
        return self._is_multivariate

    @property
    def n_variates(self):
        return self._n_variates
