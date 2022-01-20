"""
Overloading Layers tensorflow object
"""

import tensorflow as tf


class Layer(tf.keras.layers.Layer):
    """
    Override tensorflow Layer class to integrate a `quantiles` attribute which is `None`
    if the Model is not computing confidence intervals.
    It checks during compiling if the loss function has quantiles attribute, hence it defines
    its internal `quantiles` attribute to fit the loss `quantiles` one.

    See tensorflow documentation for more details.
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self._quantiles = None
        self._n_quantiles = 0

    @property
    def quantiles(self):
        """Return the `quantiles` attribute."""

        return self._quantiles

    @property
    def n_quantiles(self):
        """Return the number of quantiles."""

        return self._n_quantiles

    def _set_quantiles(self, value):
        """Reset the `built` attribute to False and change the value of `quantiles`"""

        self.built = False
        self._n_quantiles = len(value)
        self._quantiles = value
