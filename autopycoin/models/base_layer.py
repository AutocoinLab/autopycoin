"""
Overloading Layers tensorflow object
"""

import tensorflow as tf


class Layer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._quantiles = None
        self._n_quantiles = None

    @property
    def quantiles(self):
        return self._quantiles

    def _set_quantiles(self, value):
        self.built = False
        self._n_quantiles = len(value)
        self._quantiles = value
