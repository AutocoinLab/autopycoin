"""
Overloading Model tensorflow object
"""

import tensorflow.compat.v2 as tf

from ..utils import transpose_first_to_last


class Model(tf.keras.Model):
    """
    Overloading tensorflow Model class to integrate a `quantiles` attribute which is `None`
    if the Model is not computing prediction and confidence intervals.
    It checks during compiling if the loss function has quantiles attribute, hnce it defines
    its internal `quantiles` attribute to fit the loss `quantiles` one.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._quantiles = None
        self._n_quantiles = 0

    @property
    def quantiles(self):
        """Return quantiles attribute."""
        return self._quantiles

    @property
    def n_quantiles(self):
        """Return the number of quantiles."""
        return self._n_quantiles

    # TODO: Avoid to rebuild weights when quantiles of model is not None
    def _set_quantiles(self, value):
        """Modify the shape of the layers to match with the new quantile values."""
        self.built = False
        for idx, _ in enumerate(self.layers):
            self.layers[idx]._set_quantiles(value)  # pylint: disable=protected-access
        self._n_quantiles = len(value)
        self._quantiles = value

    def compile(
        self,
        optimizer="rmsprop",
        loss=None,
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=None,
        steps_per_execution=None,
        **kwargs,
    ):
        """Compile method from tensorflow. When compiling with uncertainty loss defining quantiles
        it defines `quantiles` attribute in thz model and sublayers."""

        if hasattr(loss, "quantiles"):
            self._set_quantiles(loss.quantiles)

        return super().compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            loss_weights=loss_weights,
            weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly,
            steps_per_execution=steps_per_execution,
            **kwargs,
        )

    def __call__(self, *args, **kwargs):
        """See tensorflow documentation"""
        outputs = super().__call__(*args, **kwargs)
        if self.n_quantiles > 1 and not isinstance(outputs, tuple):
            outputs = transpose_first_to_last(outputs)
        return outputs

    def train_step(self, data):
        """See tensorflow documentation"""
        if self.n_quantiles > 1:
            data = (data[0], tf.expand_dims(data[1], axis=-1))
        return super().train_step(data)

    def test_step(self, data):
        """See tensorflow documentation"""
        if self.n_quantiles > 1:
            data = (data[0], tf.expand_dims(data[1], axis=-1))
        return super().test_step(data)
