"""
Overloading Model tensorflow object
"""

import tensorflow.compat.v2 as tf


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

    @property
    def quantiles(self):
        """Return quantiles attribute."""
        return self._quantiles

    # TODO: Avoid to rebuild weights when quantiles of model is not None
    def _set_quantiles(self, value):
        """Modify the shape of the layers to match with the new quantile values."""
        self.built = False
        for idx, _ in enumerate(self.layers):
            self.layers[idx]._set_quantiles(value)  # pylint: disable=protected-access
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
        outputs = super().__call__(*args, **kwargs)
        # TODO: not isinstance(outputs, tuple) need to be deleted when stack will become a layer
        if self.quantiles is not None and not isinstance(outputs, tuple):
            if len(self.quantiles) > 1:
                outputs = tf.transpose(outputs, perm=tf.concat([tf.range(1, len(outputs.shape)), [0]], axis=0))
        return outputs

    def train_step(self, data):
        if self.quantiles is not None:
            data = (data[0], tf.expand_dims(data[1], axis=-1))
        return super().train_step(data)

    def test_step(self, data):
        if self.quantiles is not None:
            data = (data[0], tf.expand_dims(data[1], axis=-1))
        return super().test_step(data)
