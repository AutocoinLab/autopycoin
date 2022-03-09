"""
Overloading Model tensorflow object
"""

import tensorflow.compat.v2 as tf
from keras.engine import data_adapter

from ..utils import transpose_first_to_last


class Model(tf.keras.Model):
    """
    Overloading tensorflow Model class to integrate a `quantiles` attribute which is `None`
    if the Model is not computing prediction and confidence intervals.
    It checks during compiling if the loss function has quantiles attribute, hnce it defines
    its internal `quantiles` attribute to fit the loss `quantiles` one.
    """

    NOT_INSPECT = ["call", "fit", "compile"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._quantiles = None
        self._n_quantiles = 0
        self._apply_quantiles_transform = True

    def _check_quantiles_in_loss(self, loss):
        if isinstance(loss, (tuple, list)):
            quantiles = [loss_fn.quantiles for loss_fn in loss if hasattr(loss_fn, "quantiles")]
            for qtles in quantiles[1:]:
                if qtles != quantiles[0]:
                    raise ValueError(f'`quantiles` has to be identical through losses. Got {quantiles}')
            if len(quantiles) > 0:
                self._set_quantiles(quantiles[0])
        elif hasattr(loss, "quantiles"):
            self._set_quantiles(loss.quantiles)

    # TODO: Avoid to rebuild weights when quantiles of model is not None
    def _set_quantiles(self, value):
        """Modify the shape of the layers to match with the new quantile values."""
        self.built = False
        for idx, _ in enumerate(self.layers):
            if hasattr(self.layers[idx], '_set_quantiles'):
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
        """Compile method from tensorflow. When compiling with a loss defining a quantiles attribute
        it propagates this attribute to the model and sublayers."""

        # TODO: handle loss in a list
        self._check_quantiles_in_loss(loss)

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

    def __call__(self, inputs, *args, **kwargs):
        """Transpose quantiles dimension to the last position to fit with keras norm.
        See tensorflow documentation for more informations.
        """

        if self._apply_quantiles_transform:
            inputs = _expand_dims_from_quantiles(inputs, self.n_quantiles, axis=0)

        outputs = super().__call__(inputs, *args, **kwargs)

        if self._apply_quantiles_transform:
            return _apply_quantiles_transpose(outputs, self.n_quantiles)

        return outputs

    def train_step(self, data):
        """Transpose quantiles dimension to the last position to fit with keras norm.
        See tensorflow documentation
        """
        
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        y = _expand_dims_from_quantiles(y, self.n_quantiles, axis=-1)

        return super().train_step((x, y, sample_weight))

    def test_step(self, data):
        """See tensorflow documentation"""
        
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        y = _expand_dims_from_quantiles(y, self.n_quantiles, axis=-1)
        
        return super().test_step((x, y, sample_weight))

    @property
    def quantiles(self):
        """Return quantiles attribute."""
        return self._quantiles

    @property
    def n_quantiles(self):
        """Return the number of quantiles."""
        return self._n_quantiles


def _expand_dims_from_quantiles(tensors, n_quantiles, axis):
    # TODO: wrap quantiles fn
    if n_quantiles > 1 and tensors is not None:
        if isinstance(tensors, (list, tuple)):
            tensors = [_expand_dims_from_quantiles(tensors_n, n_quantiles, axis=axis) for tensors_n in tensors]
        else:
            tensors = tf.expand_dims(tensors, axis=axis)
    return tensors

def _apply_quantiles_transpose(outputs, n_quantiles):
    # TODO: wrap quantiles fn
    if n_quantiles > 1:
        if isinstance(outputs, (list, tuple)):
            return [transpose_first_to_last(output) for output in outputs]
        else:
            return transpose_first_to_last(outputs)
    return outputs
