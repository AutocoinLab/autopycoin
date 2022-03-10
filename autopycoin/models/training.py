"""
Overloading Model tensorflow object
"""

import tensorflow.compat.v2 as tf
from keras.engine import data_adapter
from typing import Tuple
from keras.losses import LossFunctionWrapper

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
        self._has_quantiles = False
        self._quantiles = None # TODO: list for each output
        self._n_quantiles = 0 # TODO: list for each output
        self._apply_quantiles_transform = True

    def _check_quantiles_in_loss(self, loss):
        """
        Check if quantiles is present in a loss function.
        If True then it returns the last quantiles found.
        """

        if isinstance(loss, (tuple, list)):
            quantiles = [self._check_quantiles_in_loss(loss_fn) for loss_fn in loss if self._check_quantiles_in_loss(loss_fn)]
            for qtles in quantiles[1:]:
                if qtles != quantiles[0]:
                    raise ValueError(f'`quantiles` has to be identical through losses. Got {quantiles}')

            if len(quantiles) > 0:
                self._built = False
                self._has_quantiles = True
                return quantiles[0]

        elif hasattr(loss, "quantiles"):
            self._built = False
            self._has_quantiles = True
            return loss.quantiles

    # TODO: Avoid to rebuild weights when quantiles of model is not None
    def _set_quantiles(self, value):
        """Modify the shape of the layers to match with the new quantile values."""

        if self.has_quantiles:
            for idx, _ in enumerate(self.layers):
                if hasattr(self.layers[idx], '_set_quantiles'):
                    self.layers[idx]._set_quantiles(value)  # pylint: disable=protected-access
            self._n_quantiles = len(value)
            self._quantiles = value

    def compile(
        self,
        optimizer="rmsprop",
        loss=None, # TODO: multiple loss one output (qloss, mse) -> leads to mse loss over estimated due to the quantiles -> raise an error? use a wrapper to select only the 0.5 quantile?
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=None,
        steps_per_execution=None,
        **kwargs,
    ):
        """Compile method from tensorflow.

        When compiling with a loss defining a quantiles attribute
        it propagates this attribute to the model and sublayers."""

        quantiles = self._check_quantiles_in_loss(loss)
        self._set_quantiles(quantiles)

        super().compile(
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
        """See tensorflow documentation for more informations.

        add the quantiles dimension if needed and
        transpose the quantiles dimension to the last position to fit with keras norm.
        """

        print(f'inputs shape from {self}: ', inputs)

        if self._apply_quantiles_transform:
            # we add one dimension to inputs tensor if there is a quantile loss
            # We squeeze it if we don't need it in the output.
            #inputs = self._expand_dims_from_losses(inputs, axis=0)
            pass

        print(f'inputs shape after expand dim quantile from {self}: ', inputs)

        outputs = super().__call__(inputs, *args, **kwargs)

        print(f'outputs shape from {self}: ', outputs)

        if self.compiled_loss:
            if not self.compiled_loss.built:
                self.compiled_loss.build(outputs)

        print('fin build')

        if self._apply_quantiles_transform:
            r = self._handle_quantiles_dim(outputs, losses=self.compiled_loss._losses)
            print(f'outputs shape after quantiles transform from {self}: ', r)
            return r

        return outputs

    def train_step(self, data):
        """Transpose quantiles dimension to the last position to fit with keras norm.

        See tensorflow documentation for more informations.
        """

        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        # We have to know as fast as possible the nature of the output losses in order
        # to check if it corresponds to a quantile loss.
        if y is not None:
            if not self.compiled_loss.built:
                self.compiled_loss.build(y)

            # We expand dimension if the output is tested with a quantile loss
            y = self._expand_dims_from_losses(y, losses=self.compiled_loss._losses, axis=-1)

        return super().train_step((x, y, sample_weight))

    def test_step(self, data):
        """See tensorflow documentation"""

        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        # We have to know as fast as possible the nature of the output losses in order
        # to check if it corresponds to a quantile loss.
        if y is not None:
            if not self.compiled_loss.built:
                self.compiled_loss.build(y)

            # We expand dimension if the output is tested with a quantile loss
            y = self._expand_dims_from_losses(y, losses=self.compiled_loss._losses, axis=-1)

        return super().test_step((x, y, sample_weight))

    def _expand_dims_from_losses(self, tensors, axis, losses=None):
        """
        Wrapper that expand the inputs tensors if they are associated with a quantile loss.
        """

        if self.has_quantiles:

            # Case 1: multi outputs - multi losses
            if isinstance(tensors, (list, tuple)) and losses:
                return [self._expand_dims_from_losses(tensors_n, axis=axis, losses=loss) for tensors_n, loss in zip(tensors, losses)]

            # Case 2: no losses (inputs)
            elif isinstance(tensors, (list, tuple)) and not losses:
                return [tf.expand_dims(tensors_n, axis=axis) for tensors_n in tensors]

            # Case 3: one output - multi losses
            elif isinstance(tensors, tf.Tensor) and isinstance(losses, (list, tuple)):
                if any([hasattr(loss.fn, 'quantiles') for loss in losses]):
                    print('expand', tf.expand_dims(tensors, axis=axis))
                    return tf.expand_dims(tensors, axis=axis)
                return tensors

            # Case 4: one output - one loss
            elif isinstance(tensors, tf.Tensor) and isinstance(losses, LossFunctionWrapper):
                return tf.expand_dims(tensors, axis=axis) if hasattr(losses.fn, 'quantiles') else tensors

            # Case 5: one output - no loss
            elif isinstance(tensors, tf.Tensor) and losses is None:
                return tf.expand_dims(tensors, axis=axis)

        print('no_expand for y', (tensors, losses, self.has_quantiles))
        return tensors

    def _handle_quantiles_dim(self, outputs, losses):
        # TODO: wrap quantiles fn
        print('input_transpose', (outputs, losses))
        if self.has_quantiles:

            # Case 1: multi outputs != multi losses
            if isinstance(outputs, (list, tuple)) and isinstance(losses, (list, tuple)) and len(outputs) != len(losses):
                return [self._handle_quantiles_dim(output, losses) for output in outputs]

            # Case 2: multi outputs = multi losses
            elif isinstance(outputs, (list, tuple)) and losses:
                return [self._handle_quantiles_dim(output, loss) for output, loss in zip(outputs, losses)]

            # Case 3: one output - multi losses
            elif isinstance(outputs, tf.Tensor) and isinstance(losses, (list, tuple)):
                if all([not hasattr(loss.fn, 'quantiles') for loss in losses]):
                    return outputs[0]
                else:
                    return transpose_first_to_last(outputs)

            # Case 4: one output - one loss
            elif isinstance(outputs, tf.Tensor) and isinstance(losses, LossFunctionWrapper):
                if not hasattr(losses.fn, 'quantiles'):
                    return outputs[0]
                else:
                    return transpose_first_to_last(outputs)

        return outputs

    @property
    def quantiles(self):
        """Return quantiles attribute."""
        return self._quantiles

    @property
    def n_quantiles(self):
        """Return the number of quantiles."""
        return self._n_quantiles

    @property
    def has_quantiles(self):
        """Return True if quantiles exists else False"""
        return self._has_quantiles
