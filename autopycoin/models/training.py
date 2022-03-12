"""
Overloading Model tensorflow object
"""

from autopycoin.losses.losses import QuantileLossError
from autopycoin.utils.data_utils import transpose_last_to_first
import tensorflow.compat.v2 as tf
from keras.engine import data_adapter
from typing import Tuple
from keras.losses import LossFunctionWrapper
import math

from ..utils import transpose_first_to_last
from ..layers import UniVariate


class QuantileModel(tf.keras.Model):
    # TODO: doc
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
            quantiles = [q for loss_fn in loss if (q := self._check_quantiles_in_loss(loss_fn))]
            for qtles in quantiles[1:]:
                if qtles != quantiles[0]:
                    raise ValueError(f'`quantiles` has to be identical through losses. Got {quantiles}')

            if len(quantiles) > 0:
                return quantiles[0]

        elif hasattr(loss, "quantiles"):
            return loss.quantiles

    # TODO: Avoid to rebuild weights when quantiles of model is not None
    def _set_quantiles(self, value):
        """Modify the shape of the layers to match with the new quantile values."""

        self._built = False

        if value:
            self._has_quantiles = True
            for idx, _ in enumerate(self.layers):
                if hasattr(self.layers[idx], '_set_quantiles'):
                    self.layers[idx]._set_quantiles(value)  # pylint: disable=protected-access
            self._n_quantiles = len(value)
            print('quatiles_in_model', value)
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

    def init_quantile_build(self, input_shape):
        """We build the losses and metrics"""

        outputs_shape = self.compute_output_shape(input_shape)

        if self.compiled_loss:
            if not self.compiled_loss.built:
                self.compiled_loss.build(outputs_shape)

        if self.compiled_metrics:
            if not self.compiled_metrics.built:
                self.compiled_metrics.build(outputs_shape, outputs_shape)

        self.function_to_apply = []

    def __call__(self, inputs, *args, **kwargs):
        """See tensorflow documentation for more informations.

        add the quantiles dimension if needed and
        transpose the quantiles dimension to the last position to fit with keras norm.
        """

        print(f'inputs shape from {self}: ', inputs)

        outputs = super().__call__(inputs, *args, **kwargs)

        print(f'outputs shape from {self}: ', outputs)

        if self._apply_quantiles_transform:
            return self._handle_quantiles_dim(outputs)

        return outputs

    def _handle_quantiles_dim(self, outputs, loss=None): # pas envie de le mettre dans call
        # TODO: wrap quantiles fn
        if self.has_quantiles:

            losses = loss if loss else self.compiled_loss._losses

            losses_is_list = isinstance(losses, (list, tuple))
            outputs_is_list = isinstance(outputs, (list, tuple))
            outputs_is_tensor = isinstance(outputs, tf.Tensor)

            # Case 1: multi outputs != multi losses
            if outputs_is_list and losses_is_list and len(outputs) != len(losses):
                return [self._handle_quantiles_dim(output) for output in outputs]

            # Case 2: multi outputs = multi losses
            elif outputs_is_list and losses_is_list:
                return [self._handle_quantiles_dim(output, loss) for output, loss in zip(outputs, losses)]

            # Case 3: one output - multi losses
            elif outputs_is_tensor and losses_is_list:
                if any([hasattr(loss.fn, 'quantiles') for loss in losses]):
                    return transpose_first_to_last(outputs)

            # Case 4: one output - one loss
            elif outputs_is_tensor and isinstance(losses, LossFunctionWrapper):
                if hasattr(losses.fn, 'quantiles'):
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


class UnivariateModel(QuantileModel):
    """
    """

    NOT_INSPECT = ["call", "fit", "compile"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._apply_multivariate_transpose = True
        self._is_multivariate = False
        self._n_variates = 0

        self.transpose = tf.identity
        self.inverse_transpose = tf.identity

    def init_univariate_build(self, input_shape):

        self.set_is_multivariate(input_shape)
        self.set_n_variates(input_shape)

        if self.is_multivariate:
            self.transpose = transpose_last_to_first
            self.inverse_transpose = transpose_first_to_last

    def __call__(self, inputs, *args, **kwargs):

        outputs = super().__call__(inputs, *args, **kwargs)
        print('last', outputs)

        if self._apply_multivariate_transpose:
            outputs = self.inverse_transpose(outputs)
            print('after last', outputs)

        self._wrap_losses_and_metrics(outputs)

        return outputs

    def call(self, inputs, **kwargs):
        """
        Need to wait until build is finished to have `n_variates`
        """

        if self._apply_multivariate_transpose:
            inputs = self.transpose(inputs)

        return inputs

    def _wrap_losses_and_metrics(self, outputs):
        """We have to build `compiled_loss` and `compiled_metrics` in order to wrap them."""

        if self.compiled_loss:
            if not self.compiled_loss.built:
                self.compiled_loss.build(outputs)
                self.compiled_loss._losses = tf.nest.map_structure(self._quantiles_and_nvariates_loss_wrapper, self.compiled_losses._losses)

        if self.compiled_metrics:
            if not self.compiled_metrics.built:
                self.compiled_metrics.build(outputs, outputs)
                self.compiled_metrics._metrics = tf.nest.map_structure(self._quantiles_and_nvariates_metrics_wrapper, self.compiled_metrics._metrics)

    def set_is_multivariate(self, input_shape):
        self._is_multivariate = bool(input_shape.rank > 2)

    def set_n_variates(self, input_shape):
        if self.is_multivariate:
            self._n_variates = input_shape[-1]

    @property
    def is_multivariate(self):
        return self._is_multivariate

    @property
    def n_variates(self):
        return self._n_variates

    def _quantiles_and_nvariates_metrics_wrapper(self, metrics):
        if not hasattr(metrics, 'quantiles') and not self.is_multivariate:
            metrics.update_state = _remove_last_dimension_to_ypred(metrics.update_state)
        elif not hasattr(metrics, 'quantiles') and self.is_multivariate:
            metrics.update_state = _remove_second_last_dimension_to_ypred(metrics.update_state)
        return metrics

    def _quantiles_and_nvariates_loss_wrapper(self, loss):
        if not hasattr(loss.fn, 'quantiles') and not self.is_multivariate:
            loss.fn = _remove_last_dimension_to_ypred(loss.fn)
        elif not hasattr(loss.fn, 'quantiles') and self.is_multivariate:
            loss.fn = _remove_second_last_dimension_to_ypred(loss.fn)
        return loss

def _remove_last_dimension_to_ypred(fn):
    """We remove the quantile dimension from y_pred if it is not needed,
    then y_true and y_pred are broadcastable.
    """

    def new_fn(y_true, y_pred, *args, **kwargs):

        if len(y_pred.shape) > len(y_true.shape):
            q = math.ceil(y_pred.shape[-1] / 2)
            y_pred = y_pred[..., q]

        return fn(y_true, y_pred, *args, **kwargs)
    return new_fn


def _remove_second_last_dimension_to_ypred(fn):
    """We remove the quantile dimension from y_pred if it is not needed,
    then y_true and y_pred are broadcastable.
    """

    def new_fn(y_true, y_pred, *args, **kwargs):

        if len(y_pred.shape) > len(y_true.shape):
            q = math.ceil(y_pred.shape[-1] / 2)
            y_pred = y_pred[..., q, :]

        return fn(y_true, y_pred, *args, **kwargs)
    return new_fn


class UnivariateTensor(tf.experimental.ExtensionType):
    values: tf.Tensor
    is_multivariate: bool
    is_quantile: bool
