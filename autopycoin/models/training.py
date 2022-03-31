"""
Overloading Model tensorflow object
"""

import abc
import math
from typing import List, Union, Callable, Any
from autopycoin.utils.data_utils import quantiles_handler
import itertools

import keras

import tensorflow.compat.v2 as tf
from keras.losses import LossFunctionWrapper

from ..utils import transpose_first_to_last, transpose_last_to_first, transpose_first_to_second_last, convert_to_list
from ..extension_type import QuantileTensor, UnivariateTensor


class AutopycoinMetaModel(abc.ABCMeta):
    """Metaclass for autopycoin models."""

    def __init__(cls, name, bases, namespace):
        cls.build = _wrap_build(cls.build)
        cls.call = _wrap_call(cls.call)

        super().__init__(name, bases, namespace)

def _wrap_call(fn):
    """Wrap the call method with a _preprocessing and _post_processing methods"""

    def call_wrapper(self, inputs, *args, **kwargs):
        inputs = self._preprocessing_wrapper(inputs)
        outputs = fn(self, inputs, *args, **kwargs)
        outputs = self._post_processing_wrapper(outputs)
        return outputs
    return call_wrapper

def _wrap_build(fn):
    """Wrap the build method with a init_params function"""

    def build_wrapper(self, inputs_shape):
        self.init_params(inputs_shape)
        return fn(self, inputs_shape)
    return build_wrapper


class BaseModel(keras.Model, metaclass=AutopycoinMetaModel):
    """Base model which defines pre/post-processing methods to override.

    Currently, four wrappers can be overriden:
    - preprocessing : Preprocess the inputs data
    - post_processing : Preprocess the outputs data
    - metrics_wrapper : Preprocess y_true or y_pred
    - losses_wrapper : Preprocess y_true or y_pred

    `Compute_output_shape` needs also to be implemented.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _handle_dim_in_losses_and_metrics(self, outputs: Union[List[tf.Tensor], tf.Tensor]):
        """Build and wrap losses and metrics."""

        if self.compiled_loss:
            if not self.compiled_loss.built:
                self.compiled_loss.build(outputs)
            self.compiled_loss._losses = tf.nest.map_structure(self.losses_wrapper, self.compiled_loss._losses)

        if self.compiled_metrics:
            if not self.compiled_metrics.built:
                self.compiled_metrics.build(outputs, outputs)
            self.compiled_metrics._metrics = tf.nest.map_structure(self.metrics_wrapper, self.compiled_metrics._metrics)

    def losses_wrapper(self, loss: LossFunctionWrapper) -> Union[Callable, LossFunctionWrapper]:
        """Wrap the `fn` function.

        See `tf.keras.losses.LossFunctionWrapper` docstring for more informations about `fn`.
        """

        raise NotImplementedError('`losses_wrapper` has to be overriden')

    def metrics_wrapper(self, metrics: Any) -> Union[Callable, LossFunctionWrapper]:
        """Wrap the update_state function.

        See s`tf.keras.metrics.Metric` docstring for more informations about `update_state`.
        """

        raise NotImplementedError('`losses_wrapper` has to be overriden')

    def _preprocessing_wrapper(self, inputs):
        return self.preprocessing(inputs)

    def preprocessing(self, inputs):
        """Public API to apply preprocessing logics to your inputs data."""

        raise NotImplementedError('`preprocessing` has to be overriden.')

    def _post_processing_wrapper(self, outputs):
        """Post-processing wrapper.

        it handles the case of:
         - one tensor vs multi loss functions or no loss function
         - one tensor vs one loss function
        """

        self._handle_dim_in_losses_and_metrics(outputs)

        losses = getattr(self.compiled_loss, '_losses', None)
        losses_is_nested = tf.nest.is_nested(losses)
        outputs_is_nested = tf.nest.is_nested(outputs)

        outputs = tuple(outputs) if outputs_is_nested else (outputs,)

        # Case 1: multi outputs != multi losses or no losses
        if losses_is_nested and len(outputs) == len(losses):
            outputs = tf.nest.map_structure(lambda output, loss: self.post_processing(output, loss), outputs, tuple(losses))
            return outputs[0] if len(outputs) == 1 else outputs

        # Case 2: multi outputs = multi losses
        else:
            outputs = tf.nest.map_structure(lambda output: self.post_processing(output, losses), outputs)
            return outputs[0] if len(outputs) == 1 else outputs

    def post_processing(self, output, losses=None):
        """Public API to apply post-processing logics to your inputs data."""

        raise NotImplementedError('`post_processing` has to be overriden.')


class QuantileModel(BaseModel): # # pylint: disable=abstract-method
    """Overloads tensorflow Model class to integrate a `quantiles` attribute.

    During the compiling phase, the model checks the existence of the attribute `quantiles` in each loss function.
    If the test is positive then the model defines several attributes based on `quantiles` found in the loss functions.
    The model propagates the attributes associated to `quantiles` to the sublayers.
    Be carefull, if the check is positive the model is no more built.

    During the first call, all compiled losses and metrics are build and a second check is perfomed to ensure that
    Each output is not associated with different `quantiles` values otherwise it raises a ValueError.

    When subclassing this model, a pre/post-processing methods can be defined.
    Also a `_preprocessing` and `_post_processing` are already defined in order to transpose the quantiles/multivariates dimensions.

    Attributes
    ----------
    has_quantiles : bool
        True if `quantiles` is not None else False. It is defined during compiling `method`. Default to False.
    quantiles : list[float]
        It defines the quantiles used in the model.
        `quantiles` can be a  list depending on the number of outputs the model computes.
        It is defined during compiling `method`.
        Default to None.
    n_quantiles : list[int] or int
        The number of quantiles the model computes.
        It is defined during compiling `method`.
        Default to 0.
    additional_shape : list[int]
        This the shapes defined by the `n_quantiles` attribute to add to your layers definitions.
        It is defined during compiling `method`.
        default to [].
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self._has_quantiles = False
        self._quantiles = None
        self._n_quantiles = 0
        self._additional_shapes = [[]]
        self._apply_quantiles_transpose = True

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

        When compiling with losses defining a quantiles attribute,
        it propagates this attribute to the model and sublayers.
        """

        # Check if `quantiles` exists
        quantiles = self._check_quantiles_in_loss(loss)

        # Defines attributes associated with `quantiles` and propagates it to sublayers
        if quantiles:
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

    def _check_quantiles_in_loss(self, loss: Union[str, tf.keras.losses.Loss, LossFunctionWrapper]) -> Union[List[Union[List[int], int]], None]:
        """Check if the loss functions define a `quantiles` attribute.

        If True then it returns the quantiles found.
        """

        # Case of multiple losses
        if isinstance(loss, (tuple, list)):
            quantiles = [q for loss_fn in loss if (q := self._check_quantiles_in_loss(loss_fn))]
            return list(itertools.chain.from_iterable(quantiles))

        # One loss
        elif hasattr(loss, "quantiles"):
            return quantiles_handler(loss.quantiles)

    # TODO: Avoid to rebuild weights when quantiles of model is not None
    def _set_quantiles(self, value: List[Union[List[int], int]], additional_shapes=None, n_quantiles=None) -> None:
        """Set attributes linked to the quantiles found in the losses functions."""

        self._built = False
        self._has_quantiles = True
        self._quantiles = value
        self._additional_shapes = additional_shapes or [[len(q)] for q in self.quantiles]
        self._n_quantiles = n_quantiles or self._additional_shapes.copy()

        # Propagates to sublayers
        for idx, _ in enumerate(self.layers):
            if hasattr(self.layers[idx], '_set_quantiles'):
                self.layers[idx]._set_quantiles(value, self._additional_shapes, self.n_quantiles)  # pylint: disable=protected-access

    def losses_wrapper(self, loss: LossFunctionWrapper) -> Union[Callable, LossFunctionWrapper]:
        """Add or remove the quantile dimension to y_pred and y_true respectively."""

        # TODO: We override the fn function which can be a Loss instance and turn it into function.
        # As below we have to recreate an instance of the loss otherwise we lose informations as the attributes etc...
        if not hasattr(loss.fn, 'quantiles') and not isinstance(loss, type(None)):
            loss.fn = _remove_dimension_to_ypred(loss.fn)
            return loss
        elif not isinstance(loss, type(None)):
            loss = _add_dimension_to_ytrue(loss, type(loss))
        return loss

    def metrics_wrapper(self, metric: Any) -> Union[Callable, LossFunctionWrapper]:
        """Add or remove the quantile dimension to y_pred and y_true respectively."""

        # TODO: We override the update_state function which can be a Loss instance and turn it into function.
        # As below we have to recreate an instance of the loss otherwise we lose informations as the attributes etc...
        if not hasattr(metric, 'quantiles') and not isinstance(metric, type(None)):
            metric.update_state = _remove_dimension_to_ypred(metric.update_state)
            return metric

        elif not isinstance(metric, type(None)):
            metric = _add_dimension_to_ytrue(metric, type(metric))

        return metric

    def preprocessing(self, inputs):
        """No preprocessing for `QuantileModel`"""
        return inputs

    def post_processing(self, outputs, losses):
        """Convert the outputs to `QuantileTensor` and apply transpose operation.

        The quantiles dimension is put to the last dimension to fit with keras norms.
        """

        if self._apply_quantiles_transpose:
            if self._check_quantiles_requirements(outputs, losses):
                outputs = transpose_first_to_last(outputs)
                if outputs.shape[-1] == 1:
                    outputs = tf.squeeze(outputs, axis=-1)
                return QuantileTensor(outputs, quantiles=True)
            return QuantileTensor(outputs, quantiles=False)
        return outputs

    def _check_quantiles_requirements(self, outputs, losses=None):
        """Check if the requirements are valids else raise a ValueError.

        Raises
        ------
        ValueError:
            If the losses don't define a same `quantiles` attribute respectively to one output.
            If the output contains a `quantiles` dimension and there isn't at least one quantile loss.
            If the output don't contains a `quantiles` dimension and there is at least one quantile loss.
            If the output `quantiles` are not broadcastable with the losses `quantiles`
        """

        if self.has_quantiles and losses:
            losses = [loss.fn for loss in convert_to_list(losses)]

            # TODO: optimization, this calculation is made twice (One in _handle_quantiles_dim_in_losses_and_metrics)
            quantiles_in_losses = [loss.quantiles[0] for loss in losses if hasattr(loss, 'quantiles')]

            check_uniform_quantiles = self._check_uniform_quantiles_through_losses(quantiles_in_losses)
            check_quantiles_in_outputs = self._check_quantiles_in_outputs(outputs)

            if not check_uniform_quantiles:
                raise ValueError(f"`quantiles` has to be identical through losses. Got losses {quantiles_in_losses}.")

            elif not any(quantiles_in_losses) and check_quantiles_in_outputs:
                raise ValueError(f"It is not allowed to train a quantile model without a quantile loss. Got a loss {losses} and an output shape {outputs.shape}.")

            elif any(quantiles_in_losses):
                if self._compare_quantiles_in_outputs_and_losses(outputs, quantiles_in_losses):
                    return True
                
                elif self._is_single_quantile(quantiles_in_losses):
                    return False

                raise ValueError(f"Quantiles in losses and outputs are not the same. Maybe you are trying to train a no quantile model "
                        f"with a quantile loss. It is possible only if there is one quantile defined as [[0.5]]. "
                        f"got outputs shape: {outputs.shape} and quantiles in losses: {quantiles_in_losses}")

        return False

    def _check_uniform_quantiles_through_losses(self, quantiles_in_losses):
        """Return True if all losses define an identical `quantiles` attribute"""

        if len(quantiles_in_losses) == 0: # Case of no quantiles in losses
            return True
        return all(q == quantiles_in_losses[idx - 1] for idx, q in enumerate(quantiles_in_losses))

    def _check_quantiles_in_outputs(self, outputs):
        """Return True if the outputs contains a `quantiles` dimension"""

        # TODO: find an other way to find if an outputs contains quantiles dimension

        return any(s == outputs.shape[:len(s)] for s in self._additional_shapes) # or self._additional_shapes

    def _compare_quantiles_in_outputs_and_losses(self, outputs, quantiles_in_losses):
        """Return True if the outputs and the quantile loss have the same `quantiles` attribute"""

        return (len(quantiles_in_losses[0]) == outputs.shape[0])

    def _is_single_quantile(self, quantiles_in_losses):
        return len(quantiles_in_losses[0]) == 1

    def get_additional_shapes(self, index):
        """Return the shape to add to your layers."""

        try:
            return self._additional_shapes[index]
        except IndexError:
            return []

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
        """Return True if quantiles exists else False."""

        return self._has_quantiles


def _remove_dimension_to_ypred(fn):
    """We remove the quantile dimension from y_pred if it is not needed,
    then y_true and y_pred are broadcastable.
    """

    @tf.function(experimental_relax_shapes=True)
    def new_fn(y_true, y_pred, *args, **kwargs):

        if y_pred.quantiles and y_pred.shape.rank > y_true.shape.rank:
            q = math.ceil(y_pred.shape[-1] / 2)
            y_pred = y_pred[..., q]

        return fn(y_true, y_pred, *args, **kwargs)

    return new_fn


def _add_dimension_to_ytrue(fn, obj):
    """We add the quantile dimension from y_true if it is needed,
    then y_true and y_pred are broadcastable.
    """

    @tf.function(experimental_relax_shapes=True)
    def new_fn(y_true, y_pred, *args, **kwargs):

        if y_pred.shape.rank > y_true.shape.rank:
            y_true = tf.expand_dims(y_true, -1)

        return fn(y_true, y_pred, *args, **kwargs)

    kwargs = fn.get_config()
    quantiles = fn.fn.quantiles
    new_fn = obj(new_fn, **kwargs)
    new_fn.fn.quantiles = quantiles
    return new_fn


class UnivariateModel(QuantileModel):
    """
    Wrapper around `QuantileModel` to integrate multivariate attributes.

    for the moment, if one of the inputs tensors is a multivariates tensor then
    all `additional_shapes` are extended by `n_variates`.
    In other words all layers extended by `additional_shapes` are multivariates layers.

    Attributes
    ----------
    is_multivariate : bool
        True if the inputs rank is higher than 2. Default to False.
    n_variates : int
        the number of variates in the inputs. Default to 0.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._apply_multivariate_transpose = True
        self._init_multivariates_params = False

        self._is_multivariate = False
        self._n_variates = []

    def preprocessing(self, inputs):
        """Init the multivariates attributes and transpose the `nvariates` dimension in first position."""

        if self._apply_multivariate_transpose and self._check_multivariates_requirements():
            return tf.nest.map_structure(transpose_last_to_first, inputs)
        return inputs

    def post_processing(self, outputs, losses=None):
        outputs = super().post_processing(outputs, losses)
        if self._apply_multivariate_transpose:
            if self._check_multivariates_requirements():
                outputs = tf.nest.map_structure(lambda outputs: transpose_first_to_second_last(outputs) if outputs.quantiles else transpose_first_to_last(outputs), outputs)
                outputs = tf.nest.map_structure(convert_to_univariate_tensor(multivariates=True), outputs)
                return outputs

            outputs = tf.nest.map_structure(convert_to_univariate_tensor(multivariates=False), outputs)
            return outputs
        return outputs

    def _check_multivariates_requirements(self):
        return self.is_multivariate

    def init_params(self, input_shape, n_variates=None, is_multivariate=None, additional_shapes=None):
        """it is called before `build`.

        Parameters
        ----------
        input_shape : tf.TensorShape
            The shape of the input tensor.
        """

        if not self._init_multivariates_params:
            if isinstance(input_shape, tuple):
                input_shape = input_shape[0]

            self._init_multivariates_params = True
            self.set_is_multivariate(input_shape, is_multivariate)
            self.set_n_variates(input_shape, n_variates)
            self.extend_additional_shape(additional_shapes)

            # Propagates to sublayers
            for idx, _ in enumerate(self.layers):
                if hasattr(self.layers[idx], 'init_params'):
                    self.layers[idx].init_params(input_shape, self.n_variates, self.is_multivariate, self._additional_shapes)  # pylint: disable=protected-access

    def set_is_multivariate(self, input_shape, is_multivariate=None):
        """Initiate `is_multivariate` attribute"""

        self._is_multivariate = is_multivariate or bool(input_shape.rank > 2)

    def set_n_variates(self, input_shape, n_variates=None):
        """Initiate `n_variates` attribute"""

        if self.is_multivariate:
            self._n_variates = convert_to_list(n_variates or input_shape[-1])

    def get_additional_shapes(self, index):
        if not self._init_multivariates_params:
            raise AssertionError('Please call `init_univariate_params` at the beginning of build.')
        return super().get_additional_shapes(index)

    def extend_additional_shape(self, additional_shapes=None):
        self._additional_shapes = additional_shapes or [s + self.n_variates for s in self._additional_shapes]

    @property
    def is_multivariate(self):
        return self._is_multivariate

    @property
    def n_variates(self):
        return self._n_variates


def convert_to_univariate_tensor(multivariates):
    def fn(tensor):
        return UnivariateTensor(values=tensor, multivariates=multivariates)
    return fn
