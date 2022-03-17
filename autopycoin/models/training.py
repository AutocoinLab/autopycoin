"""
Overloading Model tensorflow object
"""

import math
from typing import List, Union, Callable, Any
from autopycoin.utils.data_utils import quantiles_handler
import itertools

import keras

import tensorflow.compat.v2 as tf
from keras.losses import LossFunctionWrapper
from keras.engine import keras_tensor

from ..utils import transpose_first_to_last, transpose_last_to_first, transpose_first_to_second_last, convert_to_list


class QuantileModel(keras.Model):
    """Overloads tensorflow Model class to integrate a `quantiles` attribute.

    During the compiling phase, the model checks the existence of the attribute `quantiles` in each loss function. 
    If the test is positive then the model defines several attributes based on `quantiles` found in the loss functions.
    The model propagates the attributes associated to `quantiles` to the sublayers.
    Be carefull, if the check is positive the model is no more built.

    During the first call, all compiled losses and metrics are build and a second check is perfomed to ensure that
    Each output is not associated with different `quantiles` values otherwise it raises a ValueError.

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

    def __new__(cls, *args, **kwargs):
        cls.call = cls._wrap(cls.call)
        return super(tf.keras.Model, cls).__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self._has_quantiles = False
        self._quantiles = None # TODO: list for each output
        self._n_quantiles = 0 # TODO: list for each output
        self._additional_shapes = [[]]
        self._apply_quantiles_transform = True

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
    def _set_quantiles(self, value: List[Union[List[int], int]]) -> None:
        """Modify the shape of the layers to match with the new quantile values."""

        self.built = False
        self._has_quantiles = True

        # Propagates to sublayers
        for idx, _ in enumerate(self.layers):
            if hasattr(self.layers[idx], '_set_quantiles'):
                self.layers[idx]._set_quantiles(value)  # pylint: disable=protected-access

        self._quantiles = value
        self._additional_shapes = [[len(q)] for q in self.quantiles]
        self._n_quantiles = self._additional_shapes.copy()

    def _handle_quantiles_dim_in_losses_and_metrics(self, outputs: Union[List[tf.Tensor], tf.Tensor]):
        """Build and wrap losses and metrics."""

        if self.compiled_loss:
            if not self.compiled_loss.built:
                self.compiled_loss.build(outputs)
            self.compiled_loss._losses = tf.nest.map_structure(self._quantiles_and_nvariates_loss_wrapper, self.compiled_loss._losses)

        if self.compiled_metrics:
            if not self.compiled_metrics.built:
                self.compiled_metrics.build(outputs, outputs)
            self.compiled_metrics._metrics = tf.nest.map_structure(self._quantiles_and_nvariates_metrics_wrapper, self.compiled_metrics._metrics)

    def _quantiles_and_nvariates_metrics_wrapper(self, metrics: Any) -> Union[Callable, LossFunctionWrapper]:
        """Wrap the update_state function in order to add or remove the quantile dimension to y_pred and y_true respectively.

        See __call__ docstring for more informations.
        """

        # TODO: We override the update_state function which can be a Loss instance and turn it into function.
        # As below we have to recreate an instance of the loss otherwise we lose informations as the attributes etc...
        if not hasattr(metrics, 'quantiles'):
            metrics.update_state = _remove_dimension_to_ypred(metrics.update_state)
            return metrics

        metrics = _add_dimension_to_ytrue(metrics, type(metrics))
        return metrics

    def _quantiles_and_nvariates_loss_wrapper(self, loss: LossFunctionWrapper) -> Union[Callable, LossFunctionWrapper]:
        """Wrap the fn function in order to add or remove the quantile dimension to y_pred and y_true respectively.

        See __call__ docstring for more informations.
        """

        # TODO: We override the fn function which can be a Loss instance and turn it into function.
        # As below we have to recreate an instance of the loss otherwise we lose informations as the attributes etc...
        if not hasattr(loss.fn, 'quantiles'):
            loss.fn = _remove_dimension_to_ypred(loss.fn)
            return loss

        loss = _add_dimension_to_ytrue(loss, type(loss))
        return loss
        
    @classmethod
    def _wrap(cls, fn):
        @tf.function
        def call(self, inputs, *args, **kwargs):
            inputs = self._preprocessing_wrapper(inputs)
            outputs = fn(self, inputs, *args, **kwargs)
            self._handle_quantiles_dim_in_losses_and_metrics(outputs)
            outputs = self._post_processing_wrapper(outputs)
            return outputs
        return call

    def _preprocessing_wrapper(self, inputs, loss=None):
        return self._preprocessing(inputs)

    def _post_processing_wrapper(self, outputs, loss=None):

        losses = loss or getattr(self.compiled_loss, '_losses', None)
        losses_is_nested = tf.nest.is_nested(losses)
        outputs_is_nested = tf.nest.is_nested(outputs)
        outputs_is_shape = _is_nested_type(outputs, tf.TensorShape)
        outputs_is_signature = _is_nested_type(outputs, tf.TensorSpec)
        outputs_is_tensor = _is_nested_type(outputs, tf.Tensor)
        outputs_is_keras_tensor = _is_nested_type(outputs, keras_tensor.KerasTensor)

        outputs = tuple(outputs) if outputs_is_nested else (outputs,)

        # Case 1: multi outputs != multi losses or no losses
        if losses_is_nested and len(outputs) == len(losses):
            outputs = tf.nest.map_structure(lambda output, loss: self._post_processing_f(output, loss, outputs_is_shape=outputs_is_shape,
                                                                                    outputs_is_signature=outputs_is_signature,
                                                                                    outputs_is_tensor=outputs_is_tensor,
                                                                                    outputs_is_keras_tensor=outputs_is_keras_tensor), outputs, tuple(losses))
            return outputs[0] if len(outputs) == 1 else outputs

        # Case 2: multi outputs = multi losses
        else:
            outputs = tf.nest.map_structure(lambda output: self._post_processing_f(output, losses, outputs_is_shape=outputs_is_shape,
                                                                                    outputs_is_signature=outputs_is_signature,
                                                                                    outputs_is_tensor=outputs_is_tensor,
                                                                                    outputs_is_keras_tensor=outputs_is_keras_tensor), outputs)
            return outputs[0] if len(outputs) == 1 else outputs

    def _post_processing_f(self, outputs, losses, outputs_is_shape, outputs_is_signature, outputs_is_tensor, outputs_is_keras_tensor):

        if outputs_is_keras_tensor:
            return self._keras_tensor_pipeline(outputs, losses)
        elif outputs_is_tensor:
            return self._tensor_pipeline(outputs, losses)
        elif outputs_is_shape:
            return self._shape_pipeline(outputs, losses)
        
        return outputs

    def _tensor_pipeline(self, outputs, losses=None):
        if self._check_quantiles_requirements(outputs, losses):
            outputs = transpose_first_to_last(outputs, name='r6')
            return QuantileTensor(outputs, quantiles=True)
        return QuantileTensor(outputs, quantiles=False)

    def _shape_pipeline(self, outputs, losses=None):
        if self._check_quantiles_requirements(outputs, losses):
            return tf.TensorShape(tf.nest.flatten([outputs.as_list()[1:], outputs.as_list()[0]]))
        return outputs

    def _keras_tensor_pipeline(self, outputs, losses=None):
        if self._check_quantiles_requirements(outputs, losses):
            return transpose_first_to_last(outputs)
        return outputs

    def _check_quantiles_requirements(self, outputs, losses=None):
        if self._apply_quantiles_transform and self.has_quantiles and losses:
            losses = convert_to_list(losses)

            # TODO: optimization, this calculation is made twice (One in _handle_quantiles_dim_in_losses_and_metrics)
            quantiles_in_losses = [loss.fn.quantiles[0] for loss in losses if hasattr(loss.fn, 'quantiles')]

            check_uniform_quantiles = self._check_uniform_quantiles_through_losses(quantiles_in_losses)
            check_quantiles_in_outputs = self._check_quantiles_in_outputs(outputs)

            if not check_uniform_quantiles:
                raise ValueError(f"`quantiles` has to be identical through losses. Got losses {quantiles_in_losses}.")

            elif not any(quantiles_in_losses) and check_quantiles_in_outputs:
                raise ValueError(f"It is not allowed to train a quantile model without a quantile loss. Got a loss {self.loss} and an output shape {outputs.shape}.")

            elif any(quantiles_in_losses) and not check_quantiles_in_outputs:
                raise ValueError(f"It is not allowed to train a no quantile model with a quantile loss. Got a loss {self.loss} and an output shape {outputs.shape}.")
            
            elif any(quantiles_in_losses) and check_quantiles_in_outputs:
                if not self._check_quantiles_in_outputs_vs_losses(outputs, quantiles_in_losses):
                    raise ValueError(f"Quantiles in losses and outputs are not the same, got outputs shape: {outputs.shape} and quantiles in losses: {quantiles_in_losses}")
                return True
        
        return False

    def _check_uniform_quantiles_through_losses(self, quantiles_in_losses):
        """Return True if every losses define an identical `quantiles` attribute"""
        if len(quantiles_in_losses) == 0: # Case of no quantiles in losses
            return True
        return all(q == quantiles_in_losses[idx - 1] for idx, q in enumerate(quantiles_in_losses))

    def _check_quantiles_in_outputs(self, outputs):
        """Return True if the outputs contains a `quantiles` dimension"""
        # TODO: find an other way to find if an outputs contains quantiles dimension
        return any(s == outputs.shape[:len(s)] for s in self._additional_shapes)

    def _check_quantiles_in_outputs_vs_losses(self, outputs, quantiles_in_losses):
        """Return True if the outputs and the quantile loss have the same `quantiles` attribute"""
        return len(quantiles_in_losses[0]) == outputs.shape[0]

    def get_additional_shapes(self, index):
        """Return shape"""
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
        """Return True if quantiles exists else False"""
        return self._has_quantiles


def _remove_dimension_to_ypred(fn):
    """We remove the quantile dimension from y_pred if it is not needed,
    then y_true and y_pred are broadcastable.
    """

    def new_fn(y_true, y_pred, *args, **kwargs):

        if y_pred.quantiles and y_pred.shape.rank > y_true.shape.rank:
            q = math.ceil(y_pred.shape[-1] / 2)
            y_pred = y_pred[..., q]

        print('strided loss', y_true)

        return fn(y_true, y_pred, *args, **kwargs)

    return new_fn


def _add_dimension_to_ytrue(fn, obj):
    """We add the quantile dimension from y_true if it is needed,
    then y_true and y_pred are broadcastable.
    """

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
        self._is_multivariate = False
        self._n_variates = 0

        self._is_multivariate_built = False
        self._n_variates_built = False

    def _preprocessing_wrapper(self, inputs):
        self.init_univariate_params(inputs.shape)
        if self._apply_multivariate_transpose and self.is_multivariate and not []:
            print(inputs, tf.nest.map_structure(transpose_last_to_first, inputs))
            return tf.nest.map_structure(transpose_last_to_first, inputs)
        return inputs

    def _tensor_pipeline(self, outputs, losses=None):
        outputs = super()._tensor_pipeline(outputs, losses)
        if self._check_multivariates_requirements():
            outputs = tf.nest.map_structure(lambda outputs: transpose_first_to_second_last(outputs, name='r3') if outputs.quantiles else transpose_first_to_last(outputs, 'r4'), outputs)
            outputs = tf.nest.map_structure(convert_to_univariate_tensor(multivariates=True), outputs)
            return outputs
        
        outputs = tf.nest.map_structure(convert_to_univariate_tensor(multivariates=False), outputs)
        return outputs

    def _shape_pipeline(self, outputs, losses=None):
        outputs = super()._shape_pipeline(outputs, losses)
        if self._check_multivariates_requirements():
            return tf.TensorShape(tf.nest.flatten([outputs.as_list()[1:], outputs.as_list()[0]]))
        return outputs

    def _keras_tensor_pipeline(self, outputs, losses=None):
        outputs = super()._keras_tensor_pipeline(outputs, losses)

        # TODO: Find a way to use the quantiles of QuantileTensor
        check_multivariates = self._check_multivariates_requirements()
        check_quantiles = self._check_quantiles_requirements(outputs, losses)
        if check_multivariates and check_quantiles:
            return transpose_first_to_second_last(outputs, name='r')
        elif check_multivariates:
            return transpose_first_to_last(outputs, name='r2')
        return outputs

    def _check_multivariates_requirements(self):
        return self._apply_multivariate_transpose and self.is_multivariate

    def init_univariate_params(self, input_shape):
        """It can be called inside `build` method to initiate the multivariates attributes.
        
        If not then it is called inside `__call__` method.

        Parameters
        ----------
        input_shape : tf.TensorShape
            The shape of the input tensor.
        """

        if not self._is_multivariate_built or not self._n_variates_built: # If used in build then in call it's disabled
            self.set_is_multivariate(input_shape)
            self.set_n_variates(input_shape)

    def set_is_multivariate(self, input_shape):
        """Initiate `is_multivariate` attribute"""

        self._is_multivariate = bool(input_shape.rank > 2)
        self._is_multivariate_built = True

    def set_n_variates(self, input_shape):
        """Initiate `n_variates` attribute"""

        if self.is_multivariate:
            self._n_variates = input_shape[-1]
            self._additional_shapes = [s + [self._n_variates] for s in self._additional_shapes]
        self._n_variates_built = True
        
    @property
    def is_multivariate(self):
        if not self._is_multivariate_built:
            raise ValueError('`init_univariate_build` has not been called.')
        return self._is_multivariate

    @property
    def n_variates(self):
        if not self._n_variates_built:
            raise ValueError('`init_univariate_build` has not been called.')
        return self._n_variates


def convert_to_univariate_tensor(multivariates):
    def fn(tensor):
        return UnivariateTensor(values=tensor, multivariates=multivariates)
    return fn


def _is_nested_type(inputs, type):  # pylint: disable=unused-argument
    """Check the arguments to see if we are constructing a functional model."""
    # We are constructing a functional model if any of the inputs
    # are KerasTensors
    return any(
        isinstance(tensor, type)
        for tensor in tf.nest.flatten([inputs]))


class QuantileTensor(tf.experimental.ExtensionType):
    values: tf.Tensor
    quantiles: bool
    shape: tf.TensorShape
    dtype: tf.DType 

    def __init__(self, values, quantiles: bool, shape=None, dtype=None):
        self.values = values
        self.quantiles = quantiles
        self.shape = shape or self.values.shape
        self.dtype = dtype or self.values.dtype

    def __getitem__(self, key):
        print(key)
        return QuantileTensor(self.values.__getitem__(key), quantiles=self.quantiles)

    def __sub__(self, tensor):
        return QuantileTensor(self.values.__sub__(tensor), quantiles=self.quantiles)

    @property
    def rank(self):
        return self.values.rank


class UnivariateTensor(QuantileTensor):
    values: tf.Tensor
    quantiles: bool
    multivariates: bool
    shape: tf.TensorShape
    dtype: tf.DType

    def __init__(self, values: Union[QuantileTensor, tf.Tensor], multivariates: bool, quantiles: bool=None, shape=None, dtype=None):
        is_quantile_tensor = isinstance(values, QuantileTensor)
        self.values = values.values if is_quantile_tensor else values
        self.quantiles = values.quantiles if is_quantile_tensor else quantiles
        self.multivariates = multivariates
        self.shape = shape or self.values.shape
        self.dtype = dtype or self.values.dtype

    def __getitem__(self, key):
        return UnivariateTensor(self.values.__getitem__(key), quantiles=self.quantiles, multivariates=self.multivariates)

    def __sub__(self, tensor):
        return UnivariateTensor(self.values.__sub__(tensor), quantiles=self.quantiles, multivariates=self.multivariates)


@tf.experimental.dispatch_for_api(tf.linalg.matmul)
def matmul(a: Union[QuantileTensor, UnivariateTensor, tf.Tensor, tf.Variable], b: Union[QuantileTensor, UnivariateTensor, tf.Tensor, tf.Variable], transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False, a_is_sparse=False, b_is_sparse=False, output_type=None, name=None):
    x_values = a.values if isinstance(a, (UnivariateTensor, QuantileTensor)) else a
    y_values = b.values if isinstance(b, (UnivariateTensor, QuantileTensor)) else b
    if (x_is_univariate := isinstance(a, UnivariateTensor)) or isinstance(b, UnivariateTensor):
        a = a if x_is_univariate else b
        return UnivariateTensor(tf.matmul(x_values, y_values, transpose_a, transpose_b, adjoint_a, adjoint_b, a_is_sparse, b_is_sparse, output_type, name), quantiles=a.quantiles, multivariates=a.multivariates, shape=a.shape)
    elif (x_is_quantile := isinstance(a, QuantileTensor)) or isinstance(b, QuantileTensor):
        a = a if x_is_quantile else b
        return QuantileTensor(tf.matmul(x_values, y_values, transpose_a, transpose_b, adjoint_a, adjoint_b, a_is_sparse, b_is_sparse, output_type, name), quantiles=a.quantiles, shape=a.shape)
    return tf.matmul(x_values, y_values, transpose_a, transpose_b, adjoint_a, adjoint_b, a_is_sparse, b_is_sparse, output_type, name)


@tf.experimental.dispatch_for_api(tf.convert_to_tensor)
def convert_to_tensor(value: Union[QuantileTensor, UnivariateTensor], dtype=None, dtype_hint=None, name=None):
    if isinstance(value, UnivariateTensor):
        return UnivariateTensor(tf.convert_to_tensor(value.values, dtype=dtype, dtype_hint=dtype_hint, name=name), quantiles=value.quantiles, multivariates=value.multivariates)
    elif isinstance(value, QuantileTensor):
        return QuantileTensor(tf.convert_to_tensor(value.values, dtype=dtype, dtype_hint=dtype_hint, name=name), quantiles=value.quantiles)


@tf.experimental.dispatch_for_api(tf.squeeze)
def convert_to_tensor(input: Union[QuantileTensor, UnivariateTensor], axis=None, name=None):
    if isinstance(input, UnivariateTensor):
        return UnivariateTensor(tf.squeeze(input.values, axis=axis, name=name), quantiles=input.quantiles, multivariates=input.multivariates)
    elif isinstance(input, QuantileTensor):
        return QuantileTensor(tf.squeeze(input.values, axis=axis, name=name), quantiles=input.quantiles)


@tf.experimental.dispatch_for_api(tf.concat)
def concat(values: List[Union[QuantileTensor, UnivariateTensor, tf.Tensor]], axis, name='concat'):
    val = [v.values if isinstance(v, (QuantileTensor, UnivariateTensor)) else v for v in values]
    quantiles = any(v.quantiles if isinstance(v, (QuantileTensor, UnivariateTensor)) else False for v in values)
    if any(isinstance(v, UnivariateTensor) for v in values):
        multivariates = any(v.multivariates if isinstance(v, (QuantileTensor, UnivariateTensor)) else False for v in values)
        return UnivariateTensor(tf.concat(val, axis=axis, name=name), quantiles=quantiles, multivariates=multivariates)
    elif any(isinstance(v, QuantileTensor) for v in values):
        return QuantileTensor(tf.concat(val, axis=axis, name=name), quantiles=quantiles)
    
    return tf.concat(values, axis, name)


@tf.experimental.dispatch_for_api(tf.rank)
def rank(input: Union[QuantileTensor, UnivariateTensor], name=None):
    return tf.rank(input.values, name=name)


@tf.experimental.dispatch_for_api(tf.size)
def size(input: Union[QuantileTensor, UnivariateTensor], out_type=tf.int32, name=None):
    return tf.size(input.values, out_type=out_type, name=name)


@tf.experimental.dispatch_for_api(tf.shape)
def size(input: Union[QuantileTensor, UnivariateTensor], out_type=tf.int32, name=None):
    return tf.shape(input.values, out_type=out_type, name=name)


@tf.experimental.dispatch_for_api(tf.argmax)
def size(input: Union[QuantileTensor, UnivariateTensor], axis=None, output_type=tf.int64, name=None):
    return tf.argmax(input.values, axis=axis, output_type=output_type, name=name)


@tf.experimental.dispatch_for_api(tf.add_n)
def add_n(inputs: List[Union[QuantileTensor, UnivariateTensor, tf.Tensor]], name=None):
    val = [v.values if isinstance(v, (QuantileTensor, UnivariateTensor)) else v for v in inputs]
    quantiles = any([v.quantiles if isinstance(v, (QuantileTensor, UnivariateTensor)) else False for v in inputs])
    if any([isinstance(v, UnivariateTensor) for v in inputs]):
        multivariates = any([v.multivariates if isinstance(v, (QuantileTensor, UnivariateTensor)) else False for v in inputs])
        return UnivariateTensor(tf.add_n(val, name=name), quantiles=quantiles, multivariates=multivariates)
    elif any([isinstance(v, QuantileTensor) for v in inputs]):
        return QuantileTensor(tf.add_n(val, name=name), quantiles=quantiles)
    return tf.add_n(inputs, name)


@tf.experimental.dispatch_for_api(tf.transpose)
def transpose(a: Union[QuantileTensor, UnivariateTensor], perm=None, conjugate=False, name='transpose'):
    if isinstance(a, UnivariateTensor):
        return UnivariateTensor(tf.transpose(a.values, perm=perm, conjugate=conjugate, name=name), quantiles=a.quantiles, multivariates=a.multivariates)
    elif isinstance(a, QuantileTensor):
        return QuantileTensor(tf.transpose(a.values, perm=perm, conjugate=conjugate, name=name), quantiles=a.quantiles)


@tf.experimental.dispatch_for_api(tf.math.reduce_mean)
def reduce_mean(input_tensor: Union[QuantileTensor, UnivariateTensor], axis=None, keepdims=False, name=None):
    return tf.math.reduce_mean(input_tensor.values, axis=axis, keepdims=keepdims, name=name)


@tf.experimental.dispatch_for_api(tf.math.reduce_sum)
def reduce_sum(input_tensor: Union[QuantileTensor, UnivariateTensor], axis=None, keepdims=False, name=None):
    return tf.math.reduce_sum(input_tensor.values, axis=axis, keepdims=keepdims, name=name)


@tf.experimental.dispatch_for_unary_elementwise_apis(Union[QuantileTensor, UnivariateTensor])
def tensor_unary_elementwise_api_handler(api_func, x):
    if isinstance(x, UnivariateTensor):
        return UnivariateTensor(api_func(x.values), quantiles=x.quantiles, multivariates=x.multivariates)
    elif isinstance(x, QuantileTensor):
        return QuantileTensor(api_func(x.values), quantiles=x.quantiles)


@tf.experimental.dispatch_for_binary_elementwise_apis(Union[QuantileTensor, UnivariateTensor, tf.Tensor, tf.Variable], Union[QuantileTensor, UnivariateTensor, tf.Tensor, tf.Variable])
def tensor_binary_elementwise_api_handler(api_func, x, y):
    x_values = x.values if isinstance(x, (UnivariateTensor, QuantileTensor)) else x
    y_values = y.values if isinstance(y, (UnivariateTensor, QuantileTensor)) else y
    if (x_is_univariate := isinstance(x, UnivariateTensor)) or isinstance(y, UnivariateTensor):
        x = x if x_is_univariate else y
        return UnivariateTensor(api_func(x_values, y_values), quantiles=x.quantiles, multivariates=x.multivariates, shape=x.shape)
    elif (x_is_quantile := isinstance(x, QuantileTensor)) or isinstance(y, QuantileTensor):
        x = x if x_is_quantile else y
        return QuantileTensor(api_func(x_values, y_values), quantiles=x.quantiles, shape=x.shape)
    return api_func(x_values, y_values)
