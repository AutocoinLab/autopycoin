"""
Overloading Model tensorflow object
"""

from typing import List, Union, Callable, Any
import itertools

import keras
import tensorflow as tf
from keras.losses import LossFunctionWrapper, Loss
from keras.engine import data_adapter
from keras.metrics import MeanMetricWrapper, SumOverBatchSizeMetricWrapper

from ..losses import LossQuantileDimWrapper
from ..metrics import MetricQuantileDimWrapper
from ..layers.base_layer import QuantileLayer, UnivariateLayer
from ..constant import TENSOR_TYPE
from .. import AutopycoinBaseModel
from ..extension_type import QuantileTensor, UnivariateTensor
from ..utils import (
    convert_to_list,
    quantiles_handler,
    transpose_first_to_last,
    transpose_first_to_second_last,
    transpose_last_to_first,
)


class BaseModel(keras.Model, AutopycoinBaseModel):
    """Base model which defines pre/post-processing methods to override.

    This model aims to be inherited and brings six methods.
    - preprocessing : Preprocess the inputs data
    - post_processing : Preprocess the outputs data
    - init_params : initialize parameters before `build` method
    - metrics_wrapper : Preprocess y_true or y_pred
    - losses_wrapper : Preprocess y_true or y_pred
    This three wrappers have to be overriden
    - Typing check.
    """

    NOT_INSPECT = ['call', 'build', 'compile']

    def __init__(self, *args: list, **kwargs: dict):
        super().__init__(*args, **kwargs)

    def _preprocessing_wrapper(self, inputs: TENSOR_TYPE) -> TENSOR_TYPE:
        return self.preprocessing(inputs)

    def preprocessing(self, inputs: TENSOR_TYPE) -> None:
        """Public API to apply preprocessing logics to your inputs data."""

        raise NotImplementedError("`preprocessing` has to be overriden.")

    def _post_processing_wrapper(self, outputs: TENSOR_TYPE) -> TENSOR_TYPE:
        """Post-processing wrapper."""

        self.handle_dim_in_losses_and_metrics(outputs)

        losses = getattr(self.compiled_loss, "_losses", None)
        losses_is_nested = tf.nest.is_nested(losses)
        outputs_is_nested = tf.nest.is_nested(outputs)

        outputs = tuple(outputs) if outputs_is_nested else (outputs,)

        # Case 1: multi outputs != multi losses or no losses
        if losses_is_nested and len(outputs) == len(losses):
            outputs = tf.nest.map_structure(
                lambda output, loss: self.post_processing(output, losses=loss),
                outputs,
                tuple(losses),
            )

        # Case 2: multi outputs = multi losses
        else:
            outputs = tf.nest.map_structure(
                lambda output: self.post_processing(output, losses=losses), outputs
            )

        return outputs[0] if len(outputs) == 1 else outputs


    def post_processing(self, output: TENSOR_TYPE, loss: Union[LossFunctionWrapper, Loss]) -> None:
        """Public API to apply post-processing logics to your outputs data."""

        raise NotImplementedError("`post_processing` has to be overriden.")

    def handle_dim_in_losses_and_metrics(self, outputs: Union[List[List[TENSOR_TYPE]], TENSOR_TYPE]) -> None:
        """Build and wrap losses and metrics."""

        if self.compiled_loss:
            if not self.compiled_loss.built:
                self.compiled_loss.build(outputs)
            self.compiled_loss._losses = tf.nest.map_structure(
                self.losses_wrapper, self.compiled_loss._losses
            )

        if self.compiled_metrics:
            if not self.compiled_metrics.built:
                self.compiled_metrics.build(outputs, outputs)
            self.compiled_metrics._metrics = tf.nest.map_structure(
                self.metrics_wrapper, self.compiled_metrics._metrics
            )

    def losses_wrapper(
        self, loss: Union[LossFunctionWrapper, Loss]
    ):
        """Wrap the `fn` function.

        See `tf.keras.losses.LossFunctionWrapper` docstring for more informations about `fn`.
        """

        raise NotImplementedError("`losses_wrapper` has to be overriden.")

    def metrics_wrapper(self, metrics: Any):
        """Wrap the update_state function.

        See `tf.keras.metrics.Metric` docstring for more informations about `update_state`.
        """

        raise NotImplementedError("`losses_wrapper` has to be overriden.")


class QuantileModel(BaseModel, QuantileLayer):  # pylint: disable=abstract-method
    """Overloads tensorflow Model class to integrate a `quantiles` attribute.

    During the compiling phase, the model checks the existence of the attribute `quantiles` in each loss function.
    If the test is positive then the model defines several attributes based on `quantiles` found in the loss functions.
    The model propagates the attributes associated to `quantiles` to the sublayers.
    Be carefull, if the check is positive the model is no more built.

    During the first call, all compiled losses and metrics are build and a second check is perfomed to ensure that
    Each output is not associated with different `quantiles` values otherwise it raises a ValueError.

    When subclassing this model, a pre/post-processing methods can be defined.
    Also a `post_processing` are already defined in order to transpose the `quantiles` dimensions.

    See :class:`autopycoin.layers.QuantileLayer` for more information for how to acces `quantiles` dimension in building phase.

    Attributes
    ----------
    has_quantiles : bool
        True if `quantiles` is not None else False. It is defined during compiling `method`.
        Default to False.
    quantiles : list[List[float]] or None
        It defines the quantiles used in the model.
        `quantiles` is a list of lists depending on the number of outputs the model computes.
        It is defined during compiling `method`.
        Default to None.
    n_quantiles : list[int] or int
        The number of quantiles the model computes.
        It is defined during compiling `method`.
        Default to 0.
    """

    NOT_INSPECT = ['_check_quantiles_requirements', 'call', 'build', 'compile']

    def __init__(
        self, quantiles: Union[None, list[float]]=None, *args: list, **kwargs: dict
    ) -> None:

        super().__init__(*args, **kwargs)

        self._set_quantiles(quantiles)

    def preprocessing(self, inputs: TENSOR_TYPE) -> TENSOR_TYPE:
        """No preprocessing for `QuantileModel`"""
        return inputs

    def post_processing(self, outputs: TENSOR_TYPE, losses: Union[None, LossFunctionWrapper, Loss, List[Union[LossFunctionWrapper, Loss]]]) -> TENSOR_TYPE:
        """Convert the outputs to `QuantileTensor` and apply transpose operation.

        The quantiles dimension is put to the last dimension to fit with keras norms.
        There is a difference with its equivalent Model implementation, we can't check with losses
        if they have a quantile attribute hence `apply_quantiles_transpose` is set to False by default
        and if you need to implement a layer with transpose operation you have to set it to True.

        The only check used is to ensure that quantile dimension is present in the outputs tensors.
        """

        if self._check_quantiles_requirements(outputs, losses):
            outputs = transpose_first_to_last(outputs)
            if outputs.shape[-1] == 1:
                outputs = tf.squeeze(outputs, axis=-1)
            return QuantileTensor(outputs, quantiles=True)
        return QuantileTensor(outputs, quantiles=False)

    def _check_quantiles_requirements(
        self,
        outputs: TENSOR_TYPE,
        losses: Union[None, LossFunctionWrapper, Loss, List[Union[LossFunctionWrapper, Loss]]]=None,
    ) -> bool:
        """Check if the requirements are valids else raise a ValueError.

        Raises
        ------
        ValueError:
            If the losses don't define a same `quantiles` attribute respectively to one output.
            If the output contains a `quantiles` dimension and there isn't at least one quantile loss.
            If the output don't contains a `quantiles` dimension and there is at least one quantile loss.
            If the output `quantiles` are not broadcastable with the losses `quantiles`
        """

        quantiles_in_outputs = self._check_quantiles_in_outputs(outputs) and self.has_quantiles
        if quantiles_in_outputs and self.compiled_loss:

            quantiles_in_losses = [
                getattr(loss, 'quantiles', None) for loss in convert_to_list(losses) if loss is not None
            ]

            if not quantiles_in_losses and self.compiled_loss:
                raise ValueError(
                    f"It is not allowed to train a quantile model without a quantile loss. Got a loss {losses} and an output shape {outputs.shape}."
                )

        return quantiles_in_outputs

    def _check_quantiles_in_outputs(self, outputs: TENSOR_TYPE) -> bool:
        """Return True if the outputs contains a `quantiles` dimension."""

        return self.additional_shape == outputs.shape[:len(self.additional_shape)]

    def losses_wrapper(
        self, loss: Union[LossFunctionWrapper, Loss]
    ) -> Union[None, LossQuantileDimWrapper]:
        """Add or remove the quantile dimension to y_pred and y_true respectively."""
        if (
            not isinstance(loss, (LossQuantileDimWrapper, type(None))) and self.has_quantiles
        ):

            kwargs = loss.get_config()

            if isinstance(loss, LossFunctionWrapper):
                kwargs['fn'] = loss.fn
            else:
                kwargs['fn'] = loss
            
            kwargs['quantiles'] = self.quantiles

            loss = LossQuantileDimWrapper(
                **kwargs
                )
        return loss

    def metrics_wrapper(
        self,
        metric: Union[
            None, keras.metrics.Metric, MeanMetricWrapper, SumOverBatchSizeMetricWrapper
        ],
    ) -> Union[None, MetricQuantileDimWrapper]:
        """Add or remove the quantile dimension to y_pred and y_true respectively."""

        if (
            not isinstance(metric, (MetricQuantileDimWrapper, type(None))) and self.has_quantiles
        ):

            metric = MetricQuantileDimWrapper(
                fn=metric,
                quantiles=self.quantiles
                )
        return metric


class UnivariateModel(UnivariateLayer, QuantileModel):
    """
    Wrapper around `QuantileModel` to integrate `n_variates` attributes.

    for the moment, if one of the inputs tensors is a multivariates tensor then
    all `additional_shapes` are extended by `n_variates`.
    In other words all layers extended by `additional_shapes` are multivariates layers.

    Attributes
    ----------
    is_multivariate : bool
        True if the inputs rank is higher than 2. Default to False.
    n_variates : list[None | int]
        the number of variates in the inputs. Default to [].
    """

    NOT_INSPECT = ['_check_quantiles_requirements', 'call', 'build', 'compile']

    def __init__(
        self, *args: list, **kwargs: dict
    ) -> None:
        super().__init__(*args, **kwargs)

        self._is_multivariate = False
        self._n_variates = []

    def preprocessing(
        self, inputs: TENSOR_TYPE
    ) -> Union[tf.Tensor, tf.Variable, UnivariateTensor]:
        """Init the multivariates attributes and transpose the `nvariates` dimension in first position."""

        inputs = super().preprocessing(inputs)
        if self.is_multivariate:
            return tf.nest.map_structure(transpose_last_to_first, inputs)
        return inputs

    def post_processing(self, outputs: TENSOR_TYPE, **kwargs: dict) -> TENSOR_TYPE:
        outputs = super().post_processing(outputs, **kwargs)
        if self.is_multivariate:
            if outputs.quantiles:
                fn = lambda outputs: transpose_first_to_second_last(outputs)
            else:
                fn = lambda outputs: transpose_first_to_last(outputs)

            outputs = tf.nest.map_structure(
                fn,
                outputs,
            )
            return tf.nest.map_structure(
                convert_to_univariate_tensor(multivariates=True), outputs
            )
        return tf.nest.map_structure(
            convert_to_univariate_tensor(multivariates=False), outputs
        )


def convert_to_univariate_tensor(multivariates):
    def fn(tensor):
        return UnivariateTensor(values=tensor, multivariates=multivariates)

    return fn
