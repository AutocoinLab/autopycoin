"""
Overloading Model tensorflow object
"""

import math
from typing import List, Union, Callable, Any
import itertools

import keras

import tensorflow.compat.v2 as tf
from keras.losses import LossFunctionWrapper

from ..utils import convert_to_list, quantiles_handler
from ..layers.base_layer import BaseLayer, QuantileLayer, UnivariateLayer
from ..constant import TENSOR_TYPE


class BaseModel(keras.Model, BaseLayer):
    """Base model which defines pre/post-processing methods to override.

    This model aims to be inherited and brings six functionality.
    - preprocessing : Preprocess the inputs data
    - post_processing : Preprocess the outputs data
    - init_params : initialize parameters before `build` method
    - metrics_wrapper : Preprocess y_true or y_pred
    - losses_wrapper : Preprocess y_true or y_pred
    This three wrappers have to be overriden
    - Typing check.
    """

    def __init__(self, *args: list, **kwargs: dict) -> None:
        super().__init__(*args, **kwargs)

    def _handle_dim_in_losses_and_metrics(self, outputs: TENSOR_TYPE) -> None:
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
        self, loss: LossFunctionWrapper
    ) -> Union[Callable, LossFunctionWrapper]:
        """Wrap the `fn` function.

        See `tf.keras.losses.LossFunctionWrapper` docstring for more informations about `fn`.
        """

        raise NotImplementedError("`losses_wrapper` has to be overriden")

    def metrics_wrapper(self, metrics: Any) -> Union[Callable, LossFunctionWrapper]:
        """Wrap the update_state function.

        See s`tf.keras.metrics.Metric` docstring for more informations about `update_state`.
        """

        raise NotImplementedError("`losses_wrapper` has to be overriden")

    def _post_processing_wrapper(self, outputs: TENSOR_TYPE) -> TENSOR_TYPE:
        """Post-processing wrapper.

        it handles the case of:
         - one tensor vs multi loss functions or no loss function
         - one tensor vs one loss function
        """

        self._handle_dim_in_losses_and_metrics(outputs)

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
            return outputs[0] if len(outputs) == 1 else outputs

        # Case 2: multi outputs = multi losses
        else:
            outputs = tf.nest.map_structure(
                lambda output: self.post_processing(output, losses=losses), outputs
            )
            return outputs[0] if len(outputs) == 1 else outputs


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

    NOT_INSPECT = ["compile", "build", "call"]

    def __init__(
        self, apply_quantiles_transpose: bool = True, *args: list, **kwargs: dict
    ) -> None:

        super().__init__(*args, **kwargs)

        self._has_quantiles = False
        self._quantiles = None
        self._n_quantiles = 0
        self._additional_shapes = [[]]
        self.apply_quantiles_transpose = apply_quantiles_transpose

    def compile(
        self,
        optimizer="rmsprop",
        loss=None,  # TODO: multiple loss one output (qloss, mse) -> leads to mse loss over estimated due to the quantiles -> raise an error? use a wrapper to select only the 0.5 quantile?
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=None,
        steps_per_execution=None,
        **kwargs,
    ):
        """Compile method from tensorflow.

        When compiling with losses defining a quantiles attribute
        it propagates this attribute to the submodels and sublayers.
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

    def _check_quantiles_in_loss(
        self,
        loss: Union[
            str,
            tf.keras.losses.Loss,
            LossFunctionWrapper,
            List[Union[str, tf.keras.losses.Loss, LossFunctionWrapper]],
        ],
    ) -> Union[List[Union[List[int], int]], None]:
        """Check if the loss functions define a `quantiles` attribute.

        If True then it returns the quantiles found.
        """

        # Case of multiple losses
        if isinstance(loss, (tuple, list)):
            quantiles = (
                self._check_quantiles_in_loss(loss_fn) for loss_fn in loss
            )
            quantiles = [q for q in quantiles if q]
            return list(itertools.chain.from_iterable(quantiles))

        # One loss
        elif hasattr(loss, "quantiles"):
            return quantiles_handler(loss.quantiles)

    # TODO: Avoid to rebuild weights when quantiles of model is not None
    def _set_quantiles(
        self,
        value: List[List[float]],
        additional_shapes: Union[None, List[List[int]]] = None,
        n_quantiles: Union[None, List[List[int]]] = None,
    ) -> None:
        """Set attributes linked to the quantiles found in the losses functions."""

        super()._set_quantiles(value, additional_shapes, n_quantiles)

        # Propagates to sublayers
        for idx, _ in enumerate(self.layers):
            if hasattr(self.layers[idx], "_set_quantiles"):
                self.layers[idx]._set_quantiles(
                    value, self._additional_shapes, self.n_quantiles
                )  # pylint: disable=protected-access

    def losses_wrapper(
        self, loss: LossFunctionWrapper
    ) -> Union[Callable, LossFunctionWrapper]:
        """Add or remove the quantile dimension to y_pred and y_true respectively."""

        # TODO: We override the fn function which can be a Loss instance and turn it into function.
        # As below we have to recreate an instance of the loss otherwise we lose informations as the attributes etc...
        if (
            not hasattr(loss, "quantiles")
            and not isinstance(loss, type(None))
            and not hasattr(loss, "_done")
        ):
            loss.fn = _remove_dimension_to_ypred(loss.fn)
            return loss
        elif not isinstance(loss, type(None)) and not hasattr(loss, "_done"):
            loss = LossFunctionWrapper(loss)
            loss = _add_dimension_to_ytrue(loss, type(loss))
        return loss

    def metrics_wrapper(
        self,
        metric: Union[
            None, str, keras.metrics.Metric, List[Union[str, keras.metrics.Metric]]
        ],
    ) -> Union[Callable, LossFunctionWrapper]:
        """Add or remove the quantile dimension to y_pred and y_true respectively."""

        # TODO: We override the update_state function which can be a Loss instance and turn it into function.
        # As below we have to recreate an instance of the loss otherwise we lose informations as the attributes etc...
        if not hasattr(metric, "quantiles") and not isinstance(metric, type(None)):
            metric.update_state = _remove_dimension_to_ypred(metric.update_state)
            return metric

        elif not isinstance(metric, type(None)):
            metric = _add_dimension_to_ytrue(metric, type(metric))

        return metric

    def _check_quantiles_requirements(
        self,
        outputs: TENSOR_TYPE,
        losses: Union[None, tf.keras.losses.Loss, List[tf.keras.losses.Loss]] = None,
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

        if self.has_quantiles and losses:
            losses = [loss.fn for loss in convert_to_list(losses)]

            # TODO: optimization, this calculation is made twice (One in _handle_quantiles_dim_in_losses_and_metrics)
            quantiles_in_losses = [
                loss.quantiles[0] for loss in losses if hasattr(loss, "quantiles")
            ]

            check_uniform_quantiles = self._check_uniform_quantiles_through_losses(
                quantiles_in_losses
            )
            check_quantiles_in_outputs = self._check_quantiles_in_outputs(outputs)

            if not check_uniform_quantiles:
                raise ValueError(
                    f"`quantiles` has to be identical through losses. Got losses {quantiles_in_losses}."
                )

            elif not any(quantiles_in_losses) and check_quantiles_in_outputs:
                raise ValueError(
                    f"It is not allowed to train a quantile model without a quantile loss. Got a loss {losses} and an output shape {outputs.shape}."
                )

            elif any(quantiles_in_losses):
                if self._compare_quantiles_in_outputs_and_losses(
                    outputs, quantiles_in_losses
                ):
                    return True

                elif self._is_single_quantile(quantiles_in_losses):
                    return False

                raise ValueError(
                    f"Quantiles in losses and outputs are not the same. Maybe you are trying to train a no quantile model "
                    f"with a quantile loss. It is possible only if there is one quantile defined as [[0.5]]. "
                    f"got outputs shape: {outputs.shape} and quantiles in losses: {quantiles_in_losses}"
                )

        return False

    def _check_uniform_quantiles_through_losses(
        self, quantiles_in_losses: List[List[Union[int, float]]]
    ) -> bool:
        """Return True if all losses define an identical `quantiles` attribute"""

        if len(quantiles_in_losses) == 0:  # Case of no quantiles in losses
            return True
        return all(
            q == quantiles_in_losses[idx - 1]
            for idx, q in enumerate(quantiles_in_losses)
        )

    def _compare_quantiles_in_outputs_and_losses(
        self, outputs: TENSOR_TYPE, quantiles_in_losses: List[List[Union[int, float]]]
    ) -> bool:
        """Return True if the outputs and the quantile loss have the same `quantiles` attribute"""

        return len(quantiles_in_losses[0]) == outputs.shape[0]

    def _is_single_quantile(
        self, quantiles_in_losses: List[List[Union[int, float]]]
    ) -> bool:
        return len(quantiles_in_losses[0]) == 1


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
    new_fn._done = True
    return new_fn


class UnivariateModel(QuantileModel, UnivariateLayer):
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

    def __init__(
        self, apply_multivariate_transpose: bool = True, *args: list, **kwargs: dict
    ) -> None:
        super().__init__(*args, **kwargs)

        self.apply_multivariate_transpose = apply_multivariate_transpose

    def init_params(
        self,
        inputs_shape: Union[tf.TensorShape, List[tf.TensorShape]],
        n_variates: Union[None, List[Union[None, int]]] = None,
        is_multivariate: Union[None, bool] = None,
        additional_shapes: Union[None, List[List[int]]] = None,
    ) -> None:
        """Initialize attributes related to univariate model.
        
        It is called before `build`.
        Three steps are done:
        - Filter the first shape in case of multiple inputs tensors.
        - Initialize attributes: `is_multivariate`, `n_variates`.
        - Add the n_variates dimension to `additional_shape` and propagate these attributes
        to the internal layers.
        """

        super().init_params(
            inputs_shape,
            n_variates=n_variates,
            is_multivariate=is_multivariate,
            additional_shapes=additional_shapes,
        )

        # Propagates to sublayers
        for idx, _ in enumerate(self.layers):
            if hasattr(self.layers[idx], "init_params"):
                self.layers[idx].init_params(
                    inputs_shape,
                    self.n_variates,
                    self.is_multivariate,
                    self._additional_shapes,
                )  # pylint: disable=protected-access
