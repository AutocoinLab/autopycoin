"""
Functions to assess loss.
"""

import numpy as np
from typing import Callable, Union, List, Optional
import pandas as pd

import tensorflow as tf
from tensorflow.keras.backend import epsilon, maximum
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.utils import losses_utils

from ..utils import quantiles_handler
from .. import AutopycoinBaseClass


Yannotation = Union[tf.Tensor, pd.DataFrame, np.array, list]


def smape(y_true: Yannotation, y_pred: Yannotation):
    """
    Calculate the symmetric mean absolute percentage error between `y_true`and `y_pred`.

    Parameters
    ----------
    y_true : array, `dataframe`, list or `tensor of shape (batch_size, d0, .. dN)`
        Ground truth values.
    y_pred : array, `dataframe`, list or `tensor of shape (batch_size, d0, .. dN)`
        The predicted values.
    mask : bool, `Optional`
        set a mask to not take into account infinite values.
        Defaults to False.

    Returns
    -------
    error : `tensor`
        The error in %.

    Examples
    --------
    >>> from autopycoin.losses import smape
    >>> import tensorflow as tf
    >>> y_true = [[0., 1.], [0., 0.]]
    >>> y_pred = [[1., 1.], [1., 0.]]
    >>> smape(y_true, y_pred).numpy()
    array([99.999985, 99.999985], dtype=float32)
    """

    if not isinstance(y_pred, tf.RaggedTensor):
        y_pred = tf.convert_to_tensor(y_pred)

    y_true = tf.cast(y_true, dtype=y_pred.dtype)
    error = tf.abs(y_true - y_pred) / (
        maximum(tf.abs(y_true), epsilon()) + tf.abs(y_pred)
    )
    error = 200.0 * tf.reduce_mean(error, axis=-1)
    return error


def quantile_loss(
    y_true: Yannotation, y_pred: Yannotation, quantiles: List[float]
) -> tf.Tensor:
    """
    Calculate the quantile loss function, summed across all quantile outputs.

    Parameters
    ----------
    y_true : array, `dataframe`, list or `tensor of shape (batch_size, d0, .. dN)`
        Ground truth values.
    y_pred : array, `dataframe`, list or `tensor of shape (batch_size, d0, .. dN)`
        The predicted values.
    quantiles : list[float]
        The set of quantiles on which is calculated the quantile loss. The list is 1 dimension.

    Returns
    -------
    error : `tensor of shape (batch_size, d0, .. dN)`
        The error.

    Examples
    --------
    >>> from autopycoin.losses import quantile_loss
    >>> import tensorflow as tf
    >>> y_true = [[0., 1.], [0., 0.]]
    >>> y_pred = [[[1.], [1.]], [[1.], [0.]]]
    >>> quantile_loss(y_true, y_pred, quantiles=[0.5]).numpy()
    array([0.25, 0.25], dtype=float32)
    """

    if not isinstance(y_pred, tf.RaggedTensor):
        y_pred = tf.convert_to_tensor(y_pred)

    y_true = tf.cast(y_true, dtype=y_pred.dtype)
    quantiles = tf.convert_to_tensor(quantiles)

    if tf.rank(y_pred) > tf.rank(y_true):
        y_true = tf.expand_dims(y_true, -1)

    diff = y_pred - y_true
    q_loss = quantiles * tf.clip_by_value(diff, 0.0, np.inf) + (
        1 - quantiles
    ) * tf.clip_by_value(-diff, 0.0, np.inf)

    error = tf.reduce_mean(q_loss, axis=-2)
    return tf.reduce_sum(error, axis=[-1])


class SymetricMeanAbsolutePercentageError(LossFunctionWrapper, AutopycoinBaseClass):
    """
    Calculate the symetric mean absolute percentage error between `y_true` and `y_pred`.

    To avoid infinite error, we add epsilon value to zeros denominator.
    This behavior can be modified by setting mask to True.
    then, infinite instances are not taken into account in calculation.

    Parameters
    ----------
    reduction : `tf.keras.losses.Reduction, Optional`
        Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all
        cases this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training lotf such
        as `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
        https://www.tensorflow.org/tutorials/distribute/custom_training)
        for more details.
    name : str, `Optional`
        name for the op. Default to "smape".

    Examples
    --------
    >>> import tensorflow as tf
    >>> from autopycoin.losses import SymetricMeanAbsolutePercentageError
    >>> y_true = [[0., 1.], [0., 0.]]
    >>> y_pred = [[1., 1.], [1., 0.]]
    >>> # Using 'auto'/'sum_over_batch_size' reduction type.
    >>> smape = SymetricMeanAbsolutePercentageError()
    >>> smape(y_true, y_pred).numpy()
    99.999985
    >>> # Calling with 'sample_weight'.
    >>> smape(y_true, y_pred, sample_weight=[0.7, 0.3]).numpy()
    49.999992
    >>> # Using 'sum' reduction type.
    >>> smape = SymetricMeanAbsolutePercentageError(
    ...     reduction=tf.keras.losses.Reduction.SUM)
    >>> smape(y_true, y_pred).numpy()
    199.99997
    >>> # Using 'none' reduction type.
    >>> smape = SymetricMeanAbsolutePercentageError(
    ...     reduction=tf.keras.losses.Reduction.NONE)
    >>> smape(y_true, y_pred).numpy()
    array([100., 100.], dtype=float32)
    """

    def __init__(
        self,
        reduction: Optional[str] = losses_utils.ReductionV2.AUTO,
        name: Optional[str] = "smape",
    ):
        super().__init__(smape, name=name, reduction=reduction)


# TODO: ragged tensor
class QuantileLossError(LossFunctionWrapper, AutopycoinBaseClass):
    """
    Calculate the quantile loss error between `y_true` and `y_pred`
    across all examples.

    Parameters
    ----------
    quantiles : array, `dataframe`, list or `tensor of shape (batch_size, d0, .. dN)`
        The set of quantiles on which is calculated the
        quantile loss. The list needs to be in ascending order.
    reduction : `tf.keras.losses.Reduction, Optional`
        Type of `tf.keras.losses.Reduction`to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all
        cases this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training lotf such
        as `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
        https://www.tensorflow.org/tutorials/distribute/custom_training)
        for more details.
    name : str, `Optional`
        name for the op. Default to 'quantile_loss'.

    Returns
    -------
    error : `tensor of shape (batch_size, d0, .. dN)`
        The error.

    Attributes
    ----------
    quantiles : list[int]

    Examples
    --------
    >>> import tensorflow as tf
    >>> from autopycoin.losses import QuantileLossError
    >>> y_true = [[0., 1.], [0., 0.]]
    >>> y_pred = [[[1.], [1.]], [[1.], [0.]]]
    >>> # Using 'auto'/'sum_over_batch_size' reduction type.
    >>> ql = QuantileLossError(quantiles=[0.5])
    >>> ql(y_true, y_pred).numpy()
    0.5
    >>> # Calling with 'sample_weight'.
    >>> ql(y_true, y_pred, sample_weight=[0.7, 0.3]).numpy()
    0.25
    >>> # Using 'AUTO' reduction type.
    >>> ql = QuantileLossError(quantiles=[0.5],
    ...     reduction=tf.keras.losses.Reduction.AUTO)
    >>> ql(y_true, y_pred).numpy()
    0.25
    >>> # Using 'none' reduction type.
    >>> ql = QuantileLossError(quantiles=[0.5],
    ...     reduction=tf.keras.losses.Reduction.NONE)
    >>> ql(y_true, y_pred).numpy()
    array([0.25, 0.25], dtype=float32)
    >>> # Using multiple quantiles.
    >>> y_pred = [[[1.,1.,1.], [1.,1.,1.]], [[1.,1.,1.], [0.,0.,0.]]]
    >>> ql = QuantileLossError(quantiles=[0.1, 0.5, 0.9])
    >>> ql(y_true, y_pred).numpy()
    1.5
    """

    def __init__(
        self,
        quantiles: List[Union[float, int]],
        reduction: Optional[str] = losses_utils.ReductionV2.SUM,
        name: Optional[str] = "q_loss",
    ):

        self.quantiles = quantiles_handler(quantiles)

        super().__init__(
            quantile_loss, quantiles=self.quantiles, name=name, reduction=reduction
        )

    def _val___init__(self, output, *args, **kwargs):
        """Validates attributes and args for the init method."""
        quantiles = self._get_parameter(args, kwargs, name="quantiles", position=0)
        assert quantiles == [
            abs(quantile) for quantile in quantiles
        ], f"Negative quantiles are not allowed. got {quantiles}"


# TODO: write doc, test and use LossFunctionWrapper, autpycoinBaseClass.
class LagError(tf.keras.losses.Loss):
    """1 lag error MSE. Often used in time series analysis.

    Parameters
    ----------
    lag : int
    """

    def __init__(self, fn_loss: Callable, lag: int, **kwargs):
        super().__init__(**kwargs)
        self.lag = lag
        self.fn_loss = fn_loss

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        y_pred = y_pred[:, self.lag :] - y_pred[:, : -self.lag]
        y_true = y_true[:, self.lag :] - y_true[:, : -self.lag]
        return self.fn_loss(y_true, y_pred)
