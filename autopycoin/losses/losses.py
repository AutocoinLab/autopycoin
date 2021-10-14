"""
Functions to assess loss.
"""

import numpy as np

import tensorflow as tf
from tensorflow.keras.backend import epsilon
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.utils import losses_utils

from ..utils import check_infinity


def smape(y_true, y_pred, mask=False):
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

    mask = tf.convert_to_tensor(mask)

    diff = tf.abs(y_true - y_pred)
    total = tf.abs(y_true) + tf.abs(y_pred)

    error = tf.cond(
        mask,
        lambda: check_infinity(diff / total),
        lambda: diff / (total + epsilon()),
    )

    if isinstance(diff, tf.RaggedTensor):
        total_samples = tf.cast(error.row_lengths(), dtype=error.dtype)
    else:
        total_samples = diff.shape[1]

    error = 200 * tf.reduce_sum(error, axis=-1) / total_samples
    return error


def quantile_loss(y_true, y_pred, quantiles):
    """
    Calculate the quantile loss function, summed across all quantile outputs.

    Parameters
    ----------
    y_true : array, `dataframe`, list or `tensor of shape (batch_size, d0, .. dN)`
        Ground truth values.
    y_pred : array, `dataframe`, list or `tensor of shape (batch_size, d0, .. dN)`
        The predicted values.
    n_quantiles : array, `dataframe`, list or `tensor of shape (batch_size, d0, .. dN)`
        The set of quantiles on which is calculated the quantile loss.

    Returns
    -------
    error : `tensor of shape (batch_size, d0, .. dN)`
        The error.

    Examples
    --------
    >>> from autopycoin.losses import quantile_loss
    >>> import tensorflow as tf
    >>> y_true = [[0., 1.], [0., 0.]]
    >>> y_pred = [[[1., 1.], [1., 0.]]]
    >>> quantile_loss(y_true, y_pred, quantiles=[0.5]).numpy()
    array([0.25, 0.25], dtype=float32)
    """

    if not isinstance(y_pred, tf.RaggedTensor):
        y_pred = tf.convert_to_tensor(y_pred)

    y_true = tf.cast(y_true, dtype=y_pred.dtype)

    quantiles = tf.convert_to_tensor(quantiles)

    # Broadcast quantiles to y_true
    shape_broadcast = tf.concat(
        ([quantiles.shape[0]], tf.ones(tf.rank(y_pred) - 1, dtype=tf.int32)), axis=0
    )
    quantiles = tf.reshape(quantiles, shape=shape_broadcast)

    diff = y_pred - y_true
    q_loss = quantiles * tf.clip_by_value(diff, 0.0, np.inf) + (
        1 - quantiles
    ) * tf.clip_by_value(-diff, 0.0, np.inf)

    if isinstance(y_true, tf.RaggedTensor):
        total_samples = tf.cast(y_true.bounding_shape()[1], dtype=y_true.dtype)
    else:
        total_samples = y_true.shape[1]

    error = tf.math.divide(q_loss, total_samples)

    return tf.reduce_sum(error, axis=[0, -1])


class SymetricMeanAbsolutePercentageError(LossFunctionWrapper):
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
    mask : bool, `Optional`
        Define if infinite values need to be taken into account.
        Default to False.

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
    >>> # Using mask.
    >>> smape = SymetricMeanAbsolutePercentageError(mask=True)
    >>> smape(y_true, y_pred).numpy()
    100.0
    >>> # Using 'sum' reduction type.
    >>> smape = SymetricMeanAbsolutePercentageError(
    ...     reduction=tf.keras.losses.Reduction.SUM)
    >>> smape(y_true, y_pred).numpy()
    199.99997
    >>> # Using 'none' reduction type.
    >>> smape = SymetricMeanAbsolutePercentageError(
    ...     reduction=tf.keras.losses.Reduction.NONE)
    >>> smape(y_true, y_pred).numpy()
    array([99.999985, 99.999985], dtype=float32)
    """

    def __init__(
        self, reduction=losses_utils.ReductionV2.AUTO, name="smape", mask=False
    ):
        super().__init__(smape, name=name, reduction=reduction, mask=mask)


class QuantileLossError(LossFunctionWrapper):
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
    quantiles : list[:int]

    Examples
    --------
    >>> import tensorflow as tf
    >>> from autopycoin.losses import QuantileLossError
    >>> y_true = [[0., 1.], [0., 0.]]
    >>> y_pred = [[[1., 1.], [1., 0.]]]
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
    >>> y_pred = [[[1., 1.], [1., 0.]], [[1., 1.], [1., 0.]], [[1., 1.], [1., 0.]]]
    >>> ql = QuantileLossError(quantiles=[0.1, 0.5, 0.9])
    >>> ql(y_true, y_pred).numpy()
    1.5
    """

    def __init__(
        self, quantiles, reduction=losses_utils.ReductionV2.SUM, name="q_loss"
    ):

        super().__init__(
            quantile_loss, quantiles=quantiles, name=name, reduction=reduction
        )

        self.quantiles = quantiles
