# -*- coding: utf-8 -*-
"""Functions to assess loss.
"""

import tensorflow as tf
import numpy as np
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.utils import losses_utils
from autopycoin.utils import check_infinity


__ALL__ = [
    "smape",
    "mase",
    "owa",
    "MeanAbsoluteScaledError",
    "SymetricMeanAbsolutePercentageError",
    "OverallWeightedAverageError"
]


@tf.function
def smape(y_true, y_pred, mask=False):

    """Calculate the symmetric mean absolute percentage error between `y_true`and `y_pred`.
    
    parameters
    ----------
    y_true : ndarray or dataframe or list or tensor of shape `[batch_size, d0, .. dN]`.
        Ground truth values.

    y_pred : ndarray or dataframe or list or tensor of shape `[batch_size, d0, .. dN]`.
        The predicted values.

    mask : Boolean, Optional
        Define if infinite values need to be taken into account. Defaults to False.

    returns
    -------
    error : Tensor
        The error in %.
    """ 

    y_true = tf.cast(y_true, 'float')
    y_pred = tf.cast(y_pred, 'float')
    mask = tf.constant(mask)

    diff = tf.abs(y_true - y_pred)
    total = tf.abs(y_true)  +  tf.abs(y_pred)
    result = tf.cond(mask, lambda: check_infinity(diff/total),
                           lambda: diff/(total + tf.keras.backend.epsilon()))

    error = 200 * tf.reduce_sum(result, axis=1) / diff.shape[1]
    
    return error


@tf.function
def mase(y_true, y_pred, y_train, seasonality, mask=False):
    """Calculate the Mean Absolute Scaled Error.
    whereas sMAPE scales the error by the average between the forecast and ground truth, the MASE
    scales by the average error of the naïve predictor that simply copies the observation measured m
    periods in the past, thereby accounting for seasonality.
    
    parameters
    ----------
    y_true : ndarray or dataframe or list or Tensor of shape `[batch_size, d0, .. dN]`.
        Ground truth values.

    y_pred : ndarray or dataframe or list or Tensor of shape `[batch_size, d0, .. dN]`.
        The predicted values.

    y_train : ndarray or dataframe or list or Tensor of shape `[batch_size, d0, .. dN]`.
        observed series history (training values).

    seasonality : integer.
        The periodicity of the data.

    mask : Boolean, Optional
        Define if infinite values need to be taken into account. Defaults to False.

    returns
    -------
    error : Tensor of shape `[batch_size, d0, .. dN]`.
        The error in %.
    """ 

    
    y_true = tf.cast(y_true, 'float')
    y_pred = tf.cast(y_pred, 'float')
    y_train = tf.cast(y_train, 'float')
    mask = tf.constant(mask)

    diff = tf.abs(y_true - y_pred)
    y_total = tf.concat([y_train, y_true], axis=1)
    
    periodic_sum = tf.abs(y_total[:, seasonality:] - y_total[:, :-seasonality])
    seasonality = tf.cast(seasonality, 'float')
        
    states = tf.TensorArray(tf.float32, size=y_pred.shape[1])
    
    for i in tf.range(y_pred.shape[1]):
        state = tf.reduce_sum(periodic_sum[:, :(y_train.shape[1] + i)], axis=1) / (y_train.shape[1] + tf.cast(i+1, 'float') - seasonality)
        states = states.write(i, state)
        
    periodic_norm = tf.transpose(states.stack())
    error = tf.cond(mask, lambda: check_infinity(diff / periodic_norm),
                          lambda: diff / (periodic_norm + tf.keras.backend.epsilon()))

    error =  tf.reduce_sum(error, axis=1) / y_pred.shape[1]

    return error


@tf.function
def owa(y_true, y_pred, y_train, seasonality, smape_naive=1, mase_naive=1, mask=False):
    """Calculate the overall weighted average. This is a M4-
    specific metric used to rank competition entries (M4 Team, 2018b), where sMAPE and MASE metrics
    are normalized such that a seasonally-adjusted naïve forecast obtains OWA = 1.0.

    parameters
    ----------
    y_true : ndarray or dataframe or list or Tensor of shape `[batch_size, d0, .. dN]`.
        Ground truth values.

    y_pred : ndarray or dataframe or list or Tensor of shape `[batch_size, d0, .. dN]`.
        The predicted values.

    y_train : ndarray or dataframe or list or Tensor of shape `[batch_size, d0, .. dN]`.
        observed series history (training values).

    seasonality: integer.
        The periodicity of the data. 

    smape_naive : integer or Tensor of rank 0.
        The smape value for a naive model. Defaults to 1.

    mase_naive : integer or Tensor of rank 0.
        The mase value for a naive model. Defaults to 1.

    mask : Boolean, Optional
        Define if infinite values need to be taken into account. Defaults to False.

    returns
    -------
    error : Tensor of shape `[batch_size, d0, .. dN]`.
        The error in %.
    """

    mase_error= mase(y_true, y_pred, y_train, seasonality, mask=mask)
    smape_error = smape(y_true, y_pred, mask=mask)

    error = smape_error / smape_naive
    error = error + (mase_error/mase_naive)
    
    error = 1/2 * error
    
    return error

@tf.function
def quantile_loss(y_true, y_pred, quantiles):
    
    """Calculate the quantile loss function, summed across all quantile outputs.

    parameters
    ----------
    y_true : ndarray or dataframe or list or Tensor of shape `[batch_size, d0, .. dN]`.
        Ground truth values.

    y_pred : ndarray or dataframe or list or Tensor of shape `[batch_size, d0, .. dN]`.
        The predicted values.

    quantiles : ndarray or dataframe or list or Tensor of shape `[batch_size, d0, .. dN]`.
        The set of output quantiles on which is calculated the quantile loss.

    returns
    -------
    error : Tensor of shape `[batch_size, d0, .. dN]`.
        The error in %.
    """

    y_true = tf.cast(y_true, 'float')
    y_pred = tf.cast(y_pred, 'float')
    quantiles = tf.constant(quantiles, shape=(len(quantiles), 1, 1))
    quantile_loss = (quantiles * tf.clip_by_value(y_true-y_pred, 0., np.inf) + 
                     (1 - quantiles) * tf.clip_by_value(y_pred-y_true, 0., np.inf))
    
    M = y_pred.shape[0]
    result = quantile_loss / tf.cast(M, 'float')
    return tf.squeeze(tf.reduce_sum(result, tf.range(1, tf.rank(y_pred))))


class MeanAbsoluteScaledError(LossFunctionWrapper):
  """Calculate the mean absolute scaled error between `y_true`and `y_pred`.
  To avoid infinite error, we add epsilon value to null values. This behavior can be modified by setting mask to True.
  then, null instances are not taken into account in calculation.
  Standalone usage:
  >>> y_true = [[0., 1.], [0., 0.]]
  >>> y_pred = [[1., 1.], [1., 0.]]
  >>> y_train = [[0., 1.], [0., 0.]]
  >>> seasonality = 1
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
  >>> mase = MeanAbsoluteScaledError(y_train=y_train, seasonality=seasonality)
  >>> mase(y_true, y_pred).numpy()
  2500000.2
  >>> # Calling with 'sample_weight'.
  >>> mase(y_true, y_pred, sample_weight=[0.7, 0.3]).numpy()
  750000.2
  >>> # Using mask.
  >>> mase = MeanAbsoluteScaledError(y_train=y_train, seasonality=seasonality, mask=True)
  >>> mase(y_true, y_pred).numpy()
  0.25
  >>> # Using 'sum' reduction type.
  >>> mase = MeanAbsoluteScaledError(
  ...     reduction=tf.keras.losses.Reduction.SUM, y_train=y_train, seasonality=seasonality)
  >>> mase(y_true, y_pred).numpy()
  5000000.5
  >>> # Using 'none' reduction type.
  >>> mase = MeanAbsoluteScaledError(
  ...     reduction=tf.keras.losses.Reduction.NONE, y_train=y_train, seasonality=seasonality)
  >>> mase(y_true, y_pred).numpy()
  array([4.9999994e-01, 5.0000000e+06], dtype=float32)
  
  Usage with the `compile()` API:
  ```python
  model.compile(optimizer='sgd', loss=tf.keras.losses.MeanAbsoluteScaledError())
  ```
  """

  def __init__(self,
               y_train,
               seasonality,
               reduction=losses_utils.ReductionV2.AUTO,
               name='mase',
               mask=False):
    """Initializes `MeanAbsoluteScaledError` instance.

    parameters
    ----------
    y_train: ndarray or dataframe or list or Tensor of shape `[batch_size, d0, .. dN]`.
        Ground truth values.

    seasonality: integer.
        The periodicity of the data. 

    reduction : Type of `tf.keras.losses.Reduction`, Optional  
        Type of `tf.keras.losses.Reduction`to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.

    name : string, Optional
        name for the op. Defaults to 'mase'.

    mask : Boolean, Optional
        Define if infinite values need to be taken into account. Defaults to False.
    """
    super(MeanAbsoluteScaledError, self).__init__(
        mase, y_train=y_train, seasonality=seasonality, name=name, reduction=reduction, mask=mask)


class SymetricMeanAbsolutePercentageError(LossFunctionWrapper):
  """Calculate the symetric mean absolute percentage error between `y_true`and `y_pred`.
  To avoid infinite error, we add epsilon value to null values. This behavior can be modified by setting mask to True.
  then, null instances are not taken into account in calculation.
  Standalone usage:
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
  50.0
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
  
  Usage with the `compile()` API:
  ```python
  model.compile(optimizer='sgd', loss=tf.keras.losses.SymetricMeanAbsolutePercentageError())
  ```

  """

  def __init__(self,
               reduction=losses_utils.ReductionV2.AUTO,
               name='smape',
               mask=False):
    """Initializes `SymetricMeanAbsolutePercentageError` instance.

    parameters
    ----------
    reduction : Type of `tf.keras.losses.Reduction`, Optional  
        Type of `tf.keras.losses.Reduction`to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.

    name : string, Optional
        name for the op. Defaults to 'smape'.

    mask : Boolean, Optional
        Define if infinite values need to be taken into account. Defaults to False.

    """
    super(SymetricMeanAbsolutePercentageError, self).__init__(
        smape, name=name, reduction=reduction, mask=mask)


class OverallWeightedAverageError(LossFunctionWrapper):
  """Calculate the overall weighted average error between `y_true`and `y_pred`.
  To avoid infinite error, we add epsilon value to null values. This behavior can be modified by setting mask to True.
  then, null instances are not taken into account in calculation.
  Standalone usage:
  >>> y_true = [[0., 1.], [0., 0.]]
  >>> y_pred = [[1., 1.], [1., 0.]]
  >>> y_train = [[0., 1.], [0., 0.]]
  >>> seasonality = 1
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
  >>> owa = OverallWeightedAverageError(y_train=y_train, seasonality=seasonality)
  >>> owa(y_true, y_pred).numpy()
  1250050.1
  >>> # Calling with 'sample_weight'.
  >>> owa(y_true, y_pred, sample_weight=[0.7, 0.3]).numpy()
  375025.1
  >>> # Using mask.
  >>> owa = OverallWeightedAverageError(y_train=y_train, seasonality=seasonality, mask=True)
  >>> owa(y_true, y_pred).numpy()
  25.125
  >>> # Using 'sum' reduction type.
  >>> owa = OverallWeightedAverageError(
  ...     reduction=tf.keras.losses.Reduction.SUM, y_train=y_train, seasonality=seasonality)
  >>> owa(y_true, y_pred).numpy()
  2500100.2
  >>> # Using 'none' reduction type.
  >>> owa = OverallWeightedAverageError(
  ...     reduction=tf.keras.losses.Reduction.NONE, y_train=y_train, seasonality=seasonality)
  >>> owa(y_true, y_pred).numpy()
  array([5.0249992e+01, 2.5000500e+06], dtype=float32)
  
  Usage with the `compile()` API:
  ```python
  model.compile(optimizer='sgd', loss=tf.keras.losses.MeanAbsoluteError())
  ```
  """

  def __init__(self,
               y_train,
               seasonality,
               reduction=losses_utils.ReductionV2.AUTO,
               name='owa',
               mask=False):
    """Initializes `OverallWeightedAverage` instance.

    parameters
    ----------
    y_train: ndarray or dataframe or list or Tensor of shape `[batch_size, d0, .. dN]`.
        Ground truth values.

    seasonality: integer.
        The periodicity of the data. 

    reduction : Type of `tf.keras.losses.Reduction`, Optional  
        Type of `tf.keras.losses.Reduction`to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.

    name : string, Optional
        name for the op. Defaults to 'owa'.

    mask : Boolean, Optional
        Define if infinite values need to be taken into account. Defaults to False.
    """
    super(OverallWeightedAverageError, self).__init__(
        owa, y_train=y_train, seasonality=seasonality, name=name, reduction=reduction, mask=mask)

    
class QuantileLossError(LossFunctionWrapper):
  """Calculate the quantile loss error between `y_true`and `y_pred` across all examples.
  Standalone usage:
  >>> y_true = [[0., 1.], [0., 0.]]
  >>> y_pred = [[1., 1.], [1., 0.]]
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
  >>> ql = QuantileLossError(quantiles=[0.1, 0.5, 0.9])
  >>> ql(y_true, y_pred).numpy()
  1.5
  
  Usage with the `compile()` API:
  ```python
  model.compile(optimizer='sgd', loss=tf.keras.losses.QuantileLossError())
  ```
  """

  def __init__(self,
               quantiles,
               reduction=losses_utils.ReductionV2.SUM,
               name='quantile_loss'):
    """Initializes `OverallWeightedAverage` instance.

    parameters
    ----------
    quantiles : ndarray or dataframe or list or Tensor of shape `[batch_size, d0, .. dN]`.
        The set of output quantiles on which is calculated the quantile loss.

    reduction : Type of `tf.keras.losses.Reduction`, Optional  
        Type of `tf.keras.losses.Reduction`to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.

    name : string, Optional
        name for the op. Defaults to 'quantile_loss'.
    """
    super(QuantileLossError, self).__init__(
        quantile_loss, quantiles=quantiles, name=name, reduction=reduction)