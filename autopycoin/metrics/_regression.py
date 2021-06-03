# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


__ALL__ = [
    "mape",
    "smape",
    "mase",
    "owa"
]


def mape(y_true, y_pred):

    """
    Define the mean absolute percentage error.
    
        parameters
        ----------
        y_true : ndarray, DataFrame, list or Tensor.
            True values. 
        y_pred : ndarray, DataFrame, list or Tensor.
            Predicted values.

        returns
        -------
        error : Tensor
            Error of your model in %.
    """ 

    y_true = np.array(y_true, dtype='float')
    y_pred = np.array(y_pred, dtype='float')

    n = tf.abs(y_true - y_pred)
    d = tf.abs(y_true)
    error = 100 * tf.reduce_sum(n/d) / tf.dtypes.cast(tf.size(n), tf.float32)
    
    return error


def smape(y_true, y_pred):

    """
    Define the symmetric mean absolute percentage error.
    
        parameters
        ----------
        y_true : ndarray, DataFrame, list or Tensor.
            True values. 
        y_pred : ndarray, DataFrame, list or Tensor.
            Predicted values.

        returns
        -------
        error : Tensor
            Error of your model in %.
    """ 

    y_true = np.array(y_true, dtype='float')
    y_pred = np.array(y_pred, dtype='float')

    n = tf.abs(y_true - y_pred)
    d = tf.abs(y_true) +  tf.abs(y_pred)
    error = 200 * tf.reduce_sum(n/d) / tf.dtypes.cast(tf.size(n), tf.float32)
    
    return error


def mase(y_train, seasonality):
   
    """
    Define the Mean Absolute Scaled Error.
    whereas sMAPE scales the error by the average between the forecast and ground truth, the MASE
    scales by the average error of the naïve predictor that simply copies the observation measured m
    periods in the past, thereby accounting for seasonality.
    
        parameters
        ----------
        y_train : ndarray, DataFrame, list or Tensor.
            observed series history (Training values).
        seasonality: int
            Define the periodicity of the data. 
        y_true : ndarray, DataFrame, list or Tensor.
            True values. 
        y_pred : ndarray, DataFrame, list or Tensor.
            Predicted values.

        returns
        -------
        error : Tensor
            Error of your model in %.
    """ 
    
    def f(y_true, y_pred):
    
        y_true = np.array(y_true, dtype='float')
        y_pred = np.array(y_pred, dtype='float')
        y_t = np.array(y_train, dtype='float')

        n = tf.abs(y_true - y_pred)
        y = tf.concat([y_t, y_true], 0)
        naive = tf.reduce_sum([tf.abs(y_front - y_back) for y_front, y_back in zip(y[seasonality+1:], y[:-seasonality])])
        d =  naive / (tf.dtypes.cast(tf.size(y), tf.float32)-seasonality)
        error =  tf.reduce_sum(n/d) / tf.dtypes.cast(tf.size(n), tf.float32)

        return error 
    return f


def owa(y_train, seasonality):

    """
    Define the overall weighted average. This is a M4-
    specific metric used to rank competition entries (M4 Team, 2018b), where sMAPE and MASE metrics
    are normalized such that a seasonally-adjusted naïve forecast obtains OWA = 1.0.

        parameters
        ----------
        y_train : ndarray, DataFrame, list or Tensor.
            observed series history (Training values).
        seasonality: int
            Define the periodicity of the data. 
        y_true : ndarray, DataFrame, list or Tensor.
            True values. 
        y_pred : ndarray, DataFrame, list or Tensor.
            Predicted values.

        returns
        -------
        error : Tensor
            Error of your model in %.
    """

    def f(y_true, y_pred):

        y_true = np.array(y_true, dtype='float')
        y_pred = np.array(y_pred, dtype='float')
        y_t = np.array(y_train, dtype='float')
        
        smape_naive = 1
        mase_naive = 1

        mase_model = mase(y_t, seasonality)
        smape_model = smape(y_true, y_pred)

        error = (smape_model/smape_naive)
        error = error + (mase_model(y_true, y_pred)/mase_naive)
        
        error = 1/2 * error
        
        return error
    return f
    