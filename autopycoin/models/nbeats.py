# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 22:59:02 2021

@author: gaetd
"""

import tensorflow as tf
import numpy as np


class TrendBlock(tf.keras.layers.Layer):
    """ Trend block definition. Output layers are constrained which define polynomial function of small degree p.
    Therefore it is possible to get explanation from this block.
    
    Parameter
    ---------
    p_degree: integer
        Degree of the polynomial function.
    horizon: integer
        Horizon time to forecast.
    back_horizon: integer
        Past to rebuild.
    n_neurons: integer
        Number of neurons in Fully connected layers.
    """
    def __init__(self, 
                 horizon, 
                 back_horizon,
                 p_degree,   
                 n_neurons, 
                 n_quantiles, 
                 **kwargs):

        super().__init__(**kwargs)
        self._p_degree = tf.reshape(tf.range(p_degree, dtype='float32'), shape=(-1, 1)) # Shape (-1, 1) in order to broadcast horizon to all p degrees
        self._horizon = tf.cast(horizon, dtype='float32') 
        self._back_horizon = tf.cast(back_horizon, dtype='float32')
        self._n_neurons = n_neurons 
        self._n_quantiles = n_quantiles

        self.FC_stack = [tf.keras.layers.Dense(n_neurons, 
                                            activation='relu', 
                                            kernel_initializer="glorot_uniform") for _ in range(4)]
        
        self.dropout = tf.keras.layers.Dropout(0.1)
        
        self.FC_backcast = self.add_weight(shape=(n_neurons, p_degree), 
                                           initializer="glorot_uniform")
        self.FC_forecast = self.add_weight(shape=(n_quantiles, n_neurons, p_degree),
                                           initializer="glorot_uniform")

        self.forecast_coef = (tf.range(self._horizon) / self._horizon) ** self._p_degree
        self.backcast_coef = (tf.range(self._back_horizon) / self._back_horizon) ** self._p_degree
        
    def call(self, inputs):

        for dense in self.FC_stack:
            x = dense(inputs) # shape: (Batch_size, n_neurons)
            x = self.dropout(x) # We bind first layers by a dropout 
            
        theta_backcast = x @ self.FC_backcast # shape: (Batch_size, p_degree)
        theta_forecast = x @ self.FC_forecast # shape: (n_quantiles, Batch_size, p_degree)

        y_backcast = theta_backcast @ self.backcast_coef # shape: (Batch_size, backcast)
        y_forecast = theta_forecast @ self.forecast_coef # shape: (n_quantiles, Batch_size, forecast)
        
        return y_forecast, y_backcast
    

class SeasonalityBlock(tf.keras.layers.Layer):
    """
    Seasonality block definition. Output layers are constrained which define fourier series. 
    Each expansion coefficent then become a coefficient of the fourier serie. As each block and each 
    stack outputs are sum up, we decided to introduce fourier order and multiple seasonality periods.
    Therefore it is possible to get explanation from this block.
    
    Parameters
    ----------
    p_degree: integer
        Degree of the polynomial function.
    horizon: integer
        Horizon time to forecast.
    back_horizon: integer
        Past to rebuild.
    nb_neurons: integer
        Number of neurons in Fully connected layers.
    """
    def __init__(self,
                 horizon,
                 back_horizon,
                 n_neurons, 
                 periods, 
                 back_periods, 
                 forecast_fourier_order,
                 backcast_fourier_order,
                 quantiles, 
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self._horizon = horizon
        self._back_horizon = back_horizon
        self._periods = tf.cast(tf.reshape(periods, (1, -1, 1)), 'float32')
        self._back_periods = tf.cast(tf.reshape(back_periods, (1, -1, 1)), 'float32')
        self._forecast_fourier_order = tf.reshape(tf.range(forecast_fourier_order, dtype='float32'), shape=(-1, 1, 1))
        self._backcast_fourier_order = tf.reshape(tf.range(backcast_fourier_order, dtype='float32'), shape=(-1, 1, 1))

        # Workout the number of neurons needed to compute seasonality coefficients
        self._forecast_neurons = tf.reduce_sum(2 * periods)
        self._backcast_neurons = tf.reduce_sum(2 * back_periods)
        
        self.FC_stack = [tf.keras.layers.Dense(n_neurons, 
                                               activation='relu', 
                                               kernel_initializer="glorot_uniform") for _ in range(4)]
        
        self.dropout = tf.keras.layers.Dropout(0.1)
        
        self.FC_backcast = self.add_weight(shape=(n_neurons, self._backcast_neurons), 
                                           initializer="glorot_uniform")

        self.FC_forecast = self.add_weight(shape=(quantiles, n_neurons, self._forecast_neurons), 
                                           initializer="glorot_uniform")
        
        # Workout cos and sin seasonality coefficents
        time_forecast = tf.range(self._horizon, dtype='float32') / self._periods
        forecast_seasonality = 2 * np.pi * time_forecast * self._forecast_fourier_order
        forecast_seasonality = tf.concat((tf.cos(forecast_seasonality), 
                                          tf.sin(forecast_seasonality)), axis=0)

        time_backcast = tf.range(self._back_horizon, dtype='float32') / self._back_periods
        backcast_seasonality = 2 * np.pi * time_backcast * self._backcast_fourier_order
        backcast_seasonality = tf.concat((tf.cos(backcast_seasonality), 
                                          tf.sin(backcast_seasonality)), axis=0)
            
        self.forecast_coef = tf.constant(forecast_seasonality, 
                                         shape=(self._forecast_neurons, self._horizon),
                                         dtype='float32')
        
        self.backcast_coef = tf.constant(backcast_seasonality, 
                                         shape=(self._backcast_neurons, self._back_horizon), 
                                         dtype='float32')
        
    def call(self, inputs):

        for dense in self.FC_stack:
            x = dense(inputs) # shape: (Batch_size, n_neurons)
            x = self.dropout(x, training=True) # We bind first layers by a dropout 

        theta_backcast = x @ self.FC_backcast # shape: (Batch_size, 2 * fourier order)
        theta_forecast = x @ self.FC_forecast # shape: (quantiles, Batch_size, 2 * fourier order)

        y_backcast = theta_backcast @ self.forecast_coef # shape: (Batch_size, backcast)
        y_forecast = theta_forecast @ self.backcast_coef # shape: (quantiles, Batch_size, forecast)
    
        return y_forecast, y_backcast


class Stack(tf.keras.layers.Layer):
    """A stack is a series of blocks where each block produce two outputs, the forecast and the backcast. 
    All of the outputs are sum up which compose the stack output while each residual backcast is given to the following block.
    
    Parameters
    ----------
    blocks: keras Layer.
        blocks layers. they can be generic, seasonal or trend ones.

    """
    def __init__(self, blocks, **kwargs):
        
        super().__init__(self, **kwargs)

        self._blocks = blocks
                
    def call(self, inputs):

        y_forecast  = tf.constant([0.])
        for block in self._blocks:
            residual_y, y_backcast = block(inputs) # shape: (n_quantiles, Batch_size, forecast), (Batch_size, backcast)
            inputs = tf.keras.layers.Subtract()([inputs, y_backcast])
            y_forecast = tf.keras.layers.Add()([y_forecast, residual_y]) # shape: (n_quantiles, Batch_size, forecast)

        return y_forecast, inputs


class N_BEATS(tf.keras.Model):
    """This class compute the N-BEATS model. This is a univariate model which can be
     interpretable or generic. Its strong advantage resides in its structure which allows us 
     to extract the trend and the seasonality of temporal series available from the attributes
     `seasonality` and `trend`. This is an unofficial implementation.

     `@inproceedings{
        Oreshkin2020:N-BEATS,
        title={{N-BEATS}: Neural basis expansion analysis for interpretable time series forecasting},
        author={Boris N. Oreshkin and Dmitri Carpov and Nicolas Chapados and Yoshua Bengio},
        booktitle={International Conference on Learning Representations},
        year={2020},
        url={https://openreview.net/forum?id=r1ecqn4YwB}
        }`
    
    Parameters
    ----------
    stacks: keras Layer.
        stacks layers.
    """
    def __init__(self, 
                 stacks,
                 **kwargs):
                
        super().__init__(self, **kwargs)

        self._stacks = stacks

    def call(self, inputs):
        self._residuals_y = tf.TensorArray(tf.float32, size=len(self._stacks)) # Stock trend and seasonality curves during inference
        y_forecast = tf.constant([0.])
        for idx, stack in enumerate(self._stacks):
            residual_y, inputs = stack(inputs)
            self._residuals_y.write(idx, residual_y)
            y_forecast = tf.keras.layers.Add()([y_forecast, residual_y])

        return y_forecast

    @property
    def seasonality(self):
        return self._residuals_y.stack()[1:]

    @property
    def trend(self):
        return self._residuals_y.stack()[:1]