# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 22:59:02 2021

@author: gaetd
"""

from tensorflow import (
    reshape,
    cast,
    range,
    reduce_sum,
    sin,
    cos,
    concat,
    float32,
    constant,
    TensorArray,
)
from tensorflow.keras.layers import Dense, Dropout, Layer, Subtract, Add
from tensorflow.keras import Model
import numpy as np


class TrendBlock(Layer):
    """Trend block definition. Output layers are constrained which define
    polynomial function of small degree p.
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

    def __init__(
        self, horizon, back_horizon, p_degree, n_neurons, n_quantiles, **kwargs
    ):

        super().__init__(**kwargs)
        self._p_degree = reshape(
            range(p_degree, dtype="float32"), shape=(-1, 1)
        )  # Shape (-1, 1) in order to broadcast horizon to all p degrees
        self._horizon = cast(horizon, dtype="float32")
        self._back_horizon = cast(back_horizon, dtype="float32")
        self._n_neurons = n_neurons
        self._n_quantiles = n_quantiles

        self.FC_stack = [Dense(n_neurons, activation="relu") for _ in range(4)]

        self.dropout = Dropout(0.1)

        shape_FC_backcast = (n_neurons, p_degree)
        self.FC_backcast = self.add_weight(shape=shape_FC_backcast)

        shape_FC_forecast = (n_quantiles, n_neurons, p_degree)
        self.FC_forecast = self.add_weight(shape=shape_FC_forecast)

        self.forecast_coef = range(self._horizon) / self._horizon
        self.forecast_coef = self.forecast_coef ** self._p_degree
        self.backcast_coef = (
            range(self._back_horizon) / self._back_horizon
        ) ** self._p_degree

    def call(self, inputs):

        for dense in self.FC_stack:
            x = dense(inputs)  # shape: (Batch_size, n_neurons)
            x = self.dropout(x)

        theta_backcast = x @ self.FC_backcast  # shape: (Batch_size, p_degree)

        # shape: (n_quantiles, Batch_size, p_degree)
        theta_forecast = x @ self.FC_forecast

        # shape: (Batch_size, backcast)
        y_backcast = theta_backcast @ self.backcast_coef

        # shape: (n_quantiles, Batch_size, forecast)
        y_forecast = theta_forecast @ self.forecast_coef

        return y_forecast, y_backcast


class SeasonalityBlock(Layer):
    """
    Seasonality block definition. Output layers are constrained which define
    fourier series.
    Each expansion coefficent then become a coefficient of the fourier serie.
    As each block and each
    stack outputs are sum up, we decided to introduce fourier order and multiple
    seasonality periods.
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

    def __init__(
        self,
        horizon,
        back_horizon,
        n_neurons,
        periods,
        back_periods,
        forecast_fourier_order,
        backcast_fourier_order,
        quantiles,
        **kwargs
    ):

        super().__init__(**kwargs)

        self._horizon = horizon
        self._back_horizon = back_horizon

        shape = (1, -1, 1)
        self._periods = cast(reshape(periods, shape), "float32")
        self._back_periods = cast(reshape(back_periods, shape), "float32")

        shape = (-1, 1, 1)
        self._forecast_fourier_order = reshape(
            range(forecast_fourier_order, dtype="float32"), shape
        )
        self._backcast_fourier_order = reshape(
            range(backcast_fourier_order, dtype="float32"), shape
        )

        # Workout the number of neurons needed to compute seasonality
        # coefficients
        self._forecast_neurons = reduce_sum(2 * periods)
        self._backcast_neurons = reduce_sum(2 * back_periods)

        self.FC_stack = [Dense(n_neurons, activation="relu") for _ in range(4)]

        self.dropout = Dropout(0.1)

        shape_FC_backcast = (n_neurons, self._backcast_neurons)
        self.FC_backcast = self.add_weight(shape=shape_FC_backcast)

        shape_FC_forecast = (quantiles, n_neurons, self._forecast_neurons)
        self.FC_forecast = self.add_weight(
            shape=shape_FC_forecast,
        )

        # Workout cos and sin seasonality coefficents
        time_forecast = range(self._horizon, dtype="float32") / self._periods
        time_forecast = 2 * np.pi * time_forecast
        forecast_seasonality = time_forecast * self._forecast_fourier_order
        forecast_seasonality = concat(
            (cos(forecast_seasonality), sin(forecast_seasonality)), axis=0
        )

        time_backcast = range(self._back_horizon, dtype="float32") / self._back_periods
        time_backcast = 2 * np.pi * time_backcast
        backcast_seasonality = time_backcast * self._backcast_fourier_order
        backcast_seasonality = concat(
            (cos(backcast_seasonality), sin(backcast_seasonality)), axis=0
        )

        shape_forecast_coef = (self._forecast_neurons, self._horizon)
        self.forecast_coef = constant(
            forecast_seasonality,
            shape=shape_forecast_coef,
            dtype="float32",
        )

        shape_backcast_coef = (self._backcast_neurons, self._back_horizon)
        self.backcast_coef = constant(
            backcast_seasonality,
            shape=shape_backcast_coef,
            dtype="float32",
        )

    def call(self, inputs):

        for dense in self.FC_stack:
            x = dense(inputs)  # shape: (Batch_size, n_neurons)
            x = self.dropout(x, training=True)

        # shape: (Batch_size, 2 * fourier order)
        theta_backcast = x @ self.FC_backcast

        # shape: (quantiles, Batch_size, 2 * fourier order)
        theta_forecast = x @ self.FC_forecast

        # shape: (quantiles, Batch_size, 2 * fourier order)
        y_backcast = theta_backcast @ self.forecast_coef

        # shape: (quantiles, Batch_size, forecast)
        y_forecast = theta_forecast @ self.backcast_coef

        return y_forecast, y_backcast


class Stack(Layer):
    """A stack is a series of blocks where each block produce two outputs,
    the forecast and the backcast.
    All of the outputs are sum up which compose the stack output while each
    residual backcast is given to the following block.

    Parameters
    ----------
    blocks: keras Layer.
        blocks layers. they can be generic, seasonal or trend ones.

    """

    def __init__(self, blocks, **kwargs):

        super().__init__(self, **kwargs)

        self._blocks = blocks

    def call(self, inputs):

        y_forecast = constant([0.0])
        for block in self._blocks:

            # shape: (n_quantiles, Batch_size, forecast), (Batch_size, backcast)
            residual_y, y_backcast = block(inputs)
            inputs = Subtract()([inputs, y_backcast])

            # shape: (n_quantiles, Batch_size, forecast)
            y_forecast = Add()([y_forecast, residual_y])

        return y_forecast, inputs


class N_BEATS(Model):
    """This class compute the N-BEATS model. This is a univariate model which
     can be interpretable or generic. Its strong advantage resides in its
     structure which allows us to extract the trend and the seasonality of
     temporal series available from the attributes `seasonality` and `trend`.
     This is an unofficial implementation.

     `@inproceedings{
        Oreshkin2020:N-BEATS,
        title={{N-BEATS}: Neural basis expansion analysis for interpretable
                          time series forecasting},
        author={Boris N. Oreshkin and Dmitri Carpov and Nicolas Chapados
                and Yoshua Bengio},
        booktitle={International Conference on Learning Representations},
        year={2020},
        url={https://openreview.net/forum?id=r1ecqn4YwB}
        }`

    Parameters
    ----------
    stacks: keras Layer.
        stacks layers.
    """

    def __init__(self, stacks, **kwargs):

        super().__init__(self, **kwargs)

        self._stacks = stacks

    def call(self, inputs):

        # Stock trend and seasonality curves during inference
        self._residuals_y = TensorArray(float32, size=len(self._stacks))
        y_forecast = constant([0.0])

        for idx, stack in enumerate(self._stacks):
            residual_y, inputs = stack(inputs)
            self._residuals_y.write(idx, residual_y)
            y_forecast = Add()([y_forecast, residual_y])

        return y_forecast

    @property
    def seasonality(self):
        return self._residuals_y.stack()[1:]

    @property
    def trend(self):
        return self._residuals_y.stack()[:1]
