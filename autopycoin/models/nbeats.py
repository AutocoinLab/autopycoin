"""
N-BEATS implementation
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

    Parameters
    ----------
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
        self, horizon, back_horizon, p_degree, n_neurons, quantiles, drop_rate, **kwargs
    ):

        super().__init__(**kwargs)
        self._p_degree = reshape(
            range(p_degree, dtype="float32"), shape=(-1, 1)
        )  # Shape (-1, 1) in order to broadcast horizon to all p degrees
        horizon = cast(horizon, dtype="float32")
        back_horizon = cast(back_horizon, dtype="float32")

        self.FC_stack = [Dense(n_neurons, activation="relu") for _ in range(4)]

        self.dropout = Dropout(drop_rate)

        shape_FC_backcast = (n_neurons, p_degree)
        self.FC_backcast = self.add_weight(
            shape=shape_FC_backcast, name="FC_backcast_trend"
        )

        shape_FC_forecast = (quantiles, n_neurons, p_degree)
        self.FC_forecast = self.add_weight(
            shape=shape_FC_forecast, name="FC_forecast_trend"
        )

        self.forecast_coef = range(horizon) / horizon
        self.forecast_coef = self.forecast_coef ** self._p_degree
        self.backcast_coef = (
            range(back_horizon) / back_horizon
        ) ** self._p_degree

    def call(self, inputs):

        for dense in self.FC_stack:
            x = dense(inputs)  # shape: (Batch_size, n_neurons)
            x = self.dropout(x)

        theta_backcast = x @ self.FC_backcast  # shape: (Batch_size, p_degree)

        # shape: (quantiles, Batch_size, p_degree)
        theta_forecast = x @ self.FC_forecast

        # shape: (Batch_size, backcast)
        y_backcast = theta_backcast @ self.backcast_coef

        # shape: (quantiles, Batch_size, forecast)
        y_forecast = theta_forecast @ self.forecast_coef

        return y_forecast, y_backcast


class SeasonalityBlock(Layer):
    """Seasonality block definition. Output layers are constrained which define
    fourier series.
    Each expansion coefficent then become a coefficient of the fourier serie.
    As each block and each
    stack outputs are sum up, we decided to introduce fourier order and
    multiple seasonality periods.
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
        drop_rate,
        **kwargs
    ):

        super().__init__(**kwargs)

        shape = (1, -1, 1)
        periods = cast(reshape(periods, shape), "float32")
        back_periods = cast(reshape(back_periods, shape), "float32")

        shape = (-1, 1, 1)
        forecast_fourier_order = reshape(
            range(forecast_fourier_order, dtype="float32"), shape
        )
        backcast_fourier_order = reshape(
            range(backcast_fourier_order, dtype="float32"), shape
        )

        # Workout the number of neurons needed to compute seasonality
        # coefficients
        forecast_neurons = reduce_sum(2 * periods)
        backcast_neurons = reduce_sum(2 * back_periods)

        self.FC_stack = [Dense(n_neurons, activation="relu") for _ in range(4)]

        self.dropout = Dropout(drop_rate)

        shape_FC_backcast = (n_neurons, backcast_neurons)
        self.FC_backcast = self.add_weight(
            shape=shape_FC_backcast, name="FC_backcast_seasonality"
        )

        shape_FC_forecast = (quantiles, n_neurons, forecast_neurons)
        self.FC_forecast = self.add_weight(
            shape=shape_FC_forecast, name="FC_forecast_seasonality"
        )

        # Workout cos and sin seasonality coefficents
        time_forecast = range(horizon, dtype="float32")
        time_forecast = 2 * np.pi * time_forecast / periods
        forecast_seasonality = time_forecast * forecast_fourier_order
        forecast_seasonality = concat(
            (cos(forecast_seasonality), sin(forecast_seasonality)), axis=0
        )

        time_backcast = range(back_horizon, dtype="float32")
        time_backcast = 2 * np.pi * time_backcast / back_periods
        backcast_seasonality = time_backcast * backcast_fourier_order
        backcast_seasonality = concat(
            (cos(backcast_seasonality), sin(backcast_seasonality)), axis=0
        )

        shape_forecast_coef = (forecast_neurons, horizon)
        self.forecast_coef = constant(
            forecast_seasonality,
            shape=shape_forecast_coef,
            dtype="float32",
        )

        shape_backcast_coef = (backcast_neurons, back_horizon)
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


class GenericBlock(Layer):
    """Generic block definition as described in the paper.
    We can't have explanation from this kind of block because g coefficients
    are learnt.

    Parameters
    ----------
    horizon: integer
        Horizon time to horizon.
    back_horizon: integer
        Past to rebuild.
    nb_neurons: integer
        Number of neurons in Fully connected layers.
    back_neurons: integer
        Number of back_horizon expansion coefficients.
    fore_neurons: integer
        Number of horizon expansion coefficients.
    """

    def __init__(
        self, horizon, back_horizon, n_neurons, quantiles, drop_rate, **kwargs
    ):

        super().__init__(**kwargs)

        self._FC_stack = [Dense(n_neurons, activation="relu") for _ in range(4)]

        self._dropout = Dropout(drop_rate)

        shape_FC_backcast = (n_neurons, n_neurons)
        self._FC_backcast = self.add_weight(
            shape=shape_FC_backcast, name="FC_backcast_generic"
        )

        shape_FC_forecast = (quantiles, n_neurons, n_neurons)
        self._FC_forecast = self.add_weight(
            shape=shape_FC_forecast, name="FC_forecast_generic"
        )

        self._backcast = Dense(back_horizon)

        self._forecast = Dense(horizon)

    def call(self, inputs):
        # shape: (Batch_size, back_horizon)
        X = inputs
        for dense_layer in self._FC_stack:
            # shape: (Batch_size, nb_neurons)
            X = dense_layer(X)
            X = self._dropout(X, training=True)

        # shape: (quantiles, Batch_size, 2 * fourier order)
        theta_forecast = X @ self._FC_forecast

        # shape: (Batch_size, 2 * fourier order)
        theta_backcast = X @ self._FC_backcast

        # shape: (Batch_size, back_horizon)
        y_backcast = self._backcast(theta_backcast)

        # shape: (quantiles, Batch_size, horizon)
        y_forecast = self._forecast(theta_forecast)

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

            # shape: (quantiles, Batch_size, forecast),
            # (Batch_size, backcast)
            residual_y, y_backcast = block(inputs)
            inputs = Subtract()([inputs, y_backcast])

            # shape: (quantiles, Batch_size, forecast)
            y_forecast = Add()([y_forecast, residual_y])

        return y_forecast, inputs


class NBEATS(Model):
    """
    Computes the N-BEATS model. 
    
    This is a univariate model which can be interpretable or generic. Its strong advantage 
    resides in its structure which allows us to extract the trend and the seasonality of
    temporal series available from the attributes `seasonality` and `trend`.
    This is an unofficial implementation.

    Parameters
    ----------
    stacks : list of :class:`autopycoin.models.Stack`
             Stacks can be created from :class:`autopycoin.models.TrendBlock`, 
             :class:`autopycoin.models.SeasonalityBlock` or :class:`autopycoin.models.GenericBlock`.
             See stack documentation for more details.

    Attributes
    ----------
    seasonality : Tensor-like
        Seasonality composent of the output.
    trend : Tensor-like
        Trend composent of the output.
    """

    def __init__(self, stacks, **kwargs):

        super().__init__(self, **kwargs)

        self._stacks = stacks

    def call(self, inputs):
        """
        Call method used to override special method __call__.

        It implements the logic of the nbeats model.
        """

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


def create_interpretable_nbeats(
    horizon,
    back_horizon,
    periods,
    back_periods,
    horizon_fourier_order,
    back_horizon_fourier_order,
    p_degree=1,
    trend_n_neurons=16,
    seasonality_n_neurons=16,
    quantiles=1,
    share=True,
    drop_rate=0.1,
    **kwargs
):
    """Wrapper to create interpretable model. check nbeats documentation
    to know more about Parameters."""

    if share is True:
        trend_block = TrendBlock(
            horizon=horizon,
            back_horizon=back_horizon,
            p_degree=p_degree,
            n_neurons=trend_n_neurons,
            quantiles=quantiles,
            drop_rate=drop_rate,
            **kwargs
        )

        seasonality_block = SeasonalityBlock(
            horizon=horizon,
            back_horizon=back_horizon,
            periods=periods,
            back_periods=back_periods,
            horizon_fourier_order=horizon_fourier_order,
            back_horizon_fourier_order=back_horizon_fourier_order,
            n_neurons=seasonality_n_neurons,
            quantiles=quantiles,
            drop_rate=drop_rate,
            **kwargs
        )

        trendblocks = [trend_block for _ in range(3)]
        seasonalityblocks = [seasonality_block for _ in range(3)]
    else:
        trendblocks = [
            TrendBlock(
                horizon=horizon,
                back_horizon=back_horizon,
                p_degree=p_degree,
                n_neurons=trend_n_neurons,
                quantiles=quantiles,
                drop_rate=drop_rate,
                **kwargs
            )
            for _ in range(3)
        ]
        seasonalityblocks = [
            SeasonalityBlock(
                horizon=horizon,
                back_horizon=back_horizon,
                periods=periods,
                back_periods=back_periods,
                horizon_fourier_order=horizon_fourier_order,
                back_horizon_fourier_order=back_horizon_fourier_order,
                n_neurons=seasonality_n_neurons,
                quantiles=quantiles,
                drop_rate=drop_rate,
                **kwargs
            )
            for _ in range(3)
        ]

    trendstacks = Stack(trendblocks)
    seasonalitystacks = Stack(seasonalityblocks)

    return NBEATS([trendstacks, seasonalitystacks])


def create_generic_nbeats(
    horizon,
    back_horizon,
    n_neurons,
    quantiles,
    n_blocks,
    n_stacks,
    share=True,
    drop_rate=0.1,
    **kwargs
):
    """Wrapper to create generic model. check nbeats documentation to know more
    about Parameters."""

    generic_stacks = []
    if share is True:
        for _ in range(n_stacks):
            generic_block = GenericBlock(
                horizon=horizon,
                back_horizon=back_horizon,
                n_neurons=n_neurons,
                quantiles=quantiles,
                drop_rate=drop_rate,
                **kwargs
            )

            generic_blocks = [generic_block for _ in range(n_blocks)]
            generic_stacks.append(Stack(generic_blocks))

    else:
        for _ in range(n_stacks):
            generic_blocks = [
                GenericBlock(
                    horizon=horizon,
                    back_horizon=back_horizon,
                    n_neurons=n_neurons,
                    quantiles=quantiles,
                    drop_rate=drop_rate,
                    **kwargs
                )
                for _ in range(n_blocks)
            ]

            generic_stacks.append(Stack(generic_blocks))

    return NBEATS(generic_stacks)
