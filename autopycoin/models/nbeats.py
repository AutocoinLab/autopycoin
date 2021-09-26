"""
N-BEATS implementation
"""

import tensorflow as tf
from tensorflow.keras.layers import Add, Dense, Dropout, InputSpec, Layer, Subtract
from tensorflow.keras import Model
import numpy as np

class BaseBlock(Layer):
    def __init__(self, 
        horizon,
        back_horizon,
        output_last_dim_forecast,
        output_last_dim_backcast,
        n_neurons,
        quantiles,
        drop_rate,
        **kwargs):

        super().__init__(**kwargs)

        self.horizon = float(horizon)
        self.back_horizon = float(back_horizon)
        self.n_neurons = int(n_neurons)
        self.quantiles = int(quantiles)
        self.drop_rate = float(drop_rate)
        self._output_last_dim_forecast = output_last_dim_forecast
        self._output_last_dim_backcast = output_last_dim_backcast

        if self.horizon < 0 or self.back_horizon < 0:
                raise ValueError(
                    f"`horizon` and `back_horizon` parameter expected "
                    f"a positive integer, got {horizon} and {back_horizon}."
                )

        if 0 > self.drop_rate > 1:
                raise ValueError(
                    f"Received an invalid value for `drop_rate`, expected "
                    f"a float between 0 and 1, got {drop_rate}."
                )
        if self.n_neurons < 0:
            raise ValueError(
                f"Received an invalid value for `n_neurons`, expected "
                f"a positive integer, got {n_neurons}."
            )
        if self.quantiles < 1:
            raise ValueError(
                f"Received an invalid value for `quantiles`, expected "
                f"an integer >= 1, got {quantiles}."
            )

    def build(self, input_shape):
        dtype = tf.as_dtype(self.dtype or tf.float32())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError(
                f"Unable to build `{self.name}` layer with "
                "non-floating point dtype %s" % (dtype,)
            )

        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError(
                "The last dimension of the inputs to `TrendBlock` "
                "should be defined. Found `None`."
            )

        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})

        self.fc_stack = []
        for count in range(4):
            self.fc_stack.append(
                (
                    self.add_weight(
                        shape=(last_dim, self.n_neurons), name=f"fc_kernel_{self.name}_{count}"
                    ),
                    self.add_weight(
                        shape=(self.n_neurons,),
                        initializer="zeros",
                        name=f"fc_bias_{self.name}_{count}",
                    ),
                )
            )
            last_dim = self.n_neurons

        self.dropout = Dropout(self.drop_rate)

        shape_fc_forecast = (self.quantiles, self.n_neurons, self._output_last_dim_forecast)
        if self.quantiles == 1:
            shape_fc_forecast = (self.n_neurons, self._output_last_dim_forecast)

        self.fc_forecast = self.add_weight(
            shape=shape_fc_forecast, name="fc_forecast_{self.name}"
        )

        shape_fc_backcast = (self.n_neurons, self._output_last_dim_backcast)
        self.fc_backcast = self.add_weight(
            shape=shape_fc_backcast, name="fc_backcast_{self.name}"
        )

        self.forecast_coef = self.add_weight(
            shape=self.forecast_coef.shape,
            initializer=tf.constant_initializer(self.forecast_coef.numpy()),
            trainable=False,
            name="gf_constrained_{self.name}",
        )

        self.backcast_coef = self.add_weight(
            shape=self.backcast_coef.shape,
            initializer=tf.constant_initializer(self.backcast_coef.numpy()),
            trainable=False,
            name="gb_constrained_{self.name}",
        )

        self.built = True

    def call(self, inputs): # pylint: disable=arguments-differ

        for kernel, bias in self.fc_stack:
            # shape: (Batch_size, n_neurons)
            inputs = tf.nn.bias_add(tf.matmul(inputs, kernel), bias)
            inputs = tf.nn.relu(inputs)
            inputs = self.dropout(inputs, training=True)

        # shape: (Batch_size, p_degree)
        theta_backcast = tf.matmul(inputs, self.fc_backcast)

        # shape: (quantiles, Batch_size, p_degree)
        theta_forecast = tf.matmul(inputs, self.fc_forecast)

        # shape: (Batch_size, backcast)
        outputs_backcast = tf.matmul(theta_backcast, self.backcast_coef)

        # shape: (quantiles, Batch_size, forecast)
        outputs_forecast = tf.matmul(theta_forecast, self.forecast_coef)

        return outputs_forecast, outputs_backcast

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tf.compat.dimension_value(input_shape[-1]) is None:
            raise ValueError(
                "The innermost dimension of input_shape must be defined, but saw: %s"
                % (input_shape,)
            )

        if self.quantiles == 1:
            return [
                tf.TensorShape((input_shape[0], int(self.horizon))),
                tf.TensorShape((input_shape[0], int(self.back_horizon))),
            ]

        return [
            tf.TensorShape((self.quantiles, input_shape[0], int(self.horizon))),
            tf.TensorShape((input_shape[0], int(self.back_horizon))),
        ]

    def coefficient_factory(self, *args, **kwargs):
        raise NotImplementedError('When subclassing the `BaseBlock` class, you should '
                              'implement a `coefficient_factory` method.')


class TrendBlock(BaseBlock):
    """
    Trend block definition.

    This layer represents the smaller part of nbeats model.
    Final layers are constrained which define a polynomial function of small degree p.
    Therefore it is possible to get explanation from this block.

    Parameters
    ----------
    horizon : integer
        Horizon time to forecast.
    back_horizon : integer
        Past to rebuild. Usually, back_horizon is 1 to 7 times longer than horizon.
    p_degree : integer
        Degree of the polynomial function. It needs to be > 0
    n_neurons : integer
        Number of neurons in Fully connected layers. It needs to be > 0.
    quantiles : integer, default to 1.
        Number of quantiles used in the QuantileLoss function. It needs to be > 1.
        If quantiles is 1 then the ouput will have a shape of (batch_size, horizon).
        else, the ouput will have a shape of (quantiles, batch_size, horizon).
    drop_rate : float, default to 0.1.
        Rate of the dropout layer. This is used to estimate the epistemic error.
        Expected a value between 0 and 1.

    Attributes
    ----------
    p_degree : integer
    horizon : float
    back_horizon : float
    n_neurons : integer
    quantiles : integer
    drop_rate : float

    Notes
    -----
    This class has been customized to integrate additional functionalities.
    Thanks to :class:`autopycoin.loss.QuantileLossError` it is therefore possible
    to indicate the number of quantiles the output will contain.
    These quantiles are the estimations of the aleatoric error a.k.a prediction interval.
    `drop_rate` parameter is used to estimate the epistemic error a.k.a confidence interval.

    input shape:
    N-D tensor with shape: (batch_size, ..., input_dim).
    The most common situation would be a 2D input with shape (batch_size, input_dim).

    output shape:
    N-D tensor with shape: (quantiles, batch_size, ..., units).
    For instance, for a 2D input with shape (batch_size, input_dim) and a quantile parameter to 1,
    the output would have shape (batch_size, units).
    With a quantile to 2 or higher the output would have shape (quantiles, batch_size, units).
    """

    def __init__(
        self,
        horizon,
        back_horizon,
        p_degree,
        n_neurons,
        quantiles=1,
        drop_rate=0.1,
        **kwargs,
    ):

        super().__init__(horizon,
                        back_horizon,
                        p_degree,
                        p_degree,
                        n_neurons,
                        quantiles,
                        drop_rate,**kwargs)

        # Shape (-1, 1) in order to broadcast horizon to all p degrees
        self.p_degree = p_degree
        self._p_degree = tf.expand_dims(tf.range(p_degree, dtype="float32"), axis=-1)
        self.forecast_coef = self.coefficient_factory(self.horizon)
        self.backcast_coef = self.coefficient_factory(self.back_horizon)

        if self.p_degree < 0:
            raise ValueError(
                f"Received an invalid value for `p_degree`, expected "
                f"a positive integer, got {p_degree}."
            )

    def coefficient_factory(self, horizon):
        """
        Compute the coefficients used in the last layer a.k.a g constrained layer.

        Parameters
        ----------
        horizon : int
        periods : lis[int]
        fourier_orders : list[int]

        Returns
        -------
        coefficients : tensor with shape ()
            Coefficients of the g layer.
        """

        coefficients = (
            tf.range(horizon) / horizon
        ) ** self._p_degree

        return coefficients

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "p_degree": self.p_degree,
                "horizon": self.horizon,
                "back_horizon": self.back_horizon,
                "n_neurons": self.n_neurons,
                "quantiles": self.quantiles,
                "drop_rate": self.drop_rate,
            }
        )
        return config


class SeasonalityBlock(BaseBlock):
    """
    Seasonality block definition.

    This layer represents the smaller part of nbeats model.
    Output layers are constrained which define fourier series.
    Each expansion coefficent then become a coefficient of the fourier serie.
    As each block and each
    stack outputs are sum up, we decided to introduce fourier order and
    multiple seasonality periods.
    It is possible to get explanation from this block.

    Parameters
    ----------
    p_degree : integer
        Degree of the polynomial function.
    horizon : integer
        Horizon time to forecast.
    back_horizon : integer
        Past to rebuild.
    nb_neurons : integer
        Number of neurons in Fully connected layers.
    periods : list[int]
        Compute the fourier serie period in the forecasting equation.
        If it's a list all periods are taken into account in the calculation.
    back_periods : list[int]
        Compute the fourier serie period in the backcasting equation.
        If it's a list all periods are taken into account in the calculation.
    forecast_fourier_order : list[int]
        Compute the fourier order. each order element is linked the respective period. 
    backcast_fourier_order : list[int]
        Compute the fourier order. each order element is linked the respective back period. 
    quantiles : integer, default to 1.
        Number of quantiles used in the QuantileLoss function. It needs to be > 1.
        If quantiles is 1 then the ouput will have a shape of (batch_size, horizon).
        else, the ouput will have a shape of (quantiles, batch_size, horizon).
    drop_rate : float, default to 0.1.
        Rate of the dropout layer. This is used to estimate the epistemic error.
        Expected a value between 0 and 1.

    Attributes
    ----------
    horizon : float
    back_horizon : float
    periods : list[int]
    back_periods : list[int]
    forecast_fourier_order : list[int]
    backcast_fourier_order : list[int]
    n_neurons : integer
    quantiles : integer
    drop_rate : float

    Notes
    -----
    This class has been customized to integrate additional functionalities.
    Thanks to :class:`autopycoin.loss.QuantileLossError` it is therefore possible
    to indicate the number of quantiles the output will contain.
    These quantiles are the estimations of the aleatoric error a.k.a prediction interval.
    `drop_rate` parameter is used to estimate the epistemic error a.k.a confidence interval.

    input shape:
    N-D tensor with shape: (batch_size, ..., input_dim).
    The most common situation would be a 2D input with shape (batch_size, input_dim).

    output shape:
    N-D tensor with shape: (quantiles, batch_size, ..., units).
    For instance, for a 2D input with shape (batch_size, input_dim) and a quantile parameter to 1,
    the output would have shape (batch_size, units).
    With a quantile to 2 or higher the output would have shape (quantiles, batch_size, units).
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
        **kwargs,
    ):

        # Workout the number of neurons needed to compute seasonality
        # coefficients
        forecast_neurons = tf.reduce_sum(2 * periods)
        backcast_neurons = tf.reduce_sum(2 * back_periods)

        super().__init__(horizon,
                        back_horizon,
                        forecast_neurons,
                        backcast_neurons,
                        n_neurons,
                        quantiles,
                        drop_rate,**kwargs)

        self.periods = periods
        self.back_periods = back_periods
        self.forecast_fourier_order = forecast_fourier_order
        self.backcast_fourier_order = backcast_fourier_order
        self.forecast_coef = self.coefficient_factory(self.horizon, self.periods, self.forecast_fourier_order)
        self.backcast_coef = self.coefficient_factory(self.back_horizon, self.back_periods, self.backcast_fourier_order)

        if len(periods) != len(forecast_fourier_order):
            raise ValueError(
                f"`periods` and `forecast_fourier_order` are expected"
                f"to have the same length, got"
                f"{len(periods)} and {len(forecast_fourier_order)} respectively."
            )

        if len(back_periods) != len(backcast_fourier_order):
            raise ValueError(
                f"`back_periods` and `backcast_fourier_order` are expected"
                f"to have the same length, got {len(back_periods)} and {len(backcast_fourier_order)} respectively."
            )
    
    def coefficient_factory(self, horizon, periods, fourier_orders):
        """
        Compute the coefficients used in the last layer a.k.a g constrained layer.

        Parameters
        ----------
        horizon : int
        periods : lis[int]
        fourier_orders : list[int]

        Returns
        -------
        coefficients : tensor with shape ()
            Coefficients of the g layer.
        """

        periods = tf.cast(tf.reshape(periods, shape=(-1, 1)), dtype="float32")
        time_forecast = tf.range(horizon, dtype="float32")

        coefficients = []
        for fourier_order, period in zip(fourier_orders, periods):
            time_forecast = 2 * np.pi * time_forecast / period
            seasonality = time_forecast * tf.expand_dims(tf.range(fourier_order, dtype='float32'), axis=-1)
            # Workout cos and sin seasonality coefficents
            seasonality = tf.concat(
                (tf.cos(seasonality), tf.sin(seasonality)), axis=0
            )

            coefficients.append(seasonality)
        coefficients = tf.concat(coefficients, axis=0)

        return coefficients

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "horizon" : self.horizon,
                "back_horizon" : self.back_horizon,
                "n_neurons" : self.n_neurons,
                "periods" : self.periods,
                "back_periods" : self.back_periods,
                "forecast_fourier_order" : self.forecast_fourier_order,
                "backcast_fourier_order" : self.backcast_fourier_order,
                "quantiles" : self.quantiles,
                "drop_rate" : self.drop_rate
            }
        )
        return config


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

        self._fc_stack = [Dense(n_neurons, activation="relu") for _ in range(4)]

        self._dropout = Dropout(drop_rate)

        shape_fc_backcast = (n_neurons, n_neurons)
        self._fc_backcast = self.add_weight(
            shape=shape_fc_backcast, name="fc_backcast_generic"
        )

        shape_fc_forecast = (quantiles, n_neurons, n_neurons)
        self._fc_forecast = self.add_weight(
            shape=shape_fc_forecast, name="fc_forecast_generic"
        )

        self._backcast = Dense(back_horizon)

        self._forecast = Dense(horizon)

    def call(self, inputs):
        # shape: (Batch_size, back_horizon)
        X = inputs
        for dense_layer in self._fc_stack:
            # shape: (Batch_size, nb_neurons)
            X = dense_layer(X)
            X = self._dropout(X, training=True)

        # shape: (quantiles, Batch_size, 2 * fourier order)
        theta_forecast = X @ self._fc_forecast

        # shape: (Batch_size, 2 * fourier order)
        theta_backcast = X @ self._fc_backcast

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

        y_forecast = tf.constant([0.0])
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
        self._residuals_y = tf.TensorArray(tf.float32, size=len(self._stacks))
        y_forecast = tf.constant([0.0])

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
    **kwargs,
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
            **kwargs,
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
            **kwargs,
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
                **kwargs,
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
                **kwargs,
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
    **kwargs,
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
                **kwargs,
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
                    **kwargs,
                )
                for _ in range(n_blocks)
            ]

            generic_stacks.append(Stack(generic_blocks))

    return NBEATS(generic_stacks)
