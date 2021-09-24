"""
N-BEATS implementation
"""

import tensorflow as tf
from tensorflow.keras.layers import Add, Dense, Dropout, InputSpec, Layer, Subtract
from tensorflow.keras import Model
import numpy as np


class TrendBlock(Layer):
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

    Attributes
    ----------
    p_degree : Tensor of shape [p_degree, 1]
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
    These quantiles are estimations of prediction intervals.


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

        super().__init__(**kwargs)

        # Shape (-1, 1) in order to broadcast horizon to all p degrees
        self._p_degree = p_degree
        self.p_degree = tf.expand_dims(tf.range(p_degree, dtype="float"), axis=-1)

        self.horizon = float(horizon)
        self.back_horizon = float(back_horizon)
        self.n_neurons = int(n_neurons)
        self.quantiles = int(quantiles)
        self.drop_rate = float(drop_rate)

        if 0 > self.drop_rate > 1:
            raise ValueError(
                f"Received an invalid value for `drop_rate`, expected "
                f"a float betwwen 0 and 1, got {drop_rate}."
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
        if all(self.p_degree < 0):
            raise ValueError(
                f"Received an invalid value for `p_degree`, expected "
                f"a positive integer, got {p_degree}."
            )

    def build(self, input_shape):
        dtype = tf.as_dtype(self.dtype or tf.float32())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError(
                "Unable to build `TrendBlock` layer with "
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
                        shape=(last_dim, self.n_neurons), name=f"FC_kernel_{count}"
                    ),
                    self.add_weight(
                        shape=(self.n_neurons,),
                        initializer="zeros",
                        name=f"FC_bias_{count}",
                    ),
                )
            )
            last_dim = self.n_neurons

        self.dropout = Dropout(self.drop_rate)

        shape_fc_backcast = (self.n_neurons, len(self.p_degree))
        self.fc_backcast = self.add_weight(
            shape=shape_fc_backcast, name="fc_backcast_trend"
        )

        shape_fc_forecast = (self.quantiles, self.n_neurons, len(self.p_degree))
        if self.quantiles == 1:
            shape_fc_forecast = (self.n_neurons, len(self.p_degree))
        self.fc_forecast = self.add_weight(
            shape=shape_fc_forecast, name="fc_forecast_trend"
        )

        forecast_coef = (tf.range(self.horizon) / self.horizon) ** self.p_degree
        self.forecast_coef = self.add_weight(
            shape=forecast_coef.shape,
            initializer=tf.constant_initializer(forecast_coef.numpy()),
            trainable=False,
            name="gf_constained",
        )
        backcast_coef = (
            tf.range(self.back_horizon) / self.back_horizon
        ) ** self.p_degree
        self.backcast_coef = self.add_weight(
            shape=backcast_coef.shape,
            initializer=tf.constant_initializer(backcast_coef.numpy()),
            trainable=False,
            name="gb_constained",
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

    def compute_output_signature(self, input_signature):
        def check_type_return_shape(s):
            if not isinstance(s, tf.TensorSpec):
                raise TypeError(
                    "Only TensorSpec signature types are supported, "
                    "but saw signature entry: {}.".format(s)
                )
            return s.shape

        input_shape = tf.nest.map_structure(check_type_return_shape, input_signature)
        output_shape = self.compute_output_shape(input_shape)
        dtype = self._compute_dtype
        if dtype is None:
            input_dtypes = [s.dtype for s in tf.nest.flatten(input_signature)]
            # Default behavior when self.dtype is None, is to use the first input's
            # dtype.
            dtype = input_dtypes[0]
        return tf.nest.map_structure(
            lambda s: tf.TensorSpec(dtype=dtype, shape=s), output_shape
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "p_degree": self._p_degree,
                "horizon": self.horizon,
                "back_horizon": self.back_horizon,
                "n_neurons": self.n_neurons,
                "quantiles": self.quantiles,
                "drop_rate": self.drop_rate,
            }
        )
        return config


class SeasonalityBlock(Layer):
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
        **kwargs,
    ):

        super().__init__(**kwargs)

        shape = (1, -1, 1)
        periods = tf.cast(tf.reshape(periods, shape), "tf.float32")
        back_periods = tf.cast(tf.reshape(back_periods, shape), "tf.float32")

        shape = (-1, 1, 1)
        forecast_fourier_order = tf.reshape(
            range(forecast_fourier_order, dtype="tf.float32"), shape
        )
        backcast_fourier_order = tf.reshape(
            range(backcast_fourier_order, dtype="tf.float32"), shape
        )

        # Workout the number of neurons needed to compute seasonality
        # coefficients
        forecast_neurons = tf.reduce_sum(2 * periods)
        backcast_neurons = tf.reduce_sum(2 * back_periods)

        self.fc_stack = [Dense(n_neurons, activation="relu") for _ in range(4)]

        self.dropout = Dropout(drop_rate)

        shape_fc_backcast = (n_neurons, backcast_neurons)
        self.fc_backcast = self.add_weight(
            shape=shape_fc_backcast, name="fc_backcast_seasonality"
        )

        shape_fc_forecast = (quantiles, n_neurons, forecast_neurons)
        self.fc_forecast = self.add_weight(
            shape=shape_fc_forecast, name="fc_forecast_seasonality"
        )

        # Workout tf.cos and tf.sin seasonality coefficents
        time_forecast = range(horizon, dtype="tf.float32")
        time_forecast = 2 * np.pi * time_forecast / periods
        forecast_seasonality = time_forecast * forecast_fourier_order
        forecast_seasonality = tf.concat(
            (tf.cos(forecast_seasonality), tf.sin(forecast_seasonality)), axis=0
        )

        time_backcast = range(back_horizon, dtype="tf.float32")
        time_backcast = 2 * np.pi * time_backcast / back_periods
        backcast_seasonality = time_backcast * backcast_fourier_order
        backcast_seasonality = tf.concat(
            (tf.cos(backcast_seasonality), tf.sin(backcast_seasonality)), axis=0
        )

        shape_forecast_coef = (forecast_neurons, horizon)
        self.forecast_coef = tf.constant(
            forecast_seasonality,
            shape=shape_forecast_coef,
            dtype="tf.float32",
        )

        shape_backcast_coef = (backcast_neurons, back_horizon)
        self.backcast_coef = tf.constant(
            backcast_seasonality,
            shape=shape_backcast_coef,
            dtype="tf.float32",
        )

    def call(self, inputs):

        for dense in self.fc_stack:
            # shape: (Batch_size, n_neurons)
            x = dense(inputs)
            x = self.dropout(x, training=True)

        # shape: (Batch_size, 2 * fourier order)
        theta_backcast = x @ self.fc_backcast

        # shape: (quantiles, Batch_size, 2 * fourier order)
        theta_forecast = x @ self.fc_forecast

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
