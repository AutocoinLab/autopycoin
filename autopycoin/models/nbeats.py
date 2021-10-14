"""
N-BEATS implementation
"""

from typing import Union, Tuple, List
import numpy as np

import tensorflow as tf
from autopycoin.losses.losses import QuantileLossError
from tensorflow.keras.layers import Dropout, InputSpec, Layer
from tensorflow.keras import Model
from tensorflow.keras.backend import floatx


class BaseBlock(Layer):
    """
    Base class for a nbeats block.

    Your custom block need to inherit from it.

    Parameters
    ----------
    horizon : int
        Horizon time to forecast.
    back_horizon : int
        Past to rebuild. Usually, back_horizon is 1 to 7 times longer than horizon.
    output_last_dim_forecast : int
        First dimension of the last layer.
        It is equal to p_degree in case of `TrendBlock`.
    output_last_dim_backcast : int
        First dimension of the last layer.
        It is equal to p_degree in case of `TrendBlock`.
    n_neurons : int
        Number of neurons in Fully connected layers. It needs to be > 0.
    drop_rate : float
        Rate of the dropout layer. This is used to estimate the epistemic error.
        Expected a value between 0 and 1. Default to 0.

    Attributes
    ----------
    p_degree : int
    horizon : float
    back_horizon : float
    n_neurons : int
    drop_rate : float
    """

    def __init__(
        self,
        horizon : int,
        back_horizon : int,
        output_last_dim_forecast : int,
        output_last_dim_backcast : int,
        n_neurons : int,
        drop_rate : float,
        **kwargs : dict,
    ):

        super().__init__(**kwargs)

        self.horizon = float(horizon)
        self.back_horizon = float(back_horizon)
        self.n_neurons = int(n_neurons)
        self.drop_rate = float(drop_rate)
        self._output_last_dim_forecast = output_last_dim_forecast
        self._output_last_dim_backcast = output_last_dim_backcast

        # Some checks
        if self.horizon < 0 or self.back_horizon < 0:
            raise ValueError(
                f"`horizon` and `back_horizon` parameter expected "
                f"a positive integer, got {self.horizon} and {self.back_horizon}."
            )

        if 0 > self.drop_rate > 1:
            raise ValueError(
                f"Received an invalid value for `drop_rate`, expected "
                f"a float between 0 and 1, got {self.drop_rate}."
            )
        if self.n_neurons < 0:
            raise ValueError(
                f"Received an invalid value for `n_neurons`, expected "
                f"a positive integer, got {self.n_neurons}."
            )

    def build(self, input_shape : tf.TensorShape):
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
                f"The last dimension of the inputs"
                f" should be defined. Found {last_dim}."
            )

        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})

        self.fc_stack = []
        for count in range(4):
            self.fc_stack.append(
                (
                    self.add_weight(
                        shape=(last_dim, self.n_neurons),
                        name=f"fc_kernel_{self.name}_{count}",
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

        # If the model is compiling with a loss error defining uncertainty then
        # broadcast the output to take into account this uncertainty
        if hasattr(self, 'quantiles'):
            shape_fc_forecast = (
                self.quantiles,
                self.n_neurons,
                self._output_last_dim_forecast,
            )
        else:
            shape_fc_forecast = (self.n_neurons, self._output_last_dim_forecast)

        self.fc_forecast = self.add_weight(
            shape=shape_fc_forecast, name="fc_forecast_{self.name}"
        )

        shape_fc_backcast = (self.n_neurons, self._output_last_dim_backcast)
        self.fc_backcast = self.add_weight(
            shape=shape_fc_backcast, name="fc_backcast_{self.name}"
        )

        # Set weights with calculated coef
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

    def call(self, inputs : tf.Tensor) -> Tuple[tf.Tensor]:  # pylint: disable=arguments-differ
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

    def compute_output_shape(self, input_shape : tf.TensorShape) -> List[tf.TensorShape]:
        if isinstance(input_shape, tuple):
            input_shape = input_shape[0]
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tf.compat.dimension_value(input_shape[-1]) is None:
            raise ValueError(
                "The innermost dimension of input_shape must be defined, but saw: %s"
                % (input_shape,)
            )

        # If the model is compiling with a loss error defining uncertainty then
        # broadcast the output to take into account this uncertainty
        if hasattr(self, 'quantiles'):
            return[
            tf.TensorShape((self.quantiles, input_shape[0], int(self.horizon))),
            tf.TensorShape((input_shape[0], int(self.back_horizon))),
        ]

        return [
            tf.TensorShape((input_shape[0], int(self.horizon))),
            tf.TensorShape((input_shape[0], int(self.back_horizon))),
        ]

    def coefficient_factory(self, *args, **kwargs):
        """
        Compute the coefficients used in the last layer a.k.a g constrained layer.
        This method needs to be overriden.
        """
        raise NotImplementedError(
            "When subclassing the `BaseBlock` class, you should "
            "implement a `coefficient_factory` method."
        )


class TrendBlock(BaseBlock):
    """
    Trend block definition.

    This layer represents the smaller part of nbeats model.
    Final layers are constrained which define a polynomial function of small degree p.
    Therefore it is possible to get explanation from this block.

    Parameters
    ----------
    horizon : int
        Horizon time to forecast.
    back_horizon : int
        Past to rebuild. Usually, back_horizon is 1 to 7 times longer than horizon.
    p_degree : int
        Degree of the polynomial function. It needs to be > 0.
    n_neurons : int
        Number of neurons in Fully connected layers. It needs to be > 0.
    drop_rate : float
        Rate of the dropout layer. This is used to estimate the epistemic error.
        Expected a value between 0 and 1. Default to 0.

    Attributes
    ----------
    p_degree : int
    horizon : float
    back_horizon : float
    n_neurons : int
    drop_rate : float
    input_spec : `InputSpec`

    Examples
    --------
    >>> from autopycoin.models import TrendBlock, SeasonalityBlock, Stack, NBEATS
    >>> from autopycoin.losses import QuantileLossError
    >>> trend_block = TrendBlock(horizon=10,
    ...                          back_horizon=20,
    ...                          p_degree=2,
    ...                          n_neurons=16,
    ...                          drop_rate=0.1,
    ...                          name="trend_block")
    >>>
    >>> seasonality_block = SeasonalityBlock(horizon=10,
    ...                                      back_horizon=20,
    ...                                      periods=[10],
    ...                                      back_periods=[20],
    ...                                      forecast_fourier_order=[10],
    ...                                      backcast_fourier_order=[20],
    ...                                      n_neurons=15,
    ...                                      drop_rate=0.1,
    ...                                      name="seasonality_block")
    >>>
    >>> trend_blocks = [trend_block for _ in range(3)]
    >>> seasonality_blocks = [seasonality_block for _ in range(3)]
    >>> trend_stacks = Stack(trend_blocks, name="trend_stack")
    >>> seasonality_stacks = Stack(seasonality_blocks, name="seasonality_stack")
    >>>
    >>> model = NBEATS([trend_stacks, seasonality_stacks], name="interpretable_NBEATS")
    >>> model.compile(loss=QuantileLossError(quantiles=[0.5]))

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
    N-D tensor with shape: (quantiles, batch_size, ..., units) or (batch_size, ..., units) .
    For instance, for a 2D input with shape (batch_size, input_dim),
    the output would have shape (batch_size, units).
    With a QuantileLossError with 2 quantiles or higher the output would have shape (quantiles, batch_size, units).
    """

    def __init__(
        self,
        horizon : int,
        back_horizon : int,
        p_degree : int,
        n_neurons : int,
        drop_rate : float =0,
        **kwargs,
    ):

        super().__init__(
            horizon,
            back_horizon,
            p_degree,
            p_degree,
            n_neurons,
            drop_rate,
            **kwargs,
        )

        # Shape (-1, 1) in order to broadcast horizon to all p degrees
        self.p_degree = p_degree
        self._p_degree = tf.expand_dims(
            tf.range(self.p_degree, dtype=floatx()), axis=-1
        )

        # Get coef
        self.forecast_coef = self.coefficient_factory(self.horizon, self._p_degree)
        self.backcast_coef = self.coefficient_factory(self.back_horizon, self._p_degree)

        # Some checks
        if self.p_degree < 0:
            raise ValueError(
                f"Received an invalid value for `p_degree`, expected "
                f"a positive integer, got {p_degree}."
            )

    def coefficient_factory(self, horizon : int, p_degree : int) -> tf.Tensor:
        """
        Compute the coefficients used in the last layer a.k.a g constrained layer.

        Parameters
        ----------
        horizon : int
        periods : list[int]
        fourier_orders : list[int]

        Returns
        -------
        coefficients : `tensor with shape (p_degree, horizon)`
            Coefficients of the g layer.
        """

        coefficients = (tf.range(horizon) / horizon) ** p_degree

        return coefficients

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "p_degree": self.p_degree,
                "horizon": self.horizon,
                "back_horizon": self.back_horizon,
                "n_neurons": self.n_neurons,
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
    horizon : int
        Horizon time to forecast.
    back_horizon : int
        Past to rebuild.
    n_neurons : int
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
    drop_rate : float
        Rate of the dropout layer. This is used to estimate the epistemic error.
        Expected a value between 0 and 1. Default to 0.

    Attributes
    ----------
    horizon : float
    back_horizon : float
    periods : list[int]
    back_periods : list[int]
    forecast_fourier_order : list[int]
    backcast_fourier_order : list[int]
    n_neurons : int
    drop_rate : float
    input_spec : `InputSpec`

    Examples
    --------
    >>> from autopycoin.models import TrendBlock, SeasonalityBlock, Stack, NBEATS
    >>> from autopycoin.losses import QuantileLossError
    >>> trend_block = TrendBlock(horizon=10,
    ...                          back_horizon=20,
    ...                          p_degree=2,
    ...                          n_neurons=16,
    ...                          drop_rate=0.1,
    ...                          name="trend_block")
    >>> seasonality_block = SeasonalityBlock(horizon=10,
    ...                                      back_horizon=20,
    ...                                      periods=[10],
    ...                                      back_periods=[20],
    ...                                      forecast_fourier_order=[10],
    ...                                      backcast_fourier_order=[20],
    ...                                      n_neurons=15,
    ...                                      drop_rate=0.1,
    ...                                      name="seasonality_block")
    >>> trend_blocks = [trend_block for _ in range(3)]
    >>> seasonality_blocks = [seasonality_block for _ in range(3)]
    >>> trend_stacks = Stack(trend_blocks, name="trend_stack")
    >>> seasonality_stacks = Stack(seasonality_blocks, name="seasonality_stack")
    >>> # model definition and compiling
    >>> model = NBEATS([trend_stacks, seasonality_stacks], name="interpretable_NBEATS")
    >>> model.compile(loss=QuantileLossError(quantiles=[0.5]))

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
    N-D tensor with shape: (quantiles, batch_size, ..., units) or (batch_size, ..., units) .
    For instance, for a 2D input with shape (batch_size, input_dim),
    the output would have shape (batch_size, units).
    With a QuantileLossError with 2 quantiles or higher the output would have shape (quantiles, batch_size, units).
    """

    def __init__(
        self,
        horizon : int,
        back_horizon : int,
        periods : List[int],
        back_periods : List[int],
        forecast_fourier_order : List[int],
        backcast_fourier_order : List[int],
        n_neurons : int,
        drop_rate : float =0,
        **kwargs,
    ):

        # Workout the number of neurons needed to compute seasonality
        # coefficients
        forecast_neurons = tf.reduce_sum(2 * forecast_fourier_order)
        backcast_neurons = tf.reduce_sum(2 * backcast_fourier_order)

        super().__init__(
            horizon,
            back_horizon,
            forecast_neurons,
            backcast_neurons,
            n_neurons,
            drop_rate,
            **kwargs,
        )

        self.periods = periods
        self.back_periods = back_periods
        self.forecast_fourier_order = forecast_fourier_order
        self.backcast_fourier_order = backcast_fourier_order

        # Get coef
        self.forecast_coef = self.coefficient_factory(
            self.horizon, self.periods, self.forecast_fourier_order
        )
        self.backcast_coef = self.coefficient_factory(
            self.back_horizon, self.back_periods, self.backcast_fourier_order
        )

        # Some checks
        if len(self.periods) != len(self.forecast_fourier_order):
            raise ValueError(
                f"`periods` and `forecast_fourier_order` are expected"
                f"to have the same length, got"
                f"{len(self.periods)} and {len(self.forecast_fourier_order)} respectively."
            )

        if len(self.back_periods) != len(self.backcast_fourier_order):
            raise ValueError(
                f"`back_periods` and `backcast_fourier_order` are expected"
                f"to have the same length, got {len(self.back_periods)}"
                f"and {len(self.backcast_fourier_order)} respectively."
            )

    def coefficient_factory(self, horizon : int, periods : List[int], fourier_orders : List[int]) -> tf.Tensor:
        """
        Compute the coefficients used in the last layer a.k.a g constrained layer.

        Parameters
        ----------
        horizon : int
        periods : list[int]
        fourier_orders : list[int]

        Returns
        -------
        coefficients : `tensor with shape (periods * fourier_orders, horizon)`
            Coefficients of the g layer.
        """

        # Shape (-1, 1) in order to broadcast periods to all time units
        periods = tf.cast(tf.reshape(periods, shape=(-1, 1)), dtype=floatx())
        time_forecast = tf.range(horizon, dtype=floatx())

        coefficients = []
        for fourier_order, period in zip(fourier_orders, periods):
            time_forecast = 2 * np.pi * time_forecast / period
            seasonality = time_forecast * tf.expand_dims(
                tf.range(fourier_order, dtype=floatx()), axis=-1
            )

            # Workout cos and sin seasonality coefficents
            seasonality = tf.concat((tf.cos(seasonality), tf.sin(seasonality)), axis=0)
            coefficients.append(seasonality)

        coefficients = tf.concat(coefficients, axis=0)
        return coefficients

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "horizon": self.horizon,
                "back_horizon": self.back_horizon,
                "n_neurons": self.n_neurons,
                "periods": self.periods,
                "back_periods": self.back_periods,
                "forecast_fourier_order": self.forecast_fourier_order,
                "backcast_fourier_order": self.backcast_fourier_order,
                "drop_rate": self.drop_rate,
            }
        )
        return config


class GenericBlock(BaseBlock):
    """
    Generic block definition as described in the paper.

    This layer represents the smaller part of a nbeats model.
    We can't have explanation from this kind of block because g coefficients
    are learnt.

    Parameters
    ----------
    horizon : int
        Horizon time to forecast.
    back_horizon : int
        Past to rebuild.
    forecast_neurons : int
        First dimensionality of the last layer.
    backcast_neurons : int
        First dimensionality of the last layer.
    n_neurons : int
        Number of neurons in Fully connected layers.
    drop_rate : float
        Rate of the dropout layer. This is used to estimate the epistemic error.
        Expected a value between 0 and 1. Default to 0.1.

    Attributes
    ----------
    horizon : float
    back_horizon : float
    forecast_neurons : int
    backcast_neurons : int
    n_neurons : int
    drop_rate : float
    input_spec : `InputSpec`

    Examples
    --------
    >>> from autopycoin.models import GenericBlock, Stack, NBEATS
    >>> from autopycoin.losses import QuantileLossError
    >>> generic_block = GenericBlock(horizon=10,
    ...                          back_horizon=20,
    ...                          n_neurons=16,
    ...                          forecast_neurons=16,
    ...                          backcast_neurons=16,
    ...                          drop_rate=0.1,
    ...                          name="generic_block")
    >>> generic_blocks = [generic_block for _ in range(3)]
    >>> generic_stacks = Stack(generic_blocks, name="generic_stack")
    >>> # Model definition and compiling
    >>> model = NBEATS([generic_stacks, generic_stacks], name="generic_NBEATS")
    >>> model.compile(loss=QuantileLossError(quantiles=[0.5]))

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
    N-D tensor with shape: (quantiles, batch_size, ..., units) or (batch_size, ..., units) .
    For instance, for a 2D input with shape (batch_size, input_dim),
    the output would have shape (batch_size, units).
    With a QuantileLossError with 2 quantiles or higher the output would have shape (quantiles, batch_size, units).
    """

    def __init__(
        self,
        horizon : int,
        back_horizon : int,
        forecast_neurons : int,
        backcast_neurons : int,
        n_neurons : int,
        drop_rate : float = 0.1,
        **kwargs,
    ):

        super().__init__(
            horizon,
            back_horizon,
            forecast_neurons,
            backcast_neurons,
            n_neurons,
            drop_rate,
            **kwargs,
        )

        self.forecast_neurons = int(forecast_neurons)
        self.backcast_neurons = int(backcast_neurons)

        # Get coef
        self.forecast_coef = self.coefficient_factory(
            self.horizon, self.forecast_neurons
        )
        self.backcast_coef = self.coefficient_factory(
            self.back_horizon, self.backcast_neurons
        )

    def build(self, input_shape):

        super().build(input_shape)

        # Set weights with calculated coef
        self.forecast_coef = self.add_weight(
            shape=self.forecast_coef.shape,
            initializer=tf.constant_initializer(self.forecast_coef.numpy()),
            trainable=True,
            name="gf_{self.name}",
        )

        self.backcast_coef = self.add_weight(
            shape=self.backcast_coef.shape,
            initializer=tf.constant_initializer(self.backcast_coef.numpy()),
            trainable=True,
            name="gb_{self.name}",
        )

        self.built = True

    def coefficient_factory(self, horizon : Union[int,float], neurons : int) -> tf.Tensor:
        """
        Compute the coefficients used in the last layer a.k.a g layer.

        Parameters
        ----------
        horizon : int
        neurons : int

        Returns
        -------
        coefficients : `tensor with shape (horizon, neurons)`
            Coefficients of the g layer.
        """

        coefficients = tf.keras.initializers.GlorotUniform(seed=42)(
            shape=(neurons, int(horizon))
        )

        return coefficients

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "horizon": self.horizon,
                "back_horizon": self.back_horizon,
                "forecast_neurons": self.forecast_neurons,
                "backcast_neurons": self.backcast_neurons,
                "n_neurons": self.n_neurons,
                "drop_rate": self.drop_rate,
            }
        )
        return config


class Stack(Layer):
    """
    A stack is a series of blocks where each block produce two outputs,
    the forecast and the backcast.

    All forecasts are sum up and compose the stack output. In the meantime,
    each backcasts is given to the following block.

    Parameters
    ----------
    blocks : list[:class:`autopycoin.models.BaseBlock`]
        Blocks layers. they can be generic, seasonal or trend ones.
        You can also define your own block by subclassing `BaseBlock`.

    Attributes
    ----------
    blocks : list[:class:`autopycoin.models.BaseBlock`]

    Examples
    --------
    >>> from autopycoin.models import TrendBlock, SeasonalityBlock, Stack, NBEATS
    >>> from autopycoin.losses import QuantileLossError
    >>> trend_block = TrendBlock(horizon=10,
    ...                          back_horizon=20,
    ...                          p_degree=2,
    ...                          n_neurons=16,
    ...                          drop_rate=0.1,
    ...                          name="trend_block")
    >>> seasonality_block = SeasonalityBlock(horizon=10,
    ...                                      back_horizon=20,
    ...                                      periods=[10],
    ...                                      back_periods=[20],
    ...                                      forecast_fourier_order=[10],
    ...                                      backcast_fourier_order=[20],
    ...                                      n_neurons=15,
    ...                                      drop_rate=0.1,
    ...                                      name="seasonality_block")
    >>> trend_blocks = [trend_block for _ in range(3)]
    >>> seasonality_blocks = [seasonality_block for _ in range(3)]
    >>> trend_stacks = Stack(trend_blocks, name="trend_stack")
    >>> seasonality_stacks = Stack(seasonality_blocks, name="seasonality_stack")
    >>> # model definition and compiling
    >>> model = NBEATS([trend_stacks, seasonality_stacks], name="interpretable_NBEATS")
    >>> model.compile(loss=QuantileLossError(quantiles=[0.5]))

    Notes
    -----
    input shape:
    N-D tensor with shape: (batch_size, ..., input_dim).
    The most common situation would be a 2D input with shape (batch_size, input_dim).

    output shape:
    N-D tensor with shape: (quantiles, batch_size, ..., units) or (batch_size, ..., units) .
    For instance, for a 2D input with shape (batch_size, input_dim),
    the output would have shape (batch_size, units).
    With a QuantileLossError with 2 quantiles or higher the output would have shape (quantiles, batch_size, units).
    """

    def __init__(self, blocks : List[BaseBlock], **kwargs):

        super().__init__(**kwargs)

        self.blocks = blocks

        for block in self.blocks:
            if isinstance(block, type(BaseBlock)):
                raise ValueError("`blocks` is expected to inherit from `BaseBlock`")

    def call(self, inputs : tf.Tensor) -> Tuple[tf.Tensor]:

        outputs_forecast = tf.constant(0.0)
        for block in self.blocks:
            # shape:
            # (quantiles, Batch_size, forecast)
            # (Batch_size, backcast)
            outputs_residual, outputs_backcast = block(inputs)
            inputs = tf.subtract(inputs, outputs_backcast)
            # shape: (quantiles, Batch_size, forecast)
            outputs_forecast = tf.add(outputs_forecast, outputs_residual)

        return outputs_forecast, inputs

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"blocks": self.blocks})
        return config


class NBEATS(Model):
    """
    Tensorflow model defining the N-BEATS architecture.

    N-BEATS is a univariate model. Its strong advantage
    resides in its structure which allows us to extract the trend and the seasonality of
    temporal series. They are available from the attributes `seasonality` and `trend`.
    This is an unofficial implementation of the paper https://arxiv.org/abs/1905.10437.

    Parameters
    ----------
    stacks : list[:class:`autopycoin.models.Stack`]
             Stacks can be created from :class:`autopycoin.models.TrendBlock`,
             :class:`autopycoin.models.SeasonalityBlock` or :class:`autopycoin.models.GenericBlock`.
             See stack documentation for more details.

    Attributes
    ----------
    stacks : list[`tensor`]
    seasonality : `tensor`
        Seasonality component of the output.
    trend : `tensor`
        Trend component of the output.

    Examples
    --------
    >>> from autopycoin.models import TrendBlock, SeasonalityBlock, Stack, NBEATS
    >>> from autopycoin.losses import QuantileLossError
    >>> trend_block = TrendBlock(horizon=10,
    ...                          back_horizon=20,
    ...                          p_degree=2,
    ...                          n_neurons=16,
    ...                          drop_rate=0.1,
    ...                          name="trend_block")
    >>>
    >>> seasonality_block = SeasonalityBlock(horizon=10,
    ...                                      back_horizon=20,
    ...                                      periods=[10],
    ...                                      back_periods=[20],
    ...                                      forecast_fourier_order=[10],
    ...                                      backcast_fourier_order=[20],
    ...                                      n_neurons=15,
    ...                                      drop_rate=0.1,
    ...                                      name="seasonality_block")
    >>>
    >>> trend_blocks = [trend_block for _ in range(3)]
    >>> seasonality_blocks = [seasonality_block for _ in range(3)]
    >>> trend_stacks = Stack(trend_blocks, name="trend_stack")
    >>> seasonality_stacks = Stack(seasonality_blocks, name="seasonality_stack")
    >>>
    >>> model = NBEATS([trend_stacks, seasonality_stacks], name="interpretable_NBEATS")
    >>> model.compile(loss=QuantileLossError(quantiles=[0.5]))

    Notes
    -----
    NBEATS supports the estimation of aleotoric and epistemic errors with:
    
    - Aleotoric interval : :class:`autopycoin.loss.QuantileLossError`
    - Epistemic interval : MCDropout

    You can use :class:`autopycoin.loss.QuantileLossError` as loss error to estimate the
    aleotoric error. Also, run multiple time a prediction with `drop_date` > 0 to estimate
    the epistemic error.

    *Input shape*:
    N-D tensor with shape: (batch_size, ..., input_dim).
    The most common situation would be a 2D input with shape (batch_size, input_dim).

    *Output shape*:
    N-D tensor with shape: (quantiles, batch_size, ..., units) or (batch_size, ..., units) .
    For instance, for a 2D input with shape (batch_size, input_dim),
    the output would have shape (batch_size, units).
    With a QuantileLossError with 2 quantiles or higher the output would have shape (quantiles, batch_size, units).
    """

    def __init__(self, stacks : List[Stack], **kwargs):

        super().__init__(self, **kwargs)

        # Stacks where blocks are defined
        self.stacks = stacks

        # check if stack are inherit from Stack
        for stack in self.stacks:
            if not isinstance(stack, Stack):
                raise ValueError("`stacks` is expected to inherit from `Stack`")

    def compile(self,
                optimizer='rmsprop',
                loss=None,
                metrics=None,
                loss_weights=None,
                weighted_metrics=None,
                run_eagerly=None,
                steps_per_execution=None,
                **kwargs):

        if isinstance(loss, QuantileLossError):
            # will be changed.
            # need a generator to go through all blocks
            for stack in self.stacks:
                for block in stack.blocks:
                    block.built = False
                    block.quantiles = len(loss.quantiles)

        return super().compile(optimizer=optimizer,
                                loss=loss,
                                metrics=metrics,
                                loss_weights=loss_weights,
                                weighted_metrics=weighted_metrics,
                                run_eagerly=run_eagerly,
                                steps_per_execution=steps_per_execution,
                                **kwargs)

    def call(self, inputs):

        if isinstance(inputs, tuple):
            inputs = inputs[0]

        # Stock trend and seasonality curves during inference
        self._outputs_residual = tf.TensorArray(tf.float32, size=len(self.stacks))
        outputs_forecast = tf.constant(0.0)

        for idx, stack in enumerate(self.stacks):
            # shape:
            # (quantiles, Batch_size, forecast)
            # (Batch_size, backcast)
            outputs_residual, inputs = stack(inputs)
            self._outputs_residual.write(idx, outputs_residual)

            # shape: (quantiles, Batch_size, forecast)
            outputs_forecast = tf.math.add(outputs_forecast, outputs_residual)

        self._outputs_residual = self._outputs_residual.stack()
        return outputs_forecast

    @property
    def seasonality(self) -> tf.Tensor:
        """The seasonality component of the output."""
        if not isinstance(self.stacks[1].blocks[0], SeasonalityBlock):
            raise AttributeError(
                f"Only seasonality block defines a seasonality, got {self.stacks[1].blocks[0]}"
            )
        return self._outputs_residual[1:]

    @property
    def trend(self) -> tf.Tensor:
        """The trend component of the output."""
        if not isinstance(self.stacks[0].blocks[0], TrendBlock):
            raise AttributeError(
                f"Only trend block defines a trend, got {self.stacks[0].blocks[0]}"
            )
        return self._outputs_residual[:1]

    def get_config(self) -> dict:
        return {"stacks": self.stacks}


def create_interpretable_nbeats(
    horizon,
    back_horizon,
    periods,
    back_periods,
    forecast_fourier_order,
    backcast_fourier_order,
    p_degree=1,
    trend_n_neurons=16,
    seasonality_n_neurons=16,
    drop_rate=0,
    share=True,
    **kwargs,
):
    """
    Wrapper to create interpretable model using recommandations of the paper authors.
    Two stacks are created with 3 blocks each. The fist entirely composed by trend blocks,
    The second entirely composed by seasonality blocks.

    In the same stack, it is possible to share the weights between blocks.

    Parameters
    ----------
    horizon : int
        Horizon time to forecast.
    back_horizon : int
        Past to rebuild. Usually, back_horizon = n * horizon with n between 1 and 7.
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
    p_degree : int
        Degree of the polynomial function. It needs to be > 0.
    trend_n_neurons : int
        Number of neurons in th Fully connected trend layers.
    seasonality_n_neurons: int
        Number of neurons in Fully connected seasonality layers.
    drop_rate : float
        Rate of the dropout layer. This is used to estimate the epistemic error.
        Expected a value between 0 and 1. Default to 0.
    share : bool
        If True, the weights are shared between blocks inside a stack. Dafault to True.

    Returns
    -------
    model : :class:`autopycoin.models.NBEATS`
        Return an interpetable model with two stacks. One composed by 3 `TrendBlock`
        objects and a second composed by 3 `SeasonalityBlock` objects.

    Examples
    --------
    >>> from autopycoin.models import create_interpretable_nbeats
    >>> from autopycoin.losses import QuantileLossError
    >>> model = create_interpretable_nbeats(horizon=2,
    ...                                     back_horizon=3,
    ...                                     periods=[2],
    ...                                     back_periods=[3],
    ...                                     forecast_fourier_order=[2],
    ...                                     backcast_fourier_order=[3],
    ...                                     p_degree=1,
    ...                                     trend_n_neurons=16,
    ...                                     seasonality_n_neurons=16,
    ...                                     drop_rate=0.1,
    ...                                     share=True)
    >>> model.compile(loss=QuantileLossError(quantiles=[0.5]))
    """

    if share is True:
        trend_block = TrendBlock(
            horizon=horizon,
            back_horizon=back_horizon,
            p_degree=p_degree,
            n_neurons=trend_n_neurons,
            drop_rate=drop_rate,
            name="trend_block",
            **kwargs,
        )

        seasonality_block = SeasonalityBlock(
            horizon=horizon,
            back_horizon=back_horizon,
            periods=periods,
            back_periods=back_periods,
            forecast_fourier_order=forecast_fourier_order,
            backcast_fourier_order=backcast_fourier_order,
            n_neurons=seasonality_n_neurons,
            drop_rate=drop_rate,
            name="seasonality_block",
            **kwargs,
        )

        trend_blocks = [trend_block for _ in range(3)]
        seasonality_blocks = [seasonality_block for _ in range(3)]
    else:
        trend_blocks = [
            TrendBlock(
                horizon=horizon,
                back_horizon=back_horizon,
                p_degree=p_degree,
                n_neurons=trend_n_neurons,
                drop_rate=drop_rate,
                name="trend_block",
                **kwargs,
            )
            for _ in range(3)
        ]
        seasonality_blocks = [
            SeasonalityBlock(
                horizon=horizon,
                back_horizon=back_horizon,
                periods=periods,
                back_periods=back_periods,
                forecast_fourier_order=forecast_fourier_order,
                backcast_fourier_order=backcast_fourier_order,
                n_neurons=seasonality_n_neurons,
                drop_rate=drop_rate,
                name="seasonality_block",
                **kwargs,
            )
            for _ in range(3)
        ]

    trend_stacks = Stack(trend_blocks, name="trend_stack")
    seasonality_stacks = Stack(seasonality_blocks, name="seasonality_stack")
    model = NBEATS([trend_stacks, seasonality_stacks], name="interpretable_NBEATS")

    return model


def create_generic_nbeats(
    horizon,
    back_horizon,
    forecast_neurons,
    backcast_neurons,
    n_neurons,
    n_blocks,
    n_stacks,
    drop_rate=0,
    share=True,
    **kwargs,
):
    """
    Wrapper to create generic model.

    In the same stack, it is possible to share the weights between blocks.

    Parameters
    ----------
    horizon : int
        Horizon time to forecast.
    back_horizon : int
        Past to rebuild. Usually, back_horizon = n * horizon with n between 1 and 7.
    n_neurons : int
        Number of neurons in th Fully connected generic layers.
    n_blocks : int
        Number of blocks per stack.
    n_stacks : int
        Number of stacks in the model.
    drop_rate : float
        Rate of the dropout layer. This is used to estimate the epistemic error.
        Expected a value between 0 and 1. Default to 0.
    share : bool
        If True, the weights are shared between blocks inside a stack. Default to True.

    Returns
    -------
    model : :class:`autopycoin.models.NBEATS`
        Return an generic model with n stacks defined by the parameter `n_stack`
        and respoectively n blocks defined by `n_blocks`.

    Examples
    --------
    >>> from autopycoin.models import create_generic_nbeats
    >>> from autopycoin.losses import QuantileLossError
    >>> model = create_generic_nbeats(horizon=2,
    ...                               back_horizon=3,
    ...                               forecast_neurons=16,
    ...                               backcast_neurons=16,
    ...                               n_neurons=16,
    ...                               n_blocks=3,
    ...                               n_stacks=3,
    ...                               drop_rate=0.1,
    ...                               share=True)
    >>> model.compile(loss=QuantileLossError(quantiles=[0.5]))
    """

    generic_stacks = []
    if share is True:
        for _ in range(n_stacks):
            generic_block = GenericBlock(
                horizon=horizon,
                back_horizon=back_horizon,
                forecast_neurons=forecast_neurons,
                backcast_neurons=backcast_neurons,
                n_neurons=n_neurons,
                drop_rate=drop_rate,
                name="generic_block",
                **kwargs,
            )

            generic_blocks = [generic_block for _ in range(n_blocks)]
            generic_stacks.append(Stack(generic_blocks, name="generic_stack"))

    else:
        for _ in range(n_stacks):
            generic_blocks = [
                GenericBlock(
                    horizon=horizon,
                    back_horizon=back_horizon,
                    forecast_neurons=forecast_neurons,
                    backcast_neurons=backcast_neurons,
                    n_neurons=n_neurons,
                    drop_rate=drop_rate,
                    name="generic_block",
                    **kwargs,
                )
                for _ in range(n_blocks)
            ]

            generic_stacks.append(Stack(generic_blocks, name="generic_stack"))

    model = NBEATS(
        generic_stacks,
        name="generic_NBEATS",
    )
    return model
