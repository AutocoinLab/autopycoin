from typing import Tuple, List, Any
import numpy as np
import abc

import tensorflow as tf
from tensorflow.keras.layers import Dropout, InputSpec

# from tensorflow.keras import Model
from tensorflow.keras.backend import floatx

from .base_layer import Layer
from ..baseclass import AutopycoinBaseClass


class BaseBlock(Layer, AutopycoinBaseClass):
    """
    Base class of a nbeats block.

    Your custom block need to inherit from it.

    Parameters
    ----------
    input_width : int
        Horizon time to forecast.
    label_width : int
        Past to rebuild. Usually, label_width is 1 to 7 times longer than input_width.
    output_first_dim_forecast : int
        First dimension of the last layer.
        It is equal to p_degree in case of `TrendBlock`.
    output_first_dim_backcast : int
        First dimension of the last layer.
        It is equal to p_degree in case of `TrendBlock`.
    n_neurons : int
        Number of neurons in Fully connected layers.
    drop_rate : float
        Rate of the dropout layer. This is used to estimate the epistemic error.
        Expected a value between 0 and 1. Default to 0.
    name : str
        The name of the layer. It defines also the `block_type` attribute.

    Attributes
    ----------
    input_width : int
    label_width : int
    n_neurons : int
    drop_rate : float

    Raises
    ------
    ValueError:
        if `name` doesn't contain `Block`.
        `drop_rate` not between 0 and 1.
        All others arguments are not strictly positive integers.

    """

    def __init__(
        self,
        input_width: int,
        label_width: int,
        output_first_dim_forecast: int,
        output_first_dim_backcast: int,
        n_neurons: int,
        drop_rate: float,
        g_trainable: bool = False,
        interpretable: bool = False,
        block_type: str = "BaseBlock",
        **kwargs: dict,
    ):

        super().__init__(**kwargs)

        self._input_width = input_width
        self._label_width = label_width
        self._drop_rate = drop_rate
        self._n_neurons = n_neurons
        self._output_first_dim_forecast = output_first_dim_forecast
        self._output_first_dim_backcast = output_first_dim_backcast
        self._is_g_trainable = g_trainable
        self._is_interpretable = interpretable
        self._block_type = block_type

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build method from tensorflow."""
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

        # Computing fc layers
        self.fc_stack = []
        for count in range(4):
            self.fc_stack.append(
                (
                    self.add_weight(
                        shape=(last_dim, self._n_neurons),
                        name=f"fc_kernel_{self.name}_{count}",
                    ),
                    self.add_weight(
                        shape=(self._n_neurons,),
                        initializer="zeros",
                        name=f"fc_bias_{self.name}_{count}",
                    ),
                )
            )
            last_dim = self._n_neurons

        self.dropout = Dropout(self.drop_rate)

        # If the model is compiling with a loss error defining uncertainty then
        # broadcast the output to take into account this uncertainty.
        if self._n_quantiles:
            shape_fc_forecast = (
                self._n_quantiles,
                self._n_neurons,
                self._output_first_dim_forecast,
            )
        else:
            shape_fc_forecast = (self._n_neurons, self._output_first_dim_forecast)

        self.fc_forecast = self.add_weight(
            shape=shape_fc_forecast, name=f"fc_forecast_{self.name}"
        )

        shape_fc_backcast = (self._n_neurons, self._output_first_dim_backcast)
        self.fc_backcast = self.add_weight(
            shape=shape_fc_backcast, name=f"fc_backcast_{self.name}"
        )

        self.forecast_coef = self._get_forecast_coefficients()
        self.backcast_coef = self._get_backcast_coefficients()

        self.built = True

    def call(
        self, inputs: tf.Tensor
    ) -> Tuple[tf.Tensor]:  # pylint: disable=arguments-differ
        """Call method from tensorflow."""
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
        inputs_reconstruction = tf.matmul(theta_backcast, self.backcast_coef)

        # shape: (quantiles, Batch_size, forecast)
        outputs = tf.matmul(theta_forecast, self.forecast_coef)

        return outputs, inputs_reconstruction

    def compute_output_shape(
        self, input_shape: tf.TensorShape
    ) -> Tuple[tf.TensorShape]:
        """output method from tensoflow"""
        if isinstance(input_shape, tuple):
            input_shape = input_shape[0]
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tf.compat.dimension_value(input_shape[-1]) is None:
            raise ValueError(
                "The innermost dimension of input_shape must be defined, but saw: %s"
                % (input_shape,)
            )

        # If the model is compiled with a loss error defining uncertainty then
        # reshape the output to take into account this uncertainty.
        if self._n_quantiles:
            return [
                tf.TensorShape((self._n_quantiles, input_shape[0], self.input_width)),
                tf.TensorShape((input_shape[0], self.label_width)),
            ]

        return [
            tf.TensorShape((input_shape[0], self.input_width)),
            tf.TensorShape((input_shape[0], self.label_width)),
        ]

    @property
    def input_width(self):
        """Return the input_width."""
        return self._input_width

    @property
    def label_width(self):
        """Return the back input_width."""
        return self._label_width

    @property
    def drop_rate(self):
        """Return the drop rate."""
        return self._drop_rate

    @property
    def is_interpretable(self) -> bool:
        """Return True if the block is interpretable."""
        return self._is_interpretable

    @property
    def is_g_trainable(self) -> bool:
        """Return True if the last layer is trainable."""
        return self._is_g_trainable

    @property
    def block_type(self) -> str:
        """Return the block type. Default to `BaseBlock`."""
        return self._block_type

    @abc.abstractmethod
    def coefficient_factory(self, *args: list, **kwargs: dict) -> tf.Tensor:
        """
        Create the coefficients used in the last layer a.k.a g constrained layer.
        This method needs to be overriden.

        Returns
        -------
        coefficients
            `Tensor` of shape (d0, ..., output_first_dim, units)
        """
        raise NotImplementedError(
            "When subclassing the `BaseBlock` class, you should "
            "implement a `coefficient_factory` method."
        )

    @abc.abstractmethod
    def _get_forecast_coefficients(self) -> tf.Tensor:
        """
        Return the coefficients used in the forecast layer a.k.a g layer
        by calling coefficient_factory.
        This method needs to be overriden.

        Returns
        -------
        coefficients
            `Tensor` of shape (d0, ..., output_first_dim_forecast, input_width)

        Raises
        ------
        ValueError
            Raise an error if the coefficients tensor shape is not equal to
            (d0, ..., output_first_dim_forecast, input_width).
        """
        raise NotImplementedError(
            "When subclassing the `BaseBlock` class, you should "
            "implement a `_get_forecast_coefficients` method."
        )

    @abc.abstractmethod
    def _get_backcast_coefficients(self) -> tf.Tensor:
        """
        Return the coefficients used in the backcast layer a.k.a g layer
        by calling coefficient_factory.
        This method needs to be overriden.

        Returns
        -------
        coefficients
            `Tensor` of shape (d0, ..., output_first_dim_backcast, label_width)

        Raises
        ------
        ValueError
            Raise an error if the coefficients tensor shape is not equal to
            (d0, ..., output_first_dim_backcast, label_width).
        """
        raise NotImplementedError(
            "When subclassing the `BaseBlock` class, you should "
            "implement a `_get_backcast_coefficients` method."
        )

    def _val___init__(
        self, output: None, *args: list, **kwargs: dict
    ) -> None:  # pylint: disable=unused-argument
        if self.input_width <= 0 or self.label_width <= 0:
            raise ValueError(
                f"Received an invalid values for `input_width` or `label_width`, expected "
                f"strictly positive integers, got {self.input_width} and {self.label_width}."
            )

        if not (self.drop_rate >= 0.0 and self.drop_rate <= 1.0):
            raise ValueError(
                f"Received an invalid value for `drop_rate`, expected "
                f"a float between 0 and 1, got {self.drop_rate}."
            )
        if self._n_neurons <= 0:
            raise ValueError(
                f"Received an invalid value for `n_neurons`, expected "
                f"a strictly positive integer, got {self._n_neurons}."
            )

        if "Block" not in self.block_type:
            raise ValueError(f"`name` has to contain `Block`. Got {self.name}")

    def _val__get_backcast_coefficients(
        self, output: tf.Tensor, *args: list, **kwargs: dict
    ) -> None:  # pylint: disable=unused-argument
        msg_error = f"""The forecast layer doesn't match the desired shape. Got {output.shape},
                expected (..., { self._output_first_dim_forecast}, {self.input_width}"""
        assert tf.rank(output) >= 2, msg_error
        assert (
            output.shape[-1] == self.label_width
            or output.shape[-2] == self._output_first_dim_backcast
        ), msg_error

    def _val__get_forecast_coefficients(
        self, output: tf.Tensor, *args: list, **kwargs: dict
    ) -> None:  # pylint: disable=unused-argument
        msg_error = f"""The forecast layer doesn't match the desired shape. Got {output.shape},
                expected (..., { self._output_first_dim_forecast}, {self.input_width}"""
        assert tf.rank(output) >= 2, msg_error
        assert (
            output.shape[-1] == self.input_width
            or output.shape[-2] == self._output_first_dim_forecast
        ), msg_error

    def __repr__(self) -> str:
        """Return the representation."""
        return self._block_type


class TrendBlock(BaseBlock, AutopycoinBaseClass):
    """
    Trend block definition.

    This layer represents the smaller part of the nbeats model.
    Final layers are constrained which define a polynomial function of small degree p.
    Therefore it is possible to get explanation from this block.

    Parameters
    ----------
    input_width : int
        Horizon time to forecast.
    label_width : int
        Past to rebuild. Usually, label_width is 1 to 7 times longer than input_width.
    p_degree : int
        Degree of the polynomial function.
    n_neurons : int
        Number of neurons in Fully connected layers.
    drop_rate : float
        Rate of the dropout layer. This is used to estimate the epistemic error.
        Expected a value between 0 and 1. Default to 0.

    Attributes
    ----------
    p_degree : int
    input_width : int
    label_width : int
    drop_rate : float

    Raises
    ------
    ValueError:
        `drop_rate` not between 0 and 1.
        all others arguments are not strictly positive integers.

    Examples
    --------
    >>> from autopycoin.models import TrendBlock, SeasonalityBlock, Stack, NBEATS
    >>> from autopycoin.losses import QuantileLossError
    >>> trend_block = TrendBlock(input_width=10,
    ...                          label_width=20,
    ...                          p_degree=2,
    ...                          n_neurons=16,
    ...                          drop_rate=0.1,
    ...                          name="trend_block")
    >>>
    >>> seasonality_block = SeasonalityBlock(input_width=10,
    ...                                      label_width=20,
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
        input_width: int,
        label_width: int,
        p_degree: int,
        n_neurons: int,
        drop_rate: float = 0.0,
        **kwargs: dict,
    ):

        super().__init__(
            input_width=input_width,
            label_width=label_width,
            output_first_dim_forecast=p_degree,
            output_first_dim_backcast=p_degree,
            n_neurons=n_neurons,
            drop_rate=drop_rate,
            g_trainable=False,
            interpretable=True,
            block_type="TrendBlock",
            **kwargs,
        )

        # Shape (-1, 1) in order to broadcast input_width to all p degrees
        self._p_degree = p_degree

    @property
    def p_degree(self) -> int:
        """Return the degree of the trend equation."""
        return self._p_degree

    def coefficient_factory(self, input_width: float, p_degree: tf.Tensor) -> tf.Tensor:
        """
        Compute the coefficients used in the last layer a.k.a g layer.

        Parameters
        ----------
        input_width : int
        p_degree: int

        Returns
        -------
        coefficients : `tensor with shape (p_degree, input_width)`
            Coefficients of the g layer.
        """
        coefficients = (tf.range(input_width) / input_width) ** p_degree

        return coefficients

    def _get_forecast_coefficients(self) -> tf.Tensor:
        """
        Return the coefficients used in the forecast layer a.k.a gf layer.

        Returns
        -------
        coefficients
            `Tensor` of shape (d0, ..., output_first_dim_forecast, input_width)

        Raises
        ------
        ValueError
            Raise an error if the coefficients tensor shape is not equal to
            (d0, ..., output_first_dim_forecast, input_width).
        """
        # Set weights with calculated coef
        forecast_coef = self.coefficient_factory(
            self.input_width,
            tf.expand_dims(tf.range(self.p_degree, dtype=floatx()), axis=-1),
        )
        return self.add_weight(
            shape=forecast_coef.shape,
            initializer=tf.constant_initializer(forecast_coef.numpy()),
            trainable=self.is_g_trainable,
            name=f"gf_{self.name}",
        )

    def _get_backcast_coefficients(self) -> tf.Tensor:
        """
        Return the coefficients used in the backcast a.k.a gb layer layer.

        Returns
        -------
        coefficients
            `Tensor` of shape (d0, ..., output_first_dim_backcast, label_width)
        """

        # Set weights with calculated coef
        backcast_coef = self.coefficient_factory(
            self.label_width,
            tf.expand_dims(tf.range(self.p_degree, dtype=floatx()), axis=-1),
        )
        return self.add_weight(
            shape=backcast_coef.shape,
            initializer=tf.constant_initializer(backcast_coef.numpy()),
            trainable=self.is_g_trainable,
            name=f"gf_{self.name}",
        )

    def get_config(self) -> dict:
        """get_condig method from tensorflow."""
        config = super().get_config()
        config.update(
            {
                "p_degree": self.p_degree,
                "input_width": self.input_width,
                "label_width": self.label_width,
                "n_neurons": self._n_neurons,
                "drop_rate": self.drop_rate,
            }
        )
        return config

    def _val___init__(
        self, output: None, *args: list, **kwargs: dict
    ) -> None:  # pylint: disable=unused-argument
        if self.p_degree < 0:
            raise ValueError(
                f"Received an invalid value for `p_degree`, expected "
                f"a positive integer, got {self.p_degree}."
            )


class SeasonalityBlock(BaseBlock, AutopycoinBaseClass):
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
    input_width : int
        Horizon time to forecast.
    label_width : int
        Past to rebuild.
    periods : List[int]
        Compute the fourier serie period in the forecasting equation.
        If it's a list all periods are taken into account in the calculation.
    back_periods : List[int]
        Compute the fourier serie period in the backcasting equation.
        If it's a list all periods are taken into account in the calculation.
    forecast_fourier_order : List[int]
        Compute the fourier order. each order element is linked the respective period.
    backcast_fourier_order : List[int]
        Compute the fourier order. each order element is linked the respective back period.
    n_neurons : int
        Number of neurons in Fully connected layers.
    drop_rate : float
        Rate of the dropout layer. This is used to estimate the epistemic error.
        Expected a value between 0 and 1. Default to 0.

    Attributes
    ----------
    input_width : int
    label_width : int
    periods : List[int]
    back_periods : List[int]
    forecast_fourier_order : List[int]
    backcast_fourier_order : List[int]
    drop_rate : float

    Raises
    ------
    ValueError:
        `drop_rate` not between 0 and 1.
        `periods` and `forecast_fourier_order` are not strictly positive not empty list.
        `back_periods` and `backcast_fourier_order` are not strictly positive not empty list.
        all others arguments are not strictly positive integers.

    Examples
    --------
    >>> from autopycoin.models import TrendBlock, SeasonalityBlock, Stack, NBEATS
    >>> from autopycoin.losses import QuantileLossError
    >>> trend_block = TrendBlock(input_width=10,
    ...                          label_width=20,
    ...                          p_degree=2,
    ...                          n_neurons=16,
    ...                          drop_rate=0.1,
    ...                          name="trend_block")
    >>> seasonality_block = SeasonalityBlock(input_width=10,
    ...                                      label_width=20,
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
        input_width: int,
        label_width: int,
        periods: List[int],
        back_periods: List[int],
        forecast_fourier_order: List[int],
        backcast_fourier_order: List[int],
        n_neurons: int,
        drop_rate: float = 0.0,
        **kwargs: dict,
    ):

        # Workout the number of neurons needed to compute seasonality
        # coefficients
        forecast_neurons = tf.reduce_sum(2 * forecast_fourier_order)
        backcast_neurons = tf.reduce_sum(2 * backcast_fourier_order)

        super(SeasonalityBlock, self).__init__(
            input_width=input_width,
            label_width=label_width,
            output_first_dim_forecast=forecast_neurons,
            output_first_dim_backcast=backcast_neurons,
            n_neurons=n_neurons,
            drop_rate=drop_rate,
            g_trainable=False,
            interpretable=True,
            block_type="SeasonalityBlock",
            **kwargs,
        )

        self._periods = periods
        self._back_periods = back_periods
        self._forecast_fourier_order = forecast_fourier_order
        self._backcast_fourier_order = backcast_fourier_order

    @property
    def periods(self) -> List[int]:
        """Return periods."""
        return self._periods

    @property
    def back_periods(self) -> List[int]:
        """Return back periods."""
        return self._back_periods

    @property
    def forecast_fourier_order(self) -> List[int]:
        """Return fourier order."""
        return self._forecast_fourier_order

    @property
    def backcast_fourier_order(self) -> List[int]:
        """Return fourier order."""
        return self._backcast_fourier_order

    def coefficient_factory(
        self,
        input_width: float,
        periods: Tuple[int, ...],
        fourier_orders: Tuple[int, ...],
    ) -> tf.Tensor:
        """
        Compute the coefficients used in the last layer a.k.a g constrained layer.

        Parameters
        ----------
        input_width : int
        periods : Tuple[int, ...]
        fourier_orders : Tuple[int, ...]

        Returns
        -------
        coefficients : `tensor with shape (periods * fourier_orders, input_width)`
            Coefficients of the g layer.
        """

        # Shape (-1, 1) in order to broadcast periods to all time units
        periods = tf.cast(tf.reshape(periods, shape=(-1, 1)), dtype=floatx())
        time_forecast = tf.range(input_width, dtype=floatx())

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

    def _get_forecast_coefficients(self) -> tf.Tensor:
        """
        Return the coefficients used in the forecast layer a.k.a gf layer.

        Returns
        -------
        coefficients
            `Tensor` of shape (d0, ..., output_first_dim_forecast, input_width)

        Raises
        ------
        ValueError
            Raise an error if the coefficients tensor shape is not equal to
            (d0, ..., output_first_dim_forecast, input_width).
        """
        # Set weights with calculated coef
        forecast_coef = self.coefficient_factory(
            self.input_width, self.periods, self.forecast_fourier_order
        )
        return self.add_weight(
            shape=forecast_coef.shape,
            initializer=tf.constant_initializer(forecast_coef.numpy()),
            trainable=self.is_g_trainable,
            name=f"gf_{self.name}",
        )

    def _get_backcast_coefficients(self) -> tf.Tensor:
        """
        Return the coefficients used in the backcast a.k.a gb layer layer.

        Returns
        -------
        coefficients
            `Tensor` of shape (d0, ..., output_first_dim_backcast, label_width)
        """
        # Set weights with calculated coef
        backcast_coef = self.coefficient_factory(
            self.label_width, self.back_periods, self.backcast_fourier_order
        )
        return self.add_weight(
            shape=backcast_coef.shape,
            initializer=tf.constant_initializer(backcast_coef.numpy()),
            trainable=self.is_g_trainable,
            name=f"gf_{self.name}",
        )

    def get_config(self) -> dict:
        """get_config method from tensorflow."""
        config = super().get_config()
        config.update(
            {
                "input_width": self.input_width,
                "label_width": self.label_width,
                "n_neurons": self._n_neurons,
                "periods": self.periods,
                "back_periods": self.back_periods,
                "forecast_fourier_order": self.forecast_fourier_order,
                "backcast_fourier_order": self.backcast_fourier_order,
                "drop_rate": self.drop_rate,
            }
        )
        return config

    def _val___init__(
        self, output: None, *args: list, **kwargs: dict
    ) -> None:  # pylint: disable=unused-argument
        if not (
            all(period == abs(period) for period in self.periods)
            and len(self.periods) > 0
        ):
            raise ValueError(
                f"""`periods` have to be a non-empty list and all elements have to be strictly positives values.
            Got {self.periods}."""
            )

        if len(self.periods) != len(self.forecast_fourier_order):
            raise ValueError(
                f"`periods` and `forecast_fourier_order` are expected "
                f"to have the same length, got "
                f"{len(self.periods)} and {len(self.forecast_fourier_order)} respectively."
            )

        if not (
            all(back_period == abs(back_period) for back_period in self.back_periods)
            and len(self.back_periods) > 0
        ):
            raise ValueError(
                f"""`back_periods` have to be a non-empty list and all elements inside have to be strictly positives values.
            Got {self.back_periods}."""
            )

        if len(self.back_periods) != len(self.backcast_fourier_order):
            raise ValueError(
                f"`back_periods` and `backcast_fourier_order` are expected"
                f"to have the same length, got "
                f"{len(self.back_periods)} and {len(self.backcast_fourier_order)} respectively."
            )


class GenericBlock(BaseBlock, AutopycoinBaseClass):
    """
    Generic block definition as described in the paper.

    This layer represents the smaller part of a nbeats model.
    We can't have explanation from this kind of block because g coefficients
    are learnt.

    Parameters
    ----------
    input_width : int
        Horizon time to forecast.
    label_width : int
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
    input_width : int
    label_width : int
    drop_rate : float

    Examples
    --------
    >>> from autopycoin.models import GenericBlock, Stack, NBEATS
    >>> from autopycoin.losses import QuantileLossError
    >>> generic_block = GenericBlock(input_width=10,
    ...                          label_width=20,
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
        input_width: int,
        label_width: int,
        forecast_neurons: int,
        backcast_neurons: int,
        n_neurons: int,
        drop_rate: float = 0.1,
        **kwargs: dict,
    ):

        super().__init__(
            input_width=input_width,
            label_width=label_width,
            output_first_dim_forecast=forecast_neurons,
            output_first_dim_backcast=backcast_neurons,
            n_neurons=n_neurons,
            drop_rate=drop_rate,
            g_trainable=True,
            interpretable=False,
            block_type="GenericBlock",
            **kwargs,
        )

    def coefficient_factory(self, input_width: int, neurons: int) -> tf.Tensor:
        """
        Compute the coefficients used in the last layer a.k.a g layer.
        This function is used in `_get_forecast_coefficients` and
        `_get_backcast_coefficients`.

        Parameters
        ----------
        input_width : int
        neurons : int

        Returns
        -------
        coefficients : `tensor with shape (input_width, neurons)`
            Coefficients of the g layer.
        """

        coefficients = tf.keras.initializers.GlorotUniform(seed=42)(
            shape=(neurons, input_width)
        )

        return coefficients

    def _get_forecast_coefficients(self):
        """
        Return the coefficients used in the forecast layer a.k.a gf layer.

        Returns
        -------
        coefficients
            `Tensor` of shape (d0, ..., output_first_dim_forecast, input_width)

        Raises
        ------
        ValueError
            Raise an error if the coefficients tensor shape is not equal to
            (d0, ..., output_first_dim_forecast, input_width).
        """
        # Set weights with calculated coef
        forecast_coef = self.coefficient_factory(
            self.input_width, self._output_first_dim_forecast
        )
        return self.add_weight(
            shape=forecast_coef.shape,
            initializer=tf.constant_initializer(forecast_coef.numpy()),
            trainable=self.is_g_trainable,
            name=f"gf_{self.name}",
        )

    def _get_backcast_coefficients(self):
        """
        Return the coefficients used in the backcast a.k.a gb layer layer.

        Returns
        -------
        coefficients
            `Tensor` of shape (d0, ..., output_first_dim_backcast, label_width)
        """
        # Set weights with calculated coef
        backcast_coef = self.coefficient_factory(
            self.label_width, self._output_first_dim_backcast
        )
        return self.add_weight(
            shape=backcast_coef.shape,
            initializer=tf.constant_initializer(backcast_coef.numpy()),
            trainable=self.is_g_trainable,
            name=f"gf_{self.name}",
        )

    def get_config(self) -> dict:
        """get_config method from tensorflow."""
        config = super().get_config()
        config.update(
            {
                "input_width": self.input_width,
                "label_width": self.label_width,
                "forecast_neurons": self._output_first_dim_forecast,
                "backcast_neurons": self._output_first_dim_backcast,
                "n_neurons": self._n_neurons,
                "drop_rate": self.drop_rate,
            }
        )
        return config

    def _val___init__(
        self, output: None, *args: list, **kwargs: dict
    ) -> None:  # pylint: disable=unused-argument
        if self._output_first_dim_backcast <= 0 or self._output_first_dim_forecast <= 0:
            raise ValueError(
                f"""Received an invalid value for `forecast_neurons` or `backcast_neurons`,
                expected strictly postive integers, got {self._output_first_dim_backcast} and {self._output_first_dim_forecast}"""
            )
