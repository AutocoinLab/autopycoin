from typing import Tuple, List, Union
import numpy as np
import abc

import tensorflow as tf
from tensorflow.keras.layers import Dropout, InputSpec

# from tensorflow.keras import Model
from tensorflow.keras.backend import floatx

from ..utils import range_dims, convert_to_list
from .base_layer import Layer
from ..baseclass import AutopycoinBaseClass
from ..asserts import greater_or_equal, equal_length, is_between


class BaseBlock(Layer, AutopycoinBaseClass):
    """
    Base class of a nbeats block.

    Your custom block needs to inherit from it.

    Parameters
    ----------
    label_width : int
        Horizon time to forecast.
    n_neurons : int
        Number of neurons in the fully connected layers.
    drop_rate : float
        Rate of the dropout layer.
        This is used to estimate the epistemic error.
        Expected a value between 0 and 1. Default to 0.
    name : str
        The name of the layer. It defines also the `block_type` attribute.

    Attributes
    ----------
    label_width : int
    input_width : int
    input_spec : `ÌnputSpec`
    drop_rate : float
    is_interpretable : bool
    is_g_trainable : bool
    block_type : str

    Raises
    ------
    ValueError:
        If `name` doesn't contain `Block`.
        `drop_rate` is not between 0 and 1.
        All others arguments are not strictly positive integers.

    """

    NOT_INSPECT = ["call", "_output"]

    def __init__(
        self,
        label_width: int,
        n_neurons: int,
        drop_rate: float,
        g_trainable: bool = False,
        interpretable: bool = False,
        block_type: str = "BaseBlock",
        **kwargs: dict,
    ):

        super().__init__(**kwargs)

        self._label_width = label_width
        self._drop_rate = drop_rate
        self._n_neurons = n_neurons
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
        self._input_width = tf.compat.dimension_value(input_shape[-1])
        if self.input_width is None:
            raise ValueError(
                f"The last dimension of the inputs"
                f" should be defined. Found {self.input_width}."
            )

        self.input_spec = InputSpec(min_ndim=2, axes={-1: self.input_width})

        # multi univariate inputs, we need to handle the case where quantiles are defined
        self._base_shape = []
        if input_shape.rank > 2 and self._n_quantiles < 2:
            self._base_shape = input_shape.as_list()[:-2]
        elif input_shape.rank > 3 and self._n_quantiles > 1:
            self._base_shape = input_shape.as_list()[1:-2]

        # Computing fc layers
        dim = self.input_width
        self.fc_stack = []
        for count in range(4):
            self.fc_stack.append(
                (
                    self.add_weight(
                        shape=self._base_shape + [dim, self._n_neurons],
                        name=f"fc_kernel_{self.name}_{count}",
                    ),
                    self.add_weight(
                        shape=self._base_shape + [1, self._n_neurons],
                        initializer="zeros",
                        name=f"fc_bias_{self.name}_{count}",
                    ),
                )
            )
            dim = self._n_neurons

        self.dropout = Dropout(self.drop_rate)

        self._build_branch(self.label_width, branch_name="forecast")
        self._build_branch(self.input_width, branch_name="backcast")
        self.built = True

    def _build_branch(
        self, output_last_dim: int, branch_name: str
    ) -> None:
        """
        Build forecast and backcast branches.
        """

        coef = self._get_coefficients(output_last_dim, branch_name=branch_name)

        # If the model is compiling with a loss error defining uncertainty then
        # broadcast the output to take into account this uncertainty.
        shape_fc = self._base_shape + [self._n_neurons, coef.shape[0]]
        if self._n_quantiles > 1:
            print(self._n_quantiles)
            shape_fc.insert(0, self._n_quantiles)
        fc = self.add_weight(shape=shape_fc, name=f"fc_{branch_name}_{self.name}")

        # Set attributes dynamically
        setattr(self, f"fc_{branch_name}", fc)
        setattr(self, f"{branch_name}_coef", coef)

    @abc.abstractmethod
    def _get_coefficients(self, output_last_dim: int, branch_name: str) -> tf.Tensor:
        """
        Return the coefficients used in the forecast and backcast layer a.k.a g layer
        by calling coefficient_factory.
        This method needs to be overriden.

        Returns
        -------
        coefficients
            `Tensor` of shape (d0, ..., output_first_dim_forecast, label_width)

        Raises
        ------
        ValueError
            Raise an error if the coefficients tensor shape is not equal to
            (d0, ..., output_first_dim_forecast, label_width).
        """
        raise NotImplementedError(
            "When subclassing the `BaseBlock` class, you should "
            "implement a `_get_forecast_coefficients` method."
        )

    @abc.abstractmethod
    def _coefficient_factory(self, *args: list, **kwargs: dict) -> tf.Tensor:
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

    def call(
        self, inputs: tf.Tensor
    ) -> Tuple[tf.Tensor]:  # pylint: disable=arguments-differ
        """Call method from tensorflow."""

        for kernel, bias in self.fc_stack:
            # shape: (Batch_size, n_neurons)
            inputs = tf.add(tf.matmul(inputs, kernel), bias)
            inputs = tf.nn.relu(inputs)
            inputs = self.dropout(inputs, training=True)

        # shape: (quantiles, Batch_size, backcast)
        reconstructed_inputs = self._output(
            inputs, self.fc_backcast, self.backcast_coef
        )  # layers fc and coef created in _build_branch

        # shape: (quantiles, Batch_size, forecast)
        outputs = self._output(
            inputs, self.fc_forecast, self.forecast_coef
        )  # layers fc and coef created in _build_branch
        return outputs, reconstructed_inputs

    def _output(
        self, inputs: tf.Tensor, fc: tf.Tensor, coef: tf.Tensor
    ) -> Tuple[tf.Tensor]:  # pylint: disable=arguments-differ
        """Call method."""

        # shape: (Batch_size, output_first_dim_backcast)
        theta = tf.matmul(inputs, fc)
        return tf.matmul(theta, coef)

    def compute_output_shape(
        self, input_shape: tf.TensorShape
    ) -> Tuple[tf.TensorShape]:
        """output method from tensoflow."""

        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError(
                "The innermost dimension of input_shape must be defined, but saw: %s"
                % (input_shape,)
            )

        # If the model is compiled with a loss error defining uncertainty then
        # reshape the output to take into account this uncertainty.
        output_shape_forecast = input_shape[:-1] + [self.label_width]
        output_shape_backcast = input_shape[:-1] + [last_dim]
        if self._n_quantiles > 1:
            output_shape_forecast.insert(0, self._n_quantiles)
            output_shape_backcast.insert(0, self._n_quantiles)

        return [
            tf.TensorShape(output_shape_forecast),
            tf.TensorShape(output_shape_backcast),
        ]

    @property
    def label_width(self) -> int:
        """Return the label_width."""
        return self._label_width

    @property
    def input_width(self) -> int:
        """Return the input_width."""
        return self._input_width

    @property
    def drop_rate(self) -> float:
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

    def __repr__(self) -> str:
        """Return the representation."""
        return self._block_type

    def _val___init__(
        self, output: None, *args: list, **kwargs: dict
    ) -> None:  # pylint: disable=unused-argument

        is_between(self.drop_rate, 0, 1, 'drop_rate')
        greater_or_equal(self.label_width, 0, 'label_width')
        greater_or_equal(self._n_neurons, 0, 'n_neurons')

        if "Block" not in self.block_type:
            raise ValueError(f"`name` has to contain `Block`. Got {self.name}")


class TrendBlock(BaseBlock, AutopycoinBaseClass):
    """
    Trend block definition.

    This layer represents the smaller part of the nbeats model.
    Final layers are constrained which define a polynomial function of small degree p.
    Therefore it is possible to get explanation from this block.

    Parameters
    ----------
    label_width : int
        Horizon time to forecast.
    p_degree : int | float
        Degree of the polynomial function.
    n_neurons : int
        Number of neurons in Fully connected layers.
    drop_rate : float
        Rate of the dropout layer. This is used to estimate the epistemic error.
        Expected a value between 0 and 1. Default to 0.

    Attributes
    ----------
    p_degree : int
    label_width : int
    input_width : int
    input_spec : `ÌnputSpec`
    drop_rate : float
    is_interpretable : bool
    is_g_trainable : bool
    block_type : str

    Raises
    ------
    ValueError:
        `drop_rate` is not between 0 and 1.
        All others arguments are not strictly positive integers.

    Examples
    --------
    >>> from autopycoin.layers import TrendBlock, SeasonalityBlock
    >>> from autopycoin.models import Stack, NBEATS
    >>> from autopycoin.losses import QuantileLossError
    >>> trend_block = TrendBlock(label_width=10,
    ...                          p_degree=2,
    ...                          n_neurons=16,
    ...                          drop_rate=0.1,
    ...                          name="trend_block")
    >>>
    >>> seasonality_block = SeasonalityBlock(label_width=10,
    ...                                      forecast_periods=[10],
    ...                                      backcast_periods=[20],
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

    Notes
    -----
    input shape:
    N-D tensor with shape: (batch_size, ..., units).
    The most common situation would be a 2D input with shape (batch_size, units).

    output shape:
    N-D tensor with shape: (quantiles, batch_size, ..., units) or (batch_size, ..., units) .
    For instance, for a 2D input with shape (batch_size, units),
    the output would have shape (batch_size, units).
    With a QuantileLossError with 2 quantiles or higher the output would have shape (quantiles, batch_size, units).
    """

    def __init__(
        self,
        label_width: int,
        p_degree: Union[int, float] = 2,
        n_neurons: int = 32,
        drop_rate: float = 0.0,
        **kwargs: dict,
    ):

        super().__init__(
            label_width=label_width,
            n_neurons=n_neurons,
            drop_rate=drop_rate,
            g_trainable=False,
            interpretable=True,
            block_type="TrendBlock",
            **kwargs,
        )

        # Shape (-1, 1) in order to broadcast label_width to all p degrees
        self._p_degree = p_degree

    def _coefficient_factory(
        self, output_last_dim: float, p_degrees: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute the coefficients used in the last layer a.k.a g layer.

        Parameters
        ----------
        output_last_dim : int
        p_degree: int

        Returns
        -------
        coefficients : `tensor with shape (p_degree, label_width)`
            Coefficients of the g layer.
        """
        coefficients = (tf.range(output_last_dim) / output_last_dim) ** p_degrees

        return coefficients

    def _get_coefficients(self, output_last_dim: float, branch_name: str) -> tf.Tensor:
        """
        Return the coefficients calculated by the  `_coefficients_factory` method.
        """

        # Set weights with calculated coef
        coef = self._coefficient_factory(
            output_last_dim,
            range_dims(self.p_degree + 1, shape=(-1, 1)),
        )
        return self.add_weight(
            shape=coef.shape,
            initializer=tf.constant_initializer(coef.numpy()),
            trainable=self.is_g_trainable,
            name=f"g_{branch_name}_{self.name}",
        )

    def get_config(self) -> dict:
        """get_condig method from tensorflow."""
        config = super().get_config()
        config.update(
            {
                "p_degree": self.p_degree,
                "label_width": self.label_width,
                "n_neurons": self._n_neurons,
                "drop_rate": self.drop_rate,
            }
        )
        return config

    @property
    def p_degree(self) -> int:
        """Return the degree of the trend equation."""
        return self._p_degree

    def _val___init__(
        self, output: None, *args: list, **kwargs: dict
    ) -> None:  # pylint: disable=unused-argument
        greater_or_equal(self.p_degree, 0, 'p_degree')


SEASONALITY_TYPE = Union[Union[int, float], List[Union[int, float]]]

class SeasonalityBlock(BaseBlock, AutopycoinBaseClass):
    """
    Seasonality block definition.

    This layer represents the smaller part of nbeats model.
    Its internal layers are defining a fourier series.
    We introduced notion of fourier orders and seasonality periods which is not 
    used in the original paper.
    It is possible to get explanation from this block.

    Parameters
    ----------
    label_width : int
        Horizon time to forecast.
    forecast_periods : int | float | List[int | float]
        Defines the periods used in the fourier equation.
        Default to `label_width`/2 as describe in the original paper.
    backcast_periods : int | float | List[int | float]
        Compute the fourier serie period in the backcasting equation.
        Default to `input_width`/2 as describe in the original paper.
    forecast_fourier_order : int | float | List[int | float]
        Compute the fourier orders.
        Each element is the order of its respective period.
        Default to `label_width`/2 as describe in the original paper.
    backcast_fourier_order : int | float | List[int | float]
        Compute the fourier orders.
        Each element is the order of its respective back period.
        Default to `input_width`/2 as describe in the original paper.
    n_neurons : int
        Number of neurons in the fully connected layers.
    drop_rate : float
        Rate of the dropout layer.
        This is used to estimate the epistemic error.
        Expected a value between 0 and 1. Default to 0.

    Attributes
    ----------
    label_width : int
    input_width : int
    input_spec : `ÌnputSpec`
    drop_rate : float
    periods : int | float | List[int | float]
    back_periods : int | float | List[int | float]
        if not provided, then it is set during `build` method.
    forecast_fourier_order : int | float | List[int | float]
    backcast_fourier_order : int | float | List[int | float]
        if not provided, then it is set during `build` method.

    Raises
    ------
    ValueError:
        `drop_rate` not between 0 and 1.
        `periods` and `forecast_fourier_order` or their elements are not strictly positive values
        `back_periods` and `backcast_fourier_order` or their elements are not strictly positive values
        `backcast_fourier_order` and `forecast_fourier_order` don't have the same length.
        all others arguments are not strictly positive integers.

    Examples
    --------
    >>> from autopycoin.layers import TrendBlock, SeasonalityBlock
    >>> from autopycoin.models import Stack, NBEATS
    >>> from autopycoin.losses import QuantileLossError
    >>> trend_block = TrendBlock(label_width=10,
    ...                          p_degree=2,
    ...                          n_neurons=16,
    ...                          drop_rate=0.1,
    ...                          name="trend_block")
    >>> seasonality_block = SeasonalityBlock(label_width=10,
    ...                                      forecast_periods=[10],
    ...                                      backcast_periods=[20],
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

    Notes
    -----
    input shape:
    N-D tensor with shape: (batch_size, ..., units).
    The most common situation would be a 2D input with shape (batch_size, units).

    output shape:
    N-D tensor with shape: (quantiles, batch_size, ..., units) or (batch_size, ..., units) .
    For instance, for a 2D input with shape (batch_size, units),
    the output would have shape (batch_size, units).
    With a QuantileLossError with 2 quantiles or higher the output would have shape (quantiles, batch_size, units).
    """

    def __init__(
        self,
        label_width: int,
        forecast_periods: SEASONALITY_TYPE = None,
        backcast_periods: SEASONALITY_TYPE = None,
        forecast_fourier_order: SEASONALITY_TYPE = None,
        backcast_fourier_order: SEASONALITY_TYPE = None,
        n_neurons: int = 32,
        drop_rate: float = 0.0,
        **kwargs: dict,
    ):

        super(SeasonalityBlock, self).__init__(
            label_width=label_width,
            n_neurons=n_neurons,
            drop_rate=drop_rate,
            g_trainable=False,
            interpretable=True,
            block_type="SeasonalityBlock",
            **kwargs,
        )

        # forecast periods and fourier order can be calculated if not provided
        # backcast has to wait unitl `build` is called
        self._forecast_periods = (
            forecast_periods if forecast_periods else int(label_width / 2)
        )
        self._forecast_fourier_order = (
            forecast_fourier_order if forecast_fourier_order
            else self._forecast_periods
        )
        # backcast_fourier_order and backcast_periods can't be calculated
        self._backcast_periods = backcast_periods
        self._backcast_fourier_order = backcast_fourier_order

    def build(self, input_shape: tf.TensorShape):
        """Build method from tensorflow."""

        # if None then set an default value based on the input shape
        self._backcast_periods = (
            self._backcast_periods if self._backcast_periods else int(input_shape[-1] / 2)
        )
        self._backcast_fourier_order = (
            self._backcast_fourier_order if self._backcast_fourier_order else self._backcast_periods
        )

        super().build(input_shape)

    def _coefficient_factory(
        self,
        output_last_dim: float,
        periods: List[float],
        fourier_orders: List[float],
    ) -> tf.Tensor:
        """
        Compute the coefficients used in the last layer a.k.a g constrained layer.

        Parameters
        ----------
        input_width : float
        periods : Tuple[float, ...]
        fourier_orders : Tuple[float, ...]

        Returns
        -------
        coefficients : `tensor with shape (periods * fourier_orders, label_width)`
            Coefficients of the g layer.
        """

        # Shape (-1, 1) in order to broadcast periods to all time units
        periods = tf.reshape(periods, shape=(-1, 1, 1))
        time_forecast = tf.range(output_last_dim)

        seasonality = 2. * np.pi * time_forecast / periods
        seasonality = tf.expand_dims(tf.ragged.range(fourier_orders), axis=-1) * seasonality
        seasonality = tf.concat((tf.sin(seasonality), tf.cos(seasonality)), axis=0)
        return seasonality.flat_values

    def _get_coefficients(self, output_last_dim: float, branch_name: str) -> tf.Tensor:
        """
        Return the coefficients calculated by the  `_coefficients_factory` method.
        """

        periods = convert_to_list(getattr(self, branch_name + "_periods"))
        fourier_order = convert_to_list(getattr(self, branch_name + "_fourier_order"))

        # Set weights with calculated coef
        coef = self._coefficient_factory(output_last_dim, periods, fourier_order)

        return self.add_weight(
            shape=coef.shape,
            initializer=tf.constant_initializer(coef.numpy()),
            trainable=self.is_g_trainable,
            name=f"g_{branch_name}_{self.name}",
        )

    def get_config(self) -> dict:
        """get_config method from tensorflow."""
        config = super().get_config()
        config.update(
            {
                "label_width": self.label_width,
                "forecast_periods": self.forecast_periods,
                "backcast_periods": self.backcast_periods,
                "forecast_fourier_order": self.forecast_fourier_order,
                "backcast_fourier_order": self.backcast_fourier_order,
                "n_neurons": self._n_neurons,
                "drop_rate": self.drop_rate,
            }
        )
        return config

    @property
    def forecast_periods(self) -> List[int]:
        """Return periods."""
        return self._forecast_periods

    @property
    def backcast_periods(self) -> List[int]:
        """Return back periods."""
        return self._backcast_periods

    @property
    def forecast_fourier_order(self) -> List[int]:
        """Return fourier order."""
        return self._forecast_fourier_order

    @property
    def backcast_fourier_order(self) -> List[int]:
        """Return fourier order."""
        return self._backcast_fourier_order

    def _val___init__(
        self, output: None, *args: list, **kwargs: dict
    ) -> None:  # pylint: disable=unused-argument
        greater_or_equal(self.forecast_periods, 0, 'forecast_periods')
        greater_or_equal(self.forecast_fourier_order, 0, 'forecast_fourier_order')
        equal_length(self.forecast_periods, self.forecast_fourier_order,
                            'forecast_periods', 'forecast_fourier_order')

        if self.backcast_periods is not None:
            greater_or_equal(self.backcast_periods, 0, 'backcast_periods')
        if self.backcast_fourier_order is not None:
            greater_or_equal(self.backcast_fourier_order, 0, 'backcast_fourier_order')
            equal_length(self.backcast_periods, self.backcast_fourier_order,
                            'backcast_periods', 'backcast_fourier_order')


class GenericBlock(BaseBlock, AutopycoinBaseClass):
    """
    Generic block definition as described in the paper.

    This layer represents the smaller part of a nbeats model.
    We can't have explanation from this  block because g coefficients are learnt.

    Parameters
    ----------
    label_width : int
        Horizon time to forecast.
    g_forecast_neurons : int
        Dimensionality if the gf layer.
    g_backcast_neurons : int
        Dimensionality if the gb layer.
    n_neurons : int
        Number of neurons in Fully connected layers.
    drop_rate : float
        Rate of the dropout layer. This is used to estimate the epistemic error.
        Expected a value between 0 and 1. Default to 0.1.

    Attributes
    ----------
    label_width : int
    input_width : int
    input_spec : `InputSpec`
    drop_rate : float

    Examples
    --------
    >>> from autopycoin.layers import GenericBlock
    >>> from autopycoin.models import Stack, NBEATS
    >>> from autopycoin.losses import QuantileLossError
    >>> generic_block = GenericBlock(label_width=10,
    ...                          n_neurons=16,
    ...                          g_forecast_neurons=16,
    ...                          g_backcast_neurons=16,
    ...                          drop_rate=0.1,
    ...                          name="generic_block")
    >>> generic_blocks = [generic_block for _ in range(3)]
    >>> generic_stacks = Stack(generic_blocks, name="generic_stack")
    >>> # Model definition and compiling
    >>> model = NBEATS([generic_stacks, generic_stacks], name="generic_NBEATS")

    Notes
    -----
    input shape:
    N-D tensor with shape: (batch_size, ..., units).
    The most common situation would be a 2D input with shape (batch_size, units).

    output shape:
    N-D tensor with shape: (quantiles, batch_size, ..., units) or (batch_size, ..., units) .
    For instance, for a 2D input with shape (batch_size, units),
    the output would have shape (batch_size, units).
    With a QuantileLossError with 2 quantiles or higher the output would have shape (quantiles, batch_size, units).
    """

    def __init__(
        self,
        label_width: int,
        g_forecast_neurons: int = 32,
        g_backcast_neurons: int = 32,
        n_neurons: int = 32,
        drop_rate: float = 0.1,
        **kwargs: dict,
    ):

        super().__init__(
            label_width=label_width,
            n_neurons=n_neurons,
            drop_rate=drop_rate,
            g_trainable=True,
            interpretable=False,
            block_type="GenericBlock",
            **kwargs,
        )
        
        self._g_forecast_neurons = g_forecast_neurons
        self._g_backcast_neurons = g_backcast_neurons

    def _coefficient_factory(self, output_last_dim: int, neurons: int) -> tf.Tensor:
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
        coefficients : `tensor with shape (label_width, neurons)`
            Coefficients of the g layer.
        """

        coefficients = tf.keras.initializers.GlorotUniform(seed=42)(
            shape= self._base_shape + [neurons, output_last_dim]
        )

        return coefficients

    def _get_coefficients(self, output_last_dim: int, branch_name: str) -> tf.Tensor:
        """
        Return the coefficients used in forecast and backcast layer a.k.a g layer.

        Returns
        -------
        coefficients: `Tensor` of shape (d0, ..., label_width)
        """

        neurons = getattr(self, "g_" + branch_name + "_neurons")
        # Set weights with calculated coef
        coef = self._coefficient_factory(output_last_dim, neurons)
        return self.add_weight(
            shape=coef.shape,
            initializer=tf.constant_initializer(coef.numpy()),
            trainable=self.is_g_trainable,
            name=f"g_{branch_name}_{self.name}",
        )

    def get_config(self) -> dict:
        """get_config method from tensorflow."""
        config = super().get_config()
        config.update(
            {
                "label_width": self.label_width,
                "g_forecast_neurons": self.g_forecast_neurons,
                "g_backcast_neurons": self.g_backcast_neurons,
                "n_neurons": self._n_neurons,
                "drop_rate": self.drop_rate,
            }
        )
        return config

    @property
    def g_forecast_neurons(self):
        """Return the dimension of the gf layer."""
        return self._g_forecast_neurons

    @property
    def g_backcast_neurons(self):
        """Return the dimension of the gb layer."""
        return self._g_backcast_neurons

    def _val___init__(
        self, output: None, *args: list, **kwargs: dict
    ) -> None:  # pylint: disable=unused-argument
        greater_or_equal(self.g_forecast_neurons, 0, 'g_forecast_neurons')
        greater_or_equal(self.g_backcast_neurons, 0, 'g_backcast_neurons')


class Stack(Layer):
    """
    TODO: Transform to layer
    A stack is a series of blocks where each block produces two outputs,
    the forecast and the backcast.

    Inside a stack all forecasts are sum up and compose the stack output.
    In the meantime, the backcast is given to the following block.

    Parameters
    ----------
    blocks : Tuple[:class:`autopycoin.models.BaseBlock`]
        Blocks layers. they can be generic, seasonal or trend ones.
        You can also define your own block by subclassing `BaseBlock`.

    Attributes
    ----------
    blocks : Tuple[:class:`autopycoin.models.BaseBlock`]
    is_interpretable : bool
    stack_type : str
    label_width : int
    input_width : int

    Examples
    --------
    >>> from autopycoin.layers import TrendBlock, SeasonalityBlock
    >>> from autopycoin.models import Stack, NBEATS
    >>> from autopycoin.losses import QuantileLossError
    >>> trend_block = TrendBlock(label_width=20,
    ...                          p_degree=2,
    ...                          n_neurons=16,
    ...                          drop_rate=0.1,
    ...                          name="trend_block")
    >>> seasonality_block = SeasonalityBlock(label_width=20,
    ...                                      forecast_periods=[10],
    ...                                      backcast_periods=[20],
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
    N-D tensor with shape: (batch_size, ..., units).
    The most common situation would be a 2D input with shape (batch_size, units).

    output shape:
    N-D tensor with shape: (quantiles, batch_size, ..., units) or (batch_size, ..., units) .
    For instance, for a 2D input with shape (batch_size, units),
    the output would have shape (batch_size, units).
    With a QuantileLossError with 2 quantiles or higher the output would have shape (quantiles, batch_size, units).
    """

    def __init__(self, blocks: Tuple[BaseBlock, ...], **kwargs: dict):

        super().__init__(**kwargs)
        self._blocks = blocks
        self._stack_type = self._set_type()
        self._is_interpretable = self._set_interpretability()

    def call(
        self, inputs: Union[tuple, dict, list, tf.Tensor]
    ) -> Tuple[tf.Tensor, ...]:
        """Call method from tensorflow."""

        outputs = tf.constant(0.0) # init output
        for block in self.blocks:
            # outputs is (quantiles, Batch_size, forecast)
            # reconstructed_inputs is (Batch_size, backcast)
            residual_outputs, reconstructed_inputs = block(inputs)
            inputs = tf.subtract(inputs, reconstructed_inputs)
            # outputs is (quantiles, Batch_size, forecast)
            outputs = tf.add(outputs, residual_outputs)
        return outputs, inputs

    def get_config(self) -> dict:
        """get_config method from tensorflow."""
        config = super().get_config()
        config.update({"blocks": self.blocks})
        return config

    def _set_type(self) -> str:
        """Return the type of the stack."""

        block_type = self.blocks[0].block_type
        for block in self.blocks:
            if block.block_type != block_type:
                return "CustomStack"
        return block_type.replace("Block", "") + "Stack"

    def _set_interpretability(self) -> bool:
        """True if the stack is interpretable else False."""

        interpretable = all([block.is_interpretable for block in self.blocks])
        if interpretable:
            return True
        return False

    @property
    def label_width(self) -> int:
        """Return the label width."""
        return self.blocks[0].label_width

    @property
    def input_width(self) -> int:
        """Return the input width."""
        return self.blocks[0].input_width

    @property
    def blocks(self) -> List[BaseBlock]:
        """Return the list of blocks."""
        return self._blocks

    @property
    def stack_type(self) -> str:
        """Return the type of the stack.
        `CustomStack` if the blocks are all differents."""
        return self._stack_type

    @property
    def is_interpretable(self) -> bool:
        """Return True if the stack is interpretable."""
        return self._is_interpretable

    def __repr__(self):
        return self.stack_type