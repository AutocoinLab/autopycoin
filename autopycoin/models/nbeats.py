"""
N-BEATS implementation
"""

from typing import Callable, Union, Tuple, List, Optional
from typing import List, Optional

import tensorflow as tf

from .training import UnivariateModel
from ..layers import TrendBlock, SeasonalityBlock, GenericBlock, BaseBlock
from ..layers.nbeats_layers import SEASONALITY_TYPE
from .pool import BasePool


class Stack(UnivariateModel):
    """
    A stack is a series of blocks where each block produces two outputs,
    the forecast and the backcast.

    Inside a stack all forecasts are sum up and compose the stack output.
    In the meantime, the backcast is given to the following block.

    Parameters
    ----------
    blocks : tuple[:class:`autopycoin.models.BaseBlock`]
        Blocks layers. they can be generic, seasonal or trend ones.
        You can also define your own block by subclassing `BaseBlock`.

    Attributes
    ----------
    blocks : tuple[:class:`autopycoin.models.BaseBlock`]
    label_width : int
    input_width : int
    is_interpretable : bool
    stack_type : str

    Examples
    --------
    >>> from autopycoin.layers import TrendBlock, SeasonalityBlock
    >>> from autopycoin.models import Stack, NBEATS
    >>> from autopycoin.losses import QuantileLossError
    ...
    >>> trend_block = TrendBlock(label_width=20,
    ...                          p_degree=2,
    ...                          n_neurons=16,
    ...                          drop_rate=0.1,
    ...                          name="trend_block")
    ...
    >>> seasonality_block = SeasonalityBlock(label_width=20,
    ...                                      forecast_periods=[10],
    ...                                      backcast_periods=[20],
    ...                                      forecast_fourier_order=[10],
    ...                                      backcast_fourier_order=[20],
    ...                                      n_neurons=15,
    ...                                      drop_rate=0.1,
    ...                                      name="seasonality_block")
    ...
    ... # blocks creation
    >>> trend_blocks = [trend_block for _ in range(3)]
    >>> seasonality_blocks = [seasonality_block for _ in range(3)]
    ...
    ... # Stacks creation
    >>> trend_stacks = Stack(trend_blocks, name="trend_stack")
    >>> seasonality_stacks = Stack(seasonality_blocks, name="seasonality_stack")
    ...
    ... # model definition and compiling
    >>> model = NBEATS([trend_stacks, seasonality_stacks], name="interpretable_NBEATS")
    >>> model.compile(loss=QuantileLossError(quantiles=[0.5]))

    Notes
    -----
    input shape:
    N-D tensor with shape: (..., batch_size, time step).
    The most common situation would be a 2D input with shape (batch_size, time step).

    output shape:
    N-D tensor with shape: (..., batch_size, units).
    For instance, for a 2D input with shape (batch_size, units),
    the output would have shape (batch_size, units).
    With a QuantileLossError with 2 quantiles or higher the output would have shape (quantiles, batch_size, units).
    If you add 2 variables, the output would have shape (variables, quantiles, batch_size, units).
    """

    def __init__(
        self,
        blocks: Tuple[BaseBlock, ...],
        apply_quantiles_transpose: bool = False,
        apply_multivariate_transpose: bool = False,
        *args: list,
        **kwargs: dict,
    ):

        super().__init__(
            apply_quantiles_transpose=apply_quantiles_transpose,
            apply_multivariate_transpose=apply_multivariate_transpose,
            *args,
            **kwargs,
        )
        self._blocks = blocks
        self._stack_type = self._set_type()
        self._is_interpretable = self._set_interpretability()

    def call(
        self, inputs: Union[tuple, dict, list, tf.Tensor], **kwargs: dict
    ) -> Tuple[tf.Tensor, ...]:
        """Call method from tensorflow."""

        outputs = tf.constant(0.0)  # init output
        for block in self.blocks:
            # outputs is (quantiles, Batch_size, forecast)
            # reconstructed_inputs is (Batch_size, backcast)
            reconstructed_inputs, residual_outputs = block(inputs)
            inputs = tf.subtract(inputs, reconstructed_inputs)
            # outputs is (quantiles, Batch_size, forecast)
            outputs = tf.add(outputs, residual_outputs)
        return inputs, outputs

    def get_config(self) -> dict:
        """See tensorflow documentation."""

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


class NBEATS(UnivariateModel):
    """
    Tensorflow model defining the N-BEATS architecture.

    N-BEATS is a univariate model, see :class:`autopycoin.models.UnivariateModel` for more information.
    Its strong advantage resides in its structure which allows us to extract the trend and the seasonality of
    temporal series. They are available from the attributes `seasonality` and `trend`.
    This is an unofficial implementation of the paper https://arxiv.org/abs/1905.10437.

    Parameters
    ----------
    stacks : tuple[:class:`autopycoin.models.Stack`]
             Stacks can be created from :class:`autopycoin.models.TrendBlock`,
             :class:`autopycoin.models.SeasonalityBlock` or :class:`autopycoin.models.GenericBlock`.
             See stack documentation for more details.

    Attributes
    ----------
    stacks : tuple[`Tensor`]
    seasonality : `Tensor`
        Seasonality component of the output.
    trend : `Tensor`
        Trend component of the output.
    stack_outputs : `Tensor`
    is_interpretable : bool
    nbeats_type : str
    label_width : int
    input_width : int

    Examples
    --------
    >>> from autopycoin.layers import TrendBlock, SeasonalityBlock
    >>> from autopycoin.models import Stack, NBEATS
    >>> from autopycoin.losses import QuantileLossError
    >>> from autopycoin.data import random_ts
    >>> from autopycoin.dataset import WindowGenerator
    >>> import tensorflow as tf
    >>> import pandas as pd
    ...
    >>> data = random_ts(n_steps=1000,
    ...                  trend_degree=2,
    ...                  periods=[10],
    ...                  fourier_orders=[10],
    ...                  trend_mean=0,
    ...                  trend_std=1,
    ...                  seasonality_mean=0,
    ...                  seasonality_std=1,
    ...                  batch_size=1,
    ...                  n_variables=1,
    ...                  noise=True,
    ...                  seed=42)
    >>> data = pd.DataFrame(data[0].numpy(), columns=['test'])
    ...
    >>> w = WindowGenerator(
    ...        input_width=20,
    ...        label_width=10,
    ...        shift=10,
    ...        test_size=50,
    ...        valid_size=10,
    ...        flat=True,
    ...        batch_size=32,
    ...        preprocessing=lambda x,y: (x, (x, y))
    ...     )
    ...
    >>> w = w.from_array(data=data,
    ...        input_columns=['test'],
    ...        label_columns=['test'])
    >>>
    >>> trend_block = TrendBlock(label_width=w.label_width,
    ...                          p_degree=2,
    ...                          n_neurons=16,
    ...                          drop_rate=0.1,
    ...                          name="trend_block")
    >>>
    >>> seasonality_block = SeasonalityBlock(label_width=w.label_width,
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
    >>> model.compile(loss=QuantileLossError(quantiles=[0.5]))
    >>> history = model.fit(w.train, verbose=0)

    Notes
    -----
    NBEATS supports the estimation of aleotoric and epistemic errors with:

    - Aleotoric interval : :class:`autopycoin.loss.QuantileLossError`
    - Epistemic interval : MCDropout

    You can use :class:`autopycoin.loss.QuantileLossError` as loss error to estimate the
    aleotoric error. Also, run multiple times a prediction with `drop_date` > 0 to estimate
    the epistemic error.

    *Input shape*
    N-D tensor with shape: (batch_size, time step, variables) or (batch_size, time step).
    The most common situation would be a 2D input with shape (batch_size, time step).

    *Output shape*
    Two N-D tensor with shape: (batch_size, time step, variables, quantiles) or (batch_size, time step, quantiles)
    or (batch_size, time step).
    For instance, for a 2D input with shape (batch_size, units),
    the output would have shape (batch_size, units).
    With a QuantileLossError with 2 quantiles or higher the output
    would have shape (batch_size, units, quantiles).
    With a multivariate inputs the output
    would have shape (batch_size, units, variates, quantiles).
    """

    def __init__(self, stacks: Tuple[Stack, ...], *args: list, **kwargs: dict):

        super().__init__(*args, **kwargs)

        # Stacks where blocks are defined
        self._stacks = stacks

        self._is_interpretable = self._set_interpretability()
        self._nbeats_type = self._set_type()

    def build(
        self, input_shape: Union[tf.TensorShape, Tuple[tf.TensorShape, ...]]
    ) -> None:
        """See tensorflow documentation."""

        if isinstance(input_shape, tuple):
            input_shape = input_shape[0]

        super().build(input_shape)

    def compile(
        self,
        optimizer="rmsprop",
        loss=None,
        metrics=None,
        loss_weights=[0.0, 1.0],
        weighted_metrics=None,
        run_eagerly=None,
        steps_per_execution=None,
        **kwargs,
    ) -> None:

        super().compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            loss_weights=loss_weights,
            weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly,
            steps_per_execution=steps_per_execution,
            **kwargs,
        )

    def call(
        self, inputs: Union[tuple, dict, list, tf.Tensor], **kwargs: dict
    ) -> tf.Tensor:
        """Call method from tensorflow."""

        if isinstance(inputs, (tuple, list)):
            inputs = inputs[0]

        residual_inputs = tf.identity(inputs)
        outputs = tf.constant(0.0)
        for stack in self.stacks:
            # outputs_residual is (quantiles, Batch_size, forecast)
            # inputs is (Batch_size, backcast)
            residual_inputs, residual_outputs = stack(residual_inputs)
            # outputs is (quantiles, Batch_size, forecast)
            outputs = tf.math.add(outputs, residual_outputs)

        reconstructed_inputs = inputs - residual_inputs

        return reconstructed_inputs, outputs

    def seasonality(self, data: tf.Tensor) -> tf.Tensor:
        """
        Based on the paper, the seasonality is available if
        the previous stacks are composed by trend blocks.
        Else, it doesn't correspond to seasonality.

        Parameters
        ----------
        data : `Tensor`
            input data.

        Returns
        -------
        seasonality : `Tensor`
            Same shape as call inputs (see notes).

        Raises
        ------
        AttributeError
            If all previous stacks are not composed
            by trend blocks then an error is raised.
        AssertionError
            if no `SeasonalityStack` are defined.
        """

        msg_error = f"""The first stack has to be a `TrendStack`,
        hence seasonality doesn't exists . Got {self.stacks}."""

        for idx, stack in enumerate(self.stacks):
            if stack.stack_type != "TrendStack" and idx == 0:
                raise AttributeError(msg_error)
            elif stack.stack_type == "SeasonalityStack":
                start = idx
            elif stack.stack_type == "TrendStack":
                continue
            else:
                break
        if "start" not in locals():
            raise AttributeError(f"No `SeasonalityStack` defined. Got {self.stacks}")

        for stack in self.stacks[:start]:
            data, _ = stack(data)
        for stack in self.stacks[start : idx + 1]:
            data, residual_seas = stack(data)
            if "seasonality" not in locals():
                seasonality = residual_seas
            else:
                seasonality += residual_seas
        return seasonality

    def trend(self, data: tf.Tensor) -> tf.Tensor:
        """
        The trend component of the output.

        Returns
        -------
        trend : `Tensor`
            Same shape as call inputs (see notes).

        Raises
        ------
        AttributeError
            Raises an error if previous stacks are not `TrendBlock`.
        """

        for idx, stack in enumerate(self.stacks):
            if stack.stack_type != "TrendStack":
                break
        msg = f"""No `TrendStack` defined. Got {self.stacks}.
                `TrendStack` has to be defined as first stack."""
        if idx == 0:
            raise AttributeError(msg)

        for stack in self.stacks[: idx + 1]:
            data, residual_trend = stack(data)
            if "trend" not in locals():
                trend = residual_trend
            else:
                trend += residual_trend
        return trend

    def get_config(self) -> dict:
        """Get_config from tensorflow."""

        return {"stacks": self.stacks}

    def _set_interpretability(self) -> bool:
        """check if interpretable or not."""

        return all(stack.is_interpretable for stack in self.stacks)

    def _set_type(self):
        """Defines the type of Nbeats."""

        if self.is_interpretable:
            return "InterpretableNbeats"
        return "Nbeats"

    @property
    def label_width(self) -> int:
        """Return the label width."""

        return self.stacks[0].label_width

    @property
    def input_width(self) -> int:
        """Return the input width."""

        return self.input_width

    @property
    def stacks(self) -> int:
        """Return the input width."""

        return self._stacks

    @property
    def is_interpretable(self) -> bool:
        """Return True if the model is interpretable."""

        return self._is_interpretable

    @property
    def nbeats_type(self) -> str:
        """Return the Nbeats type."""

        return self._nbeats_type

    def __repr__(self):
        return self._nbeats_type


NbeatsModelsOptions = Union[
    Union[List[NBEATS], NBEATS], Union[List[Callable], Callable],
]


def create_interpretable_nbeats(
    label_width: int,
    forecast_periods: SEASONALITY_TYPE = None,
    backcast_periods: SEASONALITY_TYPE = None,
    forecast_fourier_order: SEASONALITY_TYPE = None,
    backcast_fourier_order: SEASONALITY_TYPE = None,
    p_degree: int = 1,
    trend_n_neurons: int = 252,
    seasonality_n_neurons: int = 2048,
    drop_rate: float = 0.0,
    share: bool = True,
    name: str = "interpretable_NBEATS",
    **kwargs: dict,
):
    """
    Wrapper which create an interpretable model as described in the original paper.
    Two stacks are created with 3 blocks each. The first entirely composed by trend blocks,
    The second entirely composed by seasonality blocks.

    Within the same stack, it is possible to share the weights between blocks.

    Parameters
    ----------
    label_width : int
        Past to rebuild. Usually, label_width = n * input width with n between 1 and 7.
    forecast_periods : Tuple[int, ...]
        Compute the fourier serie period in the forecast equation.
        if a list is provided then all periods are taken.
    backcast_periods : Tuple[int, ...]
        Compute the fourier serie period in the backcast equation.
        if a list is provided then all periods are taken.
    forecast_fourier_order : Tuple[int, ...]
        Compute the fourier order. each order element refers to its respective period.
    backcast_fourier_order : Tuple[int, ...]
        Compute the fourier order. each order element refers to its respective back period.
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
    >>> model = create_interpretable_nbeats(label_width=3,
    ...                                     forecast_periods=[2],
    ...                                     backcast_periods=[3],
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
            label_width=label_width,
            p_degree=p_degree,
            n_neurons=trend_n_neurons,
            drop_rate=drop_rate,
            name="trend_block",
        )

        seasonality_block = SeasonalityBlock(
            label_width=label_width,
            forecast_periods=forecast_periods,
            backcast_periods=backcast_periods,
            forecast_fourier_order=forecast_fourier_order,
            backcast_fourier_order=backcast_fourier_order,
            n_neurons=seasonality_n_neurons,
            drop_rate=drop_rate,
            name="seasonality_block",
        )

        trend_blocks = [trend_block for _ in range(3)]
        seasonality_blocks = [seasonality_block for _ in range(3)]
    else:
        trend_blocks = [
            TrendBlock(
                label_width=label_width,
                p_degree=p_degree,
                n_neurons=trend_n_neurons,
                drop_rate=drop_rate,
                name="trend_block",
            )
            for _ in range(3)
        ]
        seasonality_blocks = [
            SeasonalityBlock(
                label_width=label_width,
                forecast_periods=forecast_periods,
                backcast_periods=backcast_periods,
                forecast_fourier_order=forecast_fourier_order,
                backcast_fourier_order=backcast_fourier_order,
                n_neurons=seasonality_n_neurons,
                drop_rate=drop_rate,
                name="seasonality_block",
            )
            for _ in range(3)
        ]

    trend_stacks = Stack(trend_blocks, name="trend_stack")
    seasonality_stacks = Stack(seasonality_blocks, name="seasonality_stack")
    model = NBEATS([trend_stacks, seasonality_stacks], name=name, **kwargs)

    return model


def create_generic_nbeats(
    label_width: int,
    g_forecast_neurons: int = 524,
    g_backcast_neurons: int = 524,
    n_neurons: int = 524,
    n_blocks: int = 1,
    n_stacks: int = 30,
    drop_rate: float = 0.0,
    share: bool = False,
    name: str = "generic_NBEATS",
    **kwargs: dict,
):
    """
    Wrapper which create a generic model as described in the original paper.

    In the same stack, it is possible to share the weights between blocks.

    Parameters
    ----------
    label_width : int
        Past to rebuild. Usually, label_width = n * input width with n between 1 and 7.
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
    >>> model = create_generic_nbeats(label_width=3,
    ...                               g_forecast_neurons=16,
    ...                               g_backcast_neurons=16,
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
                label_width=label_width,
                g_forecast_neurons=g_forecast_neurons,
                g_backcast_neurons=g_backcast_neurons,
                n_neurons=n_neurons,
                drop_rate=drop_rate,
                name="generic_block",
            )

            generic_blocks = [generic_block for _ in range(n_blocks)]
            generic_stacks.append(Stack(generic_blocks, name="generic_stack"))

    else:
        for _ in range(n_stacks):
            generic_blocks = [
                GenericBlock(
                    label_width=label_width,
                    g_forecast_neurons=g_forecast_neurons,
                    g_backcast_neurons=g_backcast_neurons,
                    n_neurons=n_neurons,
                    drop_rate=drop_rate,
                    name="generic_block",
                )
                for _ in range(n_blocks)
            ]

            generic_stacks.append(Stack(generic_blocks, name="generic_stack"))

    model = NBEATS(generic_stacks, name=name, **kwargs)
    return model


# TODO: finish doc and unit testing.
class PoolNBEATS(BasePool):
    """
    Tensorflow model defining a pool of N-BEATS models.

    As described in the paper https://arxiv.org/abs/1905.10437, the state-of-the-art results
    are reached with a bagging method of N-BEATS models including interpretable and generic ones.

    The aggregation function is used in predict `method` if it is possible, i.e when the outputs shape are not differents.
    As the reconstructed inputs are masked randomly the aggregation is not perfomed on them.

    Fore more information about poll model see :class:`autopycoin.models.Pool`.

    Parameters
    ----------
    label_width : int
        Width of the targets.
        It can be not defined if `nbeats_model` is a list of NBEATS instances.
        Default to None.
    n_models : int
        Number of models inside the pool.
        The minimum value according to the paper to get SOTA results is 18.
        If NBEATS instances are provided then n_models is not used.
        Default to 18.
    nbeats_models : list[callable] or list[NBEATS]
        A list of callables which create a NBEATS model or a list of :class:`autopycoin.models.NBEATS` instances.
        If None then use a mix of generic and interpretable NBEATs model.
        Default to None.
    fn_agg : Callable
        Function of aggregation which takes an parameter axis.
        It aggregates the models outputs. Default to mean.
    seed: int
        Used in combination with tf.random.set_seed to create a
        reproducible sequence of tensors across multiple calls.

    Returns
    -------
    outputs : Tuple[Tuple[`Tensor` | QuantileTensor | UnivariateTensor], Tuple[`Tensor` | QuantileTensor | UnivariateTensor]]
        Return the reconstructed inputs and inferred outputs as tuple (reconstructed inputs, outputs).
        Reconstructed inputs is a tuple of tensors as the mask is not the same through models.
        Outputs can be a tuple of tensors or an aggregated tensor if the prediction is used through `predict` method.

    Attributes
    ----------
    see :class:`autopycoin.models.Pool`

    Examples
    --------
    >>> from autopycoin.data import random_ts
    >>> from autopycoin.models import PoolNBEATS, create_interpretable_nbeats
    >>> from autopycoin.dataset import WindowGenerator
    >>> import tensorflow as tf
    >>> import pandas as pd
    ...
    >>> data = random_ts(n_steps=1000,
    ...                  trend_degree=2,
    ...                  periods=[10],
    ...                  fourier_orders=[10],
    ...                  trend_mean=0,
    ...                  trend_std=1,
    ...                  seasonality_mean=0,
    ...                  seasonality_std=1,
    ...                  batch_size=1,
    ...                  n_variables=1,
    ...                  noise=True,
    ...                  seed=42)
    >>> data = pd.DataFrame(data[0].numpy(), columns=['test'])
    ...
    >>> w = WindowGenerator(
    ...        input_width=70,
    ...        label_width=10,
    ...        shift=10,
    ...        test_size=50,
    ...        valid_size=10,
    ...        flat=True,
    ...        batch_size=32,
    ...        preprocessing=lambda x,y: (x, (x, y))
    ...     )
    ...
    >>> w = w.from_array(data=data,
    ...        input_columns=['test'],
    ...        label_columns=['test'])
    ...
    >>> model = PoolNBEATS(
    ...             label_width=10,
    ...             n_models=2,
    ...             nbeats_models=create_interpretable_nbeats,
    ...             )
    >>> model.compile(tf.keras.optimizers.Adam(
    ...    learning_rate=0.015, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True,
    ...    name='Adam'), loss=['mse', 'mae', 'mape'], metrics=['mae'])
    >>> history = model.fit(w.train, validation_data=w.valid, epochs=1, verbose=0)
    >>> model.predict(w.test.take(1))[1].shape
    (32, 10)

    Notes
    -----
    PoolNBEATS is just a wrapper around nbeats models hence you can use epistemic loss error
    or multivariates inputs.
    This class only applies mask to its inputs.

    *Input shape*
    N-D tensor with shape: (batch_size, time step, variables) or (batch_size, time step).
    The most common situation would be a 2D input with shape (batch_size, time step).

    *Output shape*
    n N-D tensors with shape: (batch_size, time step, variables, quantiles) or (batch_size, time step, quantiles)
    or (batch_size, time step) with n the number of models generated randomly or registered in the constructor.

    For instance, for a 2D input with shape (batch_size, units) and three models,
    the output would have shape 
    (((batch_size, units), (batch_size, units), (batch_size, units)), ((batch_size, units), (batch_size, units), (batch_size, units)))
    if call is used else
    (((batch_size, units), (batch_size, units), (batch_size, units)), (batch_size, units)) if predict is used.

    The outputs tensors can be aggregated during `predict` method only if all tensors are similar in shape.
    """

    def __init__(
        self,
        label_width: int = None,
        n_models: int = 18,
        nbeats_models: Union[None, NbeatsModelsOptions] = [
            create_interpretable_nbeats,
            create_generic_nbeats,
        ],
        fn_agg: Callable = tf.reduce_mean,
        seed: Optional[int] = None,
        **kwargs: dict,
    ):

        super().__init__(
            label_width=label_width,
            n_models=n_models,
            models=nbeats_models,
            fn_agg=fn_agg,
            seed=seed,
            **kwargs,
        )

    def checks(self, nbeats_models: List[NBEATS]) -> None:
        """Check if `label_width` are equals through models instances."""

        labels_width = [model.label_width for model in nbeats_models]
        # If `label_width` is defined in the init then use it to check models else use the first model value.
        self._label_width = self.label_width or labels_width[0]
        assert all([label_width == self.label_width for label_width in labels_width]), (
            f"`label_width` parameter has to be identical through models and against the value given in the init method. "
            f"Got {labels_width} for models and `label_width` = {self.label_width}"
        )

    def build(
        self, input_shape: Union[tf.TensorShape, Tuple[tf.TensorShape, ...]]
    ) -> None:
        """See tensorflow documentation."""

        # Defines masks
        mask = tf.random.uniform(
            (self.n_models,),
            minval=0,
            maxval=int(input_shape[1] / self.label_width) or 1,
            dtype=tf.int32,
            seed=self.seed,
        )
        self._mask = input_shape[1] - (mask * self.label_width)

        super().build(input_shape)

    def call(
        self, inputs: Union[tuple, dict, list, tf.Tensor], **kwargs: dict
    ) -> tf.Tensor:
        """Call method from tensorflow Model.
        
        Make prediction with every models generated during the constructor method.
        """

        output_fn = lambda idx: self.models[idx](inputs[:, -self._mask[idx] :])
        outputs = tf.nest.map_structure(
            output_fn, [idx for idx in range(self.n_models)]
        )
        return outputs

    def preprocessing_x(
        self,
        x: Union[None, Union[Union[tf.Tensor, tf.data.Dataset], Tuple[tf.Tensor, ...]]],
    ) -> Union[Tuple[None, None], Tuple[Callable, tuple]]:
        "Apply mask inside `train_step`"

        # Build masks from PoolNBEATS `build` method
        self._maybe_build(x)

        masked_x = None
        if x is not None:
            masked_x = [x[:, -self._mask[idx] :] for idx in range(self.n_models)]

        return masked_x

    def preprocessing_y(
        self,
        y: Union[None, Union[Union[tf.Tensor, tf.data.Dataset], Tuple[tf.Tensor, ...]]],
    ) -> Union[Tuple[None, None], Tuple[Callable, tuple]]:
        "Apply mask inside `train_step`, `test_step`"

        masked_y = None
        if y is not None:
            masked_y = [
                (y[0][:, -self._mask[idx] :], y[1]) for idx in range(self.n_models)
            ]

        return masked_y

    def postprocessing_y(self, y):
        "Apply mask inside `predict_step`"

        inputs_reconstucted = [outputs[0] for outputs in y]
        y = [outputs[1] for outputs in y]

        if any(outputs.quantiles for outputs in y):
            return inputs_reconstucted, y

        return inputs_reconstucted, self.fn_agg(y, axis=0)
