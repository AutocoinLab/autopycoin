"""
N-BEATS implementation
"""

from typing import Callable, Union, Tuple, List
from numpy.random import randint
import keras_tuner as kt

import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper

from ..utils.data_utils import convert_to_list
from .training import Model
from ..baseclass import AutopycoinBaseClass
from ..layers import TrendBlock, SeasonalityBlock, GenericBlock, UniVariate, BaseBlock
from ..layers.nbeats_layers import SEASONALITY_TYPE


class Stack(Model):
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

    def __init__(self, blocks: Tuple[BaseBlock, ...], **kwargs: dict):

        super().__init__(**kwargs)
        self._blocks = blocks
        self._stack_type = self._set_type()
        self._is_interpretable = self._set_interpretability()

    def call(
        self, inputs: Union[tuple, dict, list, tf.Tensor]
    ) -> Tuple[tf.Tensor, ...]:
        """Call method from tensorflow."""

        outputs = tf.constant(0.0)  # init output
        for block in self.blocks:
            # outputs is (quantiles, Batch_size, forecast)
            # reconstructed_inputs is (Batch_size, backcast)
            residual_outputs, reconstructed_inputs = block(inputs)
            inputs = tf.subtract(inputs, reconstructed_inputs)
            # outputs is (quantiles, Batch_size, forecast)
            outputs = tf.add(outputs, residual_outputs)
        return outputs, inputs

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


class NBEATS(Model, AutopycoinBaseClass):
    """
    Tensorflow model defining the N-BEATS architecture.

    N-BEATS is a univariate model. Its strong advantage
    resides in its structure which allows us to extract the trend and the seasonality of
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
    >>> trend_block = TrendBlock(label_width=20,
    ...                          p_degree=2,
    ...                          n_neurons=16,
    ...                          drop_rate=0.1,
    ...                          name="trend_block")
    >>>
    >>> seasonality_block = SeasonalityBlock(label_width=20,
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
    N-D tensor with shape: (batch_size, time step, variables, quantiles) or (batch_size, time step, quantiles)
    or (batch_size, time step).
    For instance, for a 2D input with shape (batch_size, units),
    the output would have shape (batch_size, units).
    With a QuantileLossError with 2 quantiles or higher the output
    would have shape (quantiles, batch_size, units).
    """

    def __init__(self, stacks: Tuple[Stack, ...], **kwargs: dict):

        super().__init__(**kwargs)

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

        input_shape = tf.TensorShape(input_shape)
        # multi univariate inputs
        self.strategy_input = UniVariate(
            last_to_first=True, is_multivariate=bool(input_shape.rank > 2)
        )
        self.strategy_output = UniVariate(
            last_to_first=False, is_multivariate=bool(input_shape.rank > 2)
        )

        super().build(input_shape)

    def call(self, inputs: Union[tuple, dict, list, tf.Tensor]) -> tf.Tensor:
        """Call method from tensorflow."""

        inputs = self.strategy_input(inputs)
        outputs = tf.constant(0.0)
        for stack in self.stacks:
            # outputs_residual is (quantiles, Batch_size, forecast)
            # inputs is (Batch_size, backcast)
            residual_outputs, inputs = stack(inputs)
            # outputs is (quantiles, Batch_size, forecast)
            outputs = tf.math.add(outputs, residual_outputs)
        return self.strategy_output(outputs)  # , inputs - reconstructed_inputs

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
            _, data = stack(data)
        for stack in self.stacks[start : idx + 1]:
            residual_seas, data = stack(data)
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
            residual_trend, data = stack(data)
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


# TODO: finish doc and unit testing.
class PoolNBEATS(Model, AutopycoinBaseClass):
    """
    Tensorflow model defining a pool of N-BEATS models.

    As described in the paper https://arxiv.org/abs/1905.10437, the state-of-the-art results
    are reached with a bagging method of N-BEATS models including interpretable and generic ones.

    Parameters
    ----------
    n_models : int
        Number of models inside the pool.
    nbeats_models : list[callable]
        A list of callables which create a NBEATS model.
    losses : list[str or `tf.keras.losses.Loss`]
        List of losses used to train the models.
    fn_agg : Callable
        Function of aggregation which takes an parameter axis.
        It aggregates the models outputs. Default to mean.
    seed: int
        Used in combination with tf.random.set_seed to create a
        reproducible sequence of tensors across multiple calls.

    Attributes
    ----------

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
    ...     )
    ...
    >>> w = w.from_array(data=data,
    ...        input_columns=['test'],
    ...        label_columns=['test'])
    ...
    >>> nbeats = lambda : create_interpretable_nbeats(
    ...                 label_width=10,
    ...                 forecast_periods=[10],
    ...                 backcast_periods=[20],
    ...                 forecast_fourier_order=[10],
    ...                 backcast_fourier_order=[20],
    ...                 p_degree=2,
    ...                 trend_n_neurons=32,
    ...                 seasonality_n_neurons=32,
    ...                 drop_rate=0.1,
    ...                 share=True
    ...          )
    ...
    >>> model = PoolNBEATS(
    ...             n_models=10,
    ...             nbeats_models=nbeats,
    ...             losses=['mse', 'mae', 'mape'])
    >>> model.compile(tf.keras.optimizers.Adam(
    ...    learning_rate=0.015, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True,
    ...    name='Adam'), loss=model.get_pool_losses(), metrics=['mae'])
    >>> history = model.fit(w.train, validation_data=w.valid, epochs=1, verbose=0)
    >>> model.predict(w.test.take(1)).shape
    TensorShape([32, 10])

    Notes
    -----
    """

    def __init__(
        self,
        n_models: int,
        nbeats_models: Union[
            Union[List[NBEATS], NBEATS], Union[Callable, List[Callable]],
        ],
        losses: List[Union[str, Union[tf.keras.losses.Loss, LossFunctionWrapper]]],
        fn_agg: Callable = tf.reduce_mean,
        seed: Union[None, int] = None,
        **kwargs: dict,
    ):

        super().__init__(**kwargs)

        # Reproducible instance
        if seed is not None:
            tf.random.set_seed(seed)

        self._n_models = n_models

        # nbeats init
        self._init_nbeats(convert_to_list(nbeats_models), losses)

        # Layer definition and function to aggregate the multiple outputs
        self._fn_agg = fn_agg

    def _init_nbeats(self, nbeats_models: Union[List[Callable], List[NBEATS]], losses: List[Union[str, Union[tf.keras.losses.Loss, LossFunctionWrapper]]]) -> None:
        """Initialize nbeats models."""

        self._nbeats = nbeats_models
        if not isinstance(nbeats_models[0], NBEATS) and callable(nbeats_models[0]):
            self._nbeats = self._init_nbeats_from_callable(nbeats_models)
        else:
            self._n_models = len(nbeats_models)
        self._init_pool_losses(losses)
        self._check_label_width(self._nbeats)
        self._set_quantiles()

    def _init_pool_losses(self, losses: List[Union[str, Union[tf.keras.losses.Loss, LossFunctionWrapper]]]):
        # Init pool of losses by picking randomly loss in losses list.
        if self.n_models - len(losses) < 0:
            raise ValueError('The number of models has to be equal or larger than the number of losses.'
                             f' Got n_models={self.n_models} and losses={len(losses)}')
        losses_idx = randint(0, len(losses), size=self.n_models - len(losses))
        self._pool_losses = losses + [losses[idx] for idx in losses_idx]

    def _init_nbeats_from_callable(self, nbeats_models: List[Callable]) -> None:
        """Initialize nbeats models from callable."""

        nbeats = []
        # Init pool of models by picking randomly model in nbeats_models list.
        for nbeats_idx in randint(0, len(nbeats_models), size=self.n_models):
            model = nbeats_models[nbeats_idx]()
            nbeats.append(model)
        return nbeats

    def _check_label_width(
        self, nbeats_models: Union[List[Callable], List[NBEATS]]
    ) -> None:
        """Check if `label_width` are equals through models."""

        labels_width = [model.label_width for model in nbeats_models]
        theoric_sum = len(labels_width) * labels_width[0]
        assert sum(labels_width) - theoric_sum == 0, (
            f"`label_width` parameter has to be identical through models."
            f"Got {labels_width}"
        )
        self._label_width = labels_width[0]

    def _set_quantiles(self) -> None:
        """Set quantiles if a quantile loss is compiled."""

        for idx, loss in enumerate(self._pool_losses):
            if hasattr(loss, "quantiles"):
                self._nbeats[idx]._set_quantiles(loss.quantiles)

    def build(
        self, input_shape: Union[tf.TensorShape, Tuple[tf.TensorShape, ...]]
    ) -> None:
        """See tensorflow documentation."""

        if isinstance(input_shape, tuple):
            input_shape = input_shape[0]

        # Defines masks
        mask = tf.random.uniform(
            (self.n_models,),
            minval=0,
            maxval=int(input_shape[1] / self.label_width) or 1,
            dtype=tf.int32,
        )
        self._mask = input_shape[1] - (mask * self.label_width)

        super().build(input_shape)

    def call(self, inputs: Union[tuple, dict, list, tf.Tensor]) -> tf.Tensor:
        """Call method from tensorflow Model."""

        outputs = []
        for idx, model in enumerate(self.nbeats):
            inputs_masked = inputs[:, -self._mask[idx] :]
            outputs.append(model(inputs_masked))
            
        return outputs

    def predict(self, *args: list, **kwargs: dict) -> tf.Tensor:
        """Reduce the n outputs to a single output tensor by mean operation."""

        outputs = super().predict(*args, **kwargs)
        return self._fn_agg(outputs, axis=0)


    def get_pool_losses(self) -> list:
        """Return the pool losses."""

        return self._pool_losses

    def reset_pool_losses(
        self,
        losses: List[Union[str, tf.keras.losses.Loss]],
        seed: Union[int, None] = None,
    ) -> None:
        """Reset the pool losses based on the given losses."""

        if seed is not None:
            tf.random.set_seed(seed)
        self._init_pool_losses(losses)
        self._set_quantiles()

    @property
    def label_width(self) -> int:
        """Return the `label_width` parameter."""

        return self._label_width

    @property
    def n_models(self) -> int:
        """Return the `n_models` parameter."""

        return self._n_models

    @property
    def nbeats(self) -> list:
        """Return the nbeats pool."""

        return self._nbeats


def create_interpretable_nbeats(
    label_width: int,
    forecast_periods: SEASONALITY_TYPE = None,
    backcast_periods: SEASONALITY_TYPE = None,
    forecast_fourier_order: SEASONALITY_TYPE = None,
    backcast_fourier_order: SEASONALITY_TYPE = None,
    p_degree: int = 1,
    trend_n_neurons: int = 16,
    seasonality_n_neurons: int = 16,
    drop_rate: float = 0.0,
    share: bool = True,
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
    model = NBEATS(
        [trend_stacks, seasonality_stacks], name="interpretable_NBEATS", **kwargs
    )

    return model


def create_generic_nbeats(
    label_width: int,
    g_forecast_neurons: int,
    g_backcast_neurons: int,
    n_neurons: int,
    n_blocks: int,
    n_stacks: int,
    drop_rate: float = 0.0,
    share: bool = True,
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

    model = NBEATS(generic_stacks, name="generic_NBEATS", **kwargs)
    return model


# TODO: unit test
def interpretable_nbeats_builder(label_width: int, **kwargs: dict) -> Callable:
    """
    It defines model and hyperparameters to take into account during the optimization.
    We set main parameters but you can overread this function to customize
    your parameters selection.
    """

    def model_builder(hp) -> NBEATS:
        hp_n_neurons_trend = hp.Int(
            "neurons_trend", min_value=20, max_value=520, step=20
        )
        hp_n_neurons_seas = hp.Int("neurons_seas", min_value=20, max_value=520, step=20)
        hp_periods = kwargs.get("periods", None)
        hp_periods = hp.Choice("periods", hp_periods) if hp_periods else hp_periods
        hp_backcast_periods = kwargs.get("backcast_periods", None)
        hp_backcast_periods = (
            hp.Choice("backcast_periods", hp_backcast_periods)
            if hp_backcast_periods
            else hp_backcast_periods
        )
        hp_forecast_fourier_order = kwargs.get("forecast_fourier_order", None)
        hp_forecast_fourier_order = (
            hp.Choice("forecast_fourier_order", hp_forecast_fourier_order)
            if hp_forecast_fourier_order
            else hp_forecast_fourier_order
        )
        hp_backcast_fourier_order = kwargs.get("backcast_fourier_order", None)
        hp_backcast_fourier_order = (
            hp.Choice("backcast_fourier_order", hp_backcast_fourier_order)
            if hp_backcast_fourier_order
            else hp_backcast_fourier_order
        )
        hp_share = hp.Boolean("share")
        hp_p_degree = hp.Int("p_degree", min_value=0, max_value=3, step=1)
        loss_list = kwargs.get("loss", ["mse"])
        loss_idx = hp.Choice("loss", range(len(loss_list)))
        optimizer_list = kwargs.get(
            "optimizer", [tf.keras.optimizers.Adam(learning_rate=0.001)]
        )
        optimizer_idx = hp.Choice("optimizer", range(len(optimizer_list)))

        model = create_interpretable_nbeats(
            label_width=label_width,
            p_degree=hp_p_degree,
            forecast_periods=hp_periods,
            backcast_periods=hp_backcast_periods,
            forecast_fourier_order=hp_forecast_fourier_order,
            backcast_fourier_order=hp_backcast_fourier_order,
            trend_n_neurons=hp_n_neurons_trend,
            seasonality_n_neurons=hp_n_neurons_seas,
            share=hp_share,
        )

        model.compile(
            loss=loss_list[loss_idx],
            optimizer=optimizer_list[optimizer_idx],
            metrics=[tf.keras.metrics.MeanAbsoluteError()],
        )

        return model

    return model_builder
