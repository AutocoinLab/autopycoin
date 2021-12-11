"""
N-BEATS implementation
"""

from typing import Callable, Union, Tuple, List
from numpy.random import randint, uniform
import tensorflow as tf

# from tensorflow.keras import Model

from .training import Model
from ..baseclass import AutopycoinBaseClass
from ..layers.nbeats_layers import BaseBlock, TrendBlock, SeasonalityBlock, GenericBlock
from ..layers import UniVariate


class Stack(Model, AutopycoinBaseClass):
    """
    A stack is a series of blocks where each block produce two outputs,
    the forecast and the backcast.

    All forecasts are sum up and compose the stack output. In the meantime,
    each backcasts is given to the following block.

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

    Examples
    --------
    >>> from autopycoin.layers import TrendBlock, SeasonalityBlock
    >>> from autopycoin.models import Stack, NBEATS
    >>> from autopycoin.losses import QuantileLossError
    >>> trend_block = TrendBlock(input_width=10,
    ...                          label_width=20,
    ...                          p_degree=2,
    ...                          n_neurons=16,
    ...                          drop_rate=0.1,
    ...                          name="trend_block")
    >>> seasonality_block = SeasonalityBlock(input_width=10,
    ...                                      label_width=20,
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
    N-D tensor with shape: (batch_size, ..., input_dim).
    The most common situation would be a 2D input with shape (batch_size, input_dim).

    output shape:
    N-D tensor with shape: (quantiles, batch_size, ..., units) or (batch_size, ..., units) .
    For instance, for a 2D input with shape (batch_size, input_dim),
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

        outputs = tf.constant(0.0)
        for block in self.blocks:
            # outputs is (quantiles, Batch_size, forecast)
            # inputs_reconstruction is (Batch_size, backcast)
            reconstructed_outputs, reconstructed_inputs = block(inputs)
            inputs = tf.subtract(inputs, reconstructed_inputs)
            # outputs is (quantiles, Batch_size, forecast)
            outputs = tf.add(outputs, reconstructed_outputs)
        return outputs, inputs

    def get_config(self) -> dict:
        """get_config method from tensorflow."""
        config = super().get_config()
        config.update({"blocks": self.blocks})
        return config

    @property
    def input_width(self) -> int:
        """Return the list of blocks."""
        return self.blocks[0].input_width

    @property
    def blocks(self) -> List[BaseBlock]:
        """Return the list of blocks."""
        return self._blocks

    @property
    def stack_type(self) -> str:
        """Return the type of the stack.
        It can be `CustomStack` if the blocks are all differents or
        `...Stack` with `...` the first part of the block types."""
        return self._stack_type

    @property
    def is_interpretable(self) -> bool:
        """Return True if the stack is interpretable."""
        return self._is_interpretable

    def _set_type(self) -> str:
        """Return the type of the stack."""

        block_type = self.blocks[0].block_type
        for block in self.blocks:
            if block.block_type != block_type:
                return "CustomStack"
        return block_type.replace("Block", "") + "Stack"

    def _set_interpretability(self) -> bool:
        """Defines if the stack is interpretable."""

        interpretable = all([block.is_interpretable for block in self.blocks])
        if "Custom" not in self.stack_type and interpretable:
            return True
        return False

    def __repr__(self):
        return self.stack_type

    def val___init__(self, output: tf.Tensor, *args: list, **kwargs: dict
        ) -> None:  # pylint: disable=unused-argument
        # TODO: unit testing
        for block in self.blocks:
            assert block.input_width == self.input_width, ValueError(f"""`input_width` values are not equals along blocks.""")


class NBEATS(Model, AutopycoinBaseClass):
    """
    Tensorflow model defining the N-BEATS architecture.

    N-BEATS is a univariate model. Its strong advantage
    resides in its structure which allows us to extract the trend and the seasonality of
    temporal series. They are available from the attributes `seasonality` and `trend`.
    This is an unofficial implementation of the paper https://arxiv.org/abs/1905.10437.

    Parameters
    ----------
    stacks : Tuple[:class:`autopycoin.models.Stack`]
             Stacks can be created from :class:`autopycoin.models.TrendBlock`,
             :class:`autopycoin.models.SeasonalityBlock` or :class:`autopycoin.models.GenericBlock`.
             See stack documentation for more details.

    Attributes
    ----------
    stacks : Tuple[`Tensor`]
    seasonality : `Tensor`
        Seasonality component of the output.
    trend : `Tensor`
        Trend component of the output.
    stack_outputs : `Tensor`
    is_interpretable : bool
    nbeats_type : str

    Examples
    --------
    >>> from autopycoin.layers import TrendBlock, SeasonalityBlock
    >>> from autopycoin.models import Stack, NBEATS
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
    aleotoric error. Also, run multiple time a prediction with `drop_date` > 0 to estimate
    the epistemic error.

    *Input shape*:
    N-D tensor with shape: (batch_size, ..., input_dim).
    The most common situation would be a 2D input with shape (batch_size, input_dim).

    *Output shape*:
    N-D tensor with shape: (quantiles, batch_size, ..., units) or (batch_size, ..., units) .
    For instance, for a 2D input with shape (batch_size, input_dim),
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
        self.strategy = UniVariate()

    def call(self, inputs: Union[tuple, dict, list, tf.Tensor]) -> tf.Tensor:
        """Call method from tensorflow."""
        inputs = self.strategy(inputs)
        residual_inputs = inputs
        outputs = tf.constant(0.0)
        for stack in self.stacks:
            # outputs_residual is (quantiles, Batch_size, forecast)
            # inputs is (Batch_size, backcast)
            reconstructed_outputs, residual_inputs = stack(residual_inputs)
            # outputs is (quantiles, Batch_size, forecast)
            outputs = tf.math.add(outputs, reconstructed_outputs)
        return outputs  # , inputs - reconstructed_inputs

    def seasonality(self, data: tf.Tensor) -> tf.Tensor:
        """
        TODO : unit testing
        Based on the paper, the seasonality component can be only available if
        the previous stacks are composed by trend blocks. Else, it doesn't correspond
        to seasonality.

        Returns
        -------
        seasonality : `Tensor`
            Seasonality components with shape (quantiles, Batch_size, input_width)

        Raises
        ------
        AttributeError
            If all previous stacks are not fully composed
            by trend blocks then an error is raised. Outputs stay accessible by
            the `stack_outputs` attributes.
        """

        msg_error = f"""Seasonality doesn't exists if the stacks
        preceding a `SeasonalityStack` are not `TrendStack`. Got {self.stacks},
        expected at least (`TrendStack`, `SeasonalityStack`, ...)."""

        if self.stacks[0].stack_type != "TrendStack":
            raise AttributeError(msg_error)
        start = 0
        for idx, stack in enumerate(self.stacks):
            if stack.stack_type == "SeasonalityStack":
                start = idx
            elif stack.stack_type != "TrendStack" and start == 0:
                raise AttributeError(msg_error)
            elif stack.stack_type == "TrendStack":
                continue
            else:
                return self._output_residuals[start:idx]
        return [stack(data) for stack in self.stacks[start:]]

    def trend(self, data: tf.Tensor) -> tf.Tensor:
        """
        TODO : unit testing
        The trend component of the output.

        Returns
        -------
        trend : `Tensor`
            Tensor of shape (d0,..., batch_size, input_width).

        Raises
        ------
        AttributeError
            Raises an error if the first block is not a `TrendBlock`."""

        if self.stacks[0].stack_type != "TrendStack":
            raise AttributeError(
                f"trend doesn't exists if the firsts stacks are not `TrendStack`. Got {self.stacks}."
            )
        for idx, stack in enumerate(self.stacks):
            if stack.stack_type != "TrendStack":
                return [stack(data) for stack in self.stacks[:idx]]
        return [stack(data) for stack in self.stacks]

    def get_config(self) -> dict:
        """Get_config from tensorflow."""
        return {"stacks": self.stacks}

    @property
    def input_width(self) -> int:
        """Return the list of blocks."""
        return self.stacks[0].input_width

    @property
    def stacks(self) -> List[Stack]:
        """Return the list of stacks"""
        return self._stacks

    @property
    def is_interpretable(self) -> bool:
        """Return True if the model is interpretable."""
        return self._is_interpretable

    @property
    def nbeats_type(self) -> str:
        """Return the Nbeats type."""
        return self._nbeats_type

    def _set_interpretability(self) -> bool:
        return all(stack.is_interpretable for stack in self.stacks)

    def _set_type(self):
        """Defines the type of Nbeats."""
        if self.is_interpretable:
            return "InterpretableNbeats"
        return "Nbeats"

    def __repr__(self):
        return self._nbeats_type

    def val___init__(self, output: tf.Tensor, *args: list, **kwargs: dict
        ) -> None:  # pylint: disable=unused-argument
        for block in self.stacks:
            assert block.input_width == self.input_width, ValueError(f"""`input_width` values are not equals along stacks.""")


class PoolNBEATS(Model, AutopycoinBaseClass):
    """
    TODO: finish doc and unit testing.
    Tensorflow model defining a pool of N-BEATS models.

    As described in the paper https://arxiv.org/abs/1905.10437, the state-of-the-art results
    are reached with a bagging method of N-BEATS models including interpretable and generic ones.

    Parameters
    ----------
    n_models : int
        Number of models inside the pool.
    nbeats_models : List[Callable]
        A list of callables which create a NBEATS model.
        It has to be the same shape than nbeats_kwargs.
    nbeats_kwargs : List[Callable]
        A list of dictionnaries which define the NBEATS parameters associated to nbeats_models.
        It has to be the same shape than nbeats_models.
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
    >>> w = w.from_dataframe(data=data,
    ...        input_columns=['test'],
    ...        known_columns=[],
    ...        label_columns=['test'],
    ...        date_columns=[],)
    ...
    >>> nbeats = create_interpretable_nbeats(
    ...                 input_width=70,
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
    ...             nbeats_models=[nbeats],
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
        nbeats_models: List[NBEATS],
        losses: list,
        fn_agg: Callable = tf.reduce_mean,
        seed: Union[None, int] = None,
        **kwargs: dict,
    ):

        super().__init__(**kwargs)

        # Reproducible instance
        if seed is not None:
            tf.random.set_seed(seed)

        # Init pool of losses by picking randomly loss in losses list.
        losses_idx = randint(0, len(losses), size=n_models)
        self._pool_losses = [losses[idx] for idx in losses_idx]

        # nbeats init
        self._init_nbeats(nbeats_models, n_models=n_models)

        # Layer definition and function to aggregate the multiple outputs
        self.strategy = UniVariate()
        self._fn_agg = fn_agg

    def _init_nbeats(self, nbeats_models: List[NBEATS], n_models: int):
        """Initialize nbeats models."""

        self._nbeats = []
        # Init pool of models by picking randomly model in nbeats_models list.
        nbeats_idx = randint(0, len(nbeats_models), size=n_models)
        for idx in nbeats_idx:
            # extract input_width and label_width from nbeats_kwargs
            config = nbeats_models[idx].get_config()
            # Used to mask the inputs in order to mimic bag ensemble.
            # Set randomly n_models masks.
            self._mask = uniform(
                0,
                int(nbeats_models[idx].input_width / nbeats_models[idx].label_width),
                size=n_models,
            ).astype(tf.int32)
            config['input_width'] = config['input_width'] - (self._mask[idx] * config['label_width'])
            model = NBEATS(**config)
            self._nbeats.append(model)

        self._set_quantiles()

    def _set_quantiles(self):
        for idx, loss in enumerate(self._pool_losses):
            if hasattr(loss, "quantiles"):
                self._nbeats[idx]._set_quantiles(loss.quantiles)

    def call(self, inputs: Union[tuple, dict, list, tf.Tensor]):
        """Call method from tensorflow Model."""
        inputs = self.strategy(inputs)
        outputs = []
        for model in self._nbeats:
            inputs_masked = inputs[
                :, -model.input_width:
            ]
            outputs.append(model(inputs_masked))
        return outputs

    def predict(self, *args: list, **kwargs: dict):
        """Reduce the n outputs to a single output tensor by mean operation."""
        outputs = super().predict(*args, **kwargs)
        return self._fn_agg(outputs, axis=0)

    def get_pool_losses(self):
        """Return the pool losses."""
        return self._pool_losses

    def reset_pool_losses(
        self,
        losses: List[Union[str, tf.keras.losses.Loss]],
        seed: Union[int, None] = None,
    ):
        """Reset the pool losses based on the given losses."""
        if seed is not None:
            tf.random.set_seed(seed)
        self._pool_losses = [self.randint(len(losses)) for _ in self._nbeats]
        self._set_quantiles()


def create_interpretable_nbeats(
    input_width: int,
    label_width: int,
    forecast_periods: List[int],
    backcast_periods: List[int],
    forecast_fourier_order: List[int],
    backcast_fourier_order: List[int],
    p_degree: int = 1,
    trend_n_neurons: int = 16,
    seasonality_n_neurons: int = 16,
    drop_rate: float = 0.0,
    share: bool = True,
    **kwargs: dict,
):
    """
    Wrapper to create an interpretable model using recommendations of the paper.
    Two stacks are created with 3 blocks each. The fisrt entirely composed by trend blocks,
    The second entirely composed by seasonality blocks.

    Within the same stack, it is possible to share the weights between blocks.

    Parameters
    ----------
    input_width : int
        Horizon time to forecast.
    label_width : int
        Past to rebuild. Usually, label_width = n * input_width with n between 1 and 7.
    forecast_periods : Tuple[int, ...]
        Compute the fourier serie period in the forecasting equation.
        If it's a list all forecast_periods are taken into account in the calculation.
    backcast_periods : Tuple[int, ...]
        Compute the fourier serie period in the backcasting equation.
        If it's a list all forecast_periods are taken into account in the calculation.
    forecast_fourier_order : Tuple[int, ...]
        Compute the fourier order. each order element is linked the respective period.
    backcast_fourier_order : Tuple[int, ...]
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
    >>> model = create_interpretable_nbeats(input_width=2,
    ...                                     label_width=3,
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
            input_width=input_width,
            label_width=label_width,
            p_degree=p_degree,
            n_neurons=trend_n_neurons,
            drop_rate=drop_rate,
            name="trend_block",
        )

        seasonality_block = SeasonalityBlock(
            input_width=input_width,
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
                input_width=input_width,
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
                input_width=input_width,
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
    input_width: int,
    label_width: int,
    forecast_neurons: int,
    backcast_neurons: int,
    n_neurons: int,
    n_blocks: int,
    n_stacks: int,
    drop_rate: float = 0.0,
    share: bool = True,
    **kwargs: dict,
):
    """
    Wrapper to create a generic model.

    In the same stack, it is possible to share the weights between blocks.

    Parameters
    ----------
    input_width : int
        Horizon time to forecast.
    label_width : int
        Past to rebuild. Usually, label_width = n * input_width with n between 1 and 7.
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
    >>> model = create_generic_nbeats(input_width=2,
    ...                               label_width=3,
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
                input_width=input_width,
                label_width=label_width,
                forecast_neurons=forecast_neurons,
                backcast_neurons=backcast_neurons,
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
                    input_width=input_width,
                    label_width=label_width,
                    forecast_neurons=forecast_neurons,
                    backcast_neurons=backcast_neurons,
                    n_neurons=n_neurons,
                    drop_rate=drop_rate,
                    name="generic_block",
                )
                for _ in range(n_blocks)
            ]

            generic_stacks.append(Stack(generic_blocks, name="generic_stack"))

    model = NBEATS(generic_stacks, name="generic_NBEATS", **kwargs)
    return model
