.. title:: Basics

==================================================
Basics: Time series forecasting
==================================================

Our models
----------

This package aims to define multiple deep learning model for time series forecasting
using tensorflow. The first model available is :class:`autopycoin.models.N_BEATS`.
We provide easy-to-use functions which help you to use this model.
They can be imported as::

    >>> from autopycoin.models import create_interpretable_nbeats, create_generic_nbeats
    >>>
    >>> from autopycoin.models import create_interpretable_nbeats
    >>> from autopycoin.losses import QuantileLossError
    >>> model = create_interpretable_nbeats(label_width=3,
    ...                                     forecast_periods=[2],
    ...                                     backcast_periods=[3],
    ...                                     forecast_fourier_order=[2], # higher give more flexibility but can overfit
    ...                                     backcast_fourier_order=[3], # higher give more flexibility but can overfit
    ...                                     p_degree=1,
    ...                                     trend_n_neurons=16, # Neurons in the fully connected layers
    ...                                     seasonality_n_neurons=16, # Neurons in the fully connected layers
    ...                                     drop_rate=0,
    ...                                     share=True) # share weights across blocks
    >>> model.compile(loss="mse")

You can also use class based models::

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

It lets you free to try other combinations of layers.

This model come from the paper https://openreview.net/forum?id=r1ecqn4YwB
As you can notice, there is two kind of N-BEATS models. The interpretable one gives explanation by decomposing the outputs in seasonality and trend function
which is not the case of the generic one.
For more details, you can visit my notebook on the subject https://www.kaggle.com/gatandubuc/forecast-with-n-beats-interpretable-model.

Our losses
----------

We defined some loss functions, currently available:

* Quantile loss function
* Symetric mean absolute percentage function

The both of them can be imported as::

    >>> from autopycoin.losses import QuantileLossError, SymetricMeanAbsolutePercentageError

Or they can be used directly as functions::

    >>> from autopycoin.losses import quantile_loss, smape

these functions are using keras and tensorflow library hence they are compatible with their api.
