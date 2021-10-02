.. title:: Basics

==================================================
Basics: Time series forecasting
==================================================

Our models
----------

This package aims to define multiple deep learning model for time series forecasting
using tensorflow. The first model available is :class:`autopycoin.models.N_BEATS`.
We provide really easy-to-use functions which help you to use this model.
They can be imported as::

    >>>
    >>> from autopycoin.models import create_interpretable_nbeats, create_generic_nbeats
    >>>
    >>> nbeats_model = create_interpretable_nbeats(
    >>>     horizon=2,
    >>>     back_horizon=3,
    >>>     periods=[2],
    >>>     back_periods=[3],
    >>>     horizon_fourier_order=[2],
    >>>     back_horizon_fourier_order=[3],
    >>>     p_degree=1,
    >>>     trend_n_neurons=16,
    >>>     seasonality_n_neurons=16,
    >>>     quantiles=1,
    >>>     share=True,
    >>>     drop_rate=0.1,
    >>>)

You can also use class based models::

    >>> from autopycoin.models import TrendBlock, SeasonalityBlock, GenericBlock, Stack, NBEATS

This model come from the paper https://openreview.net/forum?id=r1ecqn4YwB
As you can notice, there is two kind of N-BEATS models. The interpretable one gives explanation by decomposing the outputs in seasonality and trend function
which is not the case of the generic one. For more details, you can visit my notebook on the subject https://www.kaggle.com/gatandubuc/forecast-with-n-beats-interpretable-model.


Our losses
----------

We defined some loss functions, currently available:

* Quantile loss function
* Symetric mean absolute percentage function

The both of them can be imported as::

    >>> from autopycoin.losses import QuantileLossError, SymetricMeanAbsolutePercentageError

Or they can be used directly as functions::

    >>> from autopycoin.losses import quantile_loss, smape
