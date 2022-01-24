"""
These functions are used to generate time series.
"""

import numpy as np
from typing import Union, List, Optional

import tensorflow as tf
from tensorflow.keras.backend import floatx

from ..utils import range_dims


def random_ts(
    n_steps: int = 100,
    trend_degree: int = 2,
    periods: List[int] = [10],
    fourier_orders: List[int] = [10],
    trend_mean: Optional[int] = 0,
    trend_std: Optional[int] = 1,
    seasonality_mean: Optional[int] = 0,
    seasonality_std: Optional[int] = 1,
    batch_size: Optional[Union[List[int], int]] = None,
    n_variables: Optional[int] = 1,
    noise: Optional[bool] = False,
    seed: Optional[int] = None,
) -> tf.Tensor:
    """
    Generate random time series composed by a trend and seasonality components.

    Coefficients for the trend and seasonality components are generated by Gaussian distributions.

    Parameters
    ----------
    n_steps : int
        Number of steps.
    trend_degree : int
        Number of degree for the trend function. Usually 1 or 2.
    periods : list[int]
        Computes the seasonality component.
        A period lower than n_step leads to a pattern which is repeating.
    fourier_orders : list[int]
        Computes the complexity of the seasonality component.
        Higher order means higher complexity.
    trend_mean : int
        Mean value associated to the trend component.
        Default to 0.
    trend_std : int
        Deviation value associated to the trend component.
        Default to 1.
    seasonality_mean : int
        Mean value associated to the seasonality component.
        Default to 0.
    seasonality_std : int
        Deviation value associated to the seasonality component.
        Default to 1.
    batch_size : int or list[int] or None
        Computes the batch_size of the time series.
        Default to None.
    n_variables : int
        Number of variables.
        Default to 1.
    noise : boolean
        Add centered normal distributed noise to the series.
        Default to False.
    seed : int or None
        Allow reproducible results.
        Default to None.

    Returns
    -------
    time_series : `tensor of shape [D0, ..., n_steps, n_variables]`
        Time series defined by a random process.

    Examples
    --------
    >>> from autopycoin.data import random_ts
    >>> data = random_ts(n_steps=10,
    ...                  trend_degree=2,
    ...                  periods=[10],
    ...                  fourier_orders=[10],
    ...                  trend_mean=0,
    ...                  trend_std=1,
    ...                  seasonality_mean=0,
    ...                  seasonality_std=1,
    ...                  batch_size=None,
    ...                  n_variables=1,
    ...                  noise=True,
    ...                  seed=42)
    >>> data.shape
    TensorShape([10, 1])
    """

    tf.random.set_seed(seed)

    if isinstance(batch_size, int):
        batch_size = [batch_size]
    elif batch_size is None:
        batch_size = []

    degree = range_dims(trend_degree, shape=(1, -1, 1))
    time = range_dims(n_steps, shape=(-1, 1, 1))

    # Workout random trend coefficients
    trend_coef = tf.random.normal(
        (*batch_size, 1, trend_degree, n_variables),
        mean=trend_mean,
        stddev=trend_std,
        dtype=floatx(),
        seed=seed,
    )

    # trend component added to ts
    time_series = tf.reduce_sum(trend_coef * ((time / n_steps) ** degree), axis=-2)

    # Shape (-1, 1) in order to broadcast periods to all time units
    periods = tf.reshape(periods, shape=(-1, 1, 1))
    periods = tf.cast(periods, dtype=floatx())

    for fourier_order, period in zip(fourier_orders, periods):

        time = 2 * np.pi * time / period
        fourier_order = range_dims(fourier_order, shape=(-1, 1))
        seasonality = time * fourier_order

        # Workout cos and sin seasonality coefficents
        seasonality = tf.concat((tf.cos(seasonality), tf.sin(seasonality)), axis=-2)

        # Workout random seasonality coefficents
        seas_coef = tf.random.normal(
            (*batch_size, 1, seasonality.shape[-1], n_variables),
            mean=seasonality_mean,
            stddev=seasonality_std,
            dtype=floatx(),
            seed=seed,
        )

        seasonality = tf.reduce_sum(seasonality * seas_coef, axis=-2)

        # Seasonality coefficients added to ts
        time_series = time_series + seasonality

    if noise:
        noise = tf.random.normal(
            (n_steps, 1), mean=0.0, stddev=1, dtype=floatx(), seed=seed
        )
        return time_series + noise

    return time_series
