"""
These functions generate time series.
"""

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.backend import floatx

from ..utils import range_dims


def random_ts(
    n_steps: int,
    trend_degree: int,
    periods: list,
    fourier_orders: list,
    trend_mean: int = 0,
    trend_std: int = 1,
    seasonality_mean: int = 0,
    seasonality_std: int = 1,
    batch_size: int = 1,
    n_variables: int = 1,
    noise: bool = False,
    seed: None or int = None,
) -> tf.Tensor:
    """
    Generate random time series with a trend and seasonality components.

    Coefficients for the trend and seasonality components
    are generated by Gaussian distributions.

    Parameters
    ----------
    n_steps : int
        Number of steps
    trend_degree : int
        Number of degree for the trend function. Usually 1 or 2.
    periods : list[int]
        Computes the seasonality component.
        A period lower than n_step indicates a pattern which is repeating.
    fourier_orders : list[int]
        Computes the complexity of the seasonality.
        Higher oreder means higher complexity.
    trend_mean : int
        Mean value generated by the random process.
        Default to 0.
    trend_std : int
        Deviation value generated by the random process.
        Default to 1.
    seasonality_mean : int
        Mean value generated by the random process.
        Default to 0.
    seasonality_std : int
        Deviation value generated by the random process.
        Default to 1.
    batch_size : int
        Number of time series.
        Default to 1.
    n_variables : int
        Number of variables.
        Default to 1.
    noise : boolean
        Add noise to the series.
        Default to False.
    seed : int or None
        Allow reproducible results.
        Default to None.

    Returns
    -------
    time_series : `tensor`
        Times series defined by a randm process.

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
    ...                  batch_size=1,
    ...                  n_variables=1,
    ...                  noise=True,
    ...                  seed=42)
    >>> data.shape
    TensorShape([1, 10, 1])
    """

    tf.random.set_seed(seed)

    if isinstance(batch_size, int):
        batch_size = [batch_size]

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
