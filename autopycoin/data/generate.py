"""
These functions generate time series.
"""

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.backend import floatx


def range_dims(tensor, shape):
    tensor = tf.range(tensor, dtype="float32")
    return tf.reshape(tensor, shape=shape)


def random_ts(
    steps,
    trend_degree,
    periods,
    fourier_orders,
    trend_mean=0,
    trend_std=1,
    seasonality_mean=0,
    seasonality_std=1,
    batch_size=1,
    n_series=1,
    noise=True,
    seed=42,
):

    tf.random.set_seed(seed)

    if isinstance(batch_size, int):
        batch_size = [batch_size]

    degree = range_dims(trend_degree, shape=(1, -1, 1))
    time = range_dims(steps, shape=(-1, 1, 1))

    # Workout random trend coefficients
    trend_coef = tf.random.normal(
        (*batch_size, 1, trend_degree, n_series),
        mean=trend_mean,
        stddev=trend_std,
        dtype=floatx(),
        seed=seed,
    )

    # trend component added to ts
    time_series = tf.reduce_sum(trend_coef * ((time / steps) ** degree), axis=-2)

    # Shape (-1, 1) in order to broadcast periods to all time units
    periods = tf.reshape(periods, shape=(-1, 1, 1))
    periods = tf.cast(periods, dtype="float32")

    for fourier_order, period in zip(fourier_orders, periods):

        time = 2 * np.pi * time / period
        fourier_order = range_dims(fourier_order, shape=(-1, 1))
        seasonality = time * fourier_order

        # Workout cos and sin seasonality coefficents
        seasonality = tf.concat((tf.cos(seasonality), tf.sin(seasonality)), axis=-2)

        # Workout random seasonality coefficents
        seas_coef = tf.random.normal(
            (*batch_size, 1, seasonality.shape[-1], n_series),
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
            (steps, 1), mean=0.0, stddev=1, dtype=floatx(), seed=seed
        )
        return time_series + noise

    return time_series
