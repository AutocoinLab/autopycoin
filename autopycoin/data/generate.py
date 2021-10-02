"""
These functions generate time series.
"""

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.backend import floatx


def random_univariate_ts(steps, trend_degree, periods, fourier_orders,
                         trend_mean=0, trend_std=1, seasonality_mean=0, seasonality_std=1,
                         batch_size=1, noise=True, seed=42):

    tf.random.set_seed(seed)

    degree = tf.range(trend_degree, dtype=floatx())
    time = tf.expand_dims(tf.range(steps, dtype=floatx()), axis=-1)

    # Workout random trend coefficients
    trend_coef = tf.random.normal(
    (batch_size, trend_degree), mean=trend_mean, stddev=trend_std, dtype=tf.dtypes.float32, seed=seed
    )

    # trend component added to ts
    time_serie = tf.reduce_sum(trend_coef * ((time/steps)** degree), axis=[-1])

    # Shape (-1, 1) in order to broadcast periods to all time units
    periods = tf.cast(tf.expand_dims(periods, axis=-1), dtype=floatx())

    for fourier_order, period in zip(fourier_orders, periods):
        time = 2 * np.pi * time / period
        seasonality = time * tf.range(fourier_order, dtype=floatx())

        # Workout cos and sin seasonality coefficents
        seasonality = tf.concat((tf.cos(seasonality), tf.sin(seasonality)), axis=-1)

        # Workout random seasonality coefficents
        seas_coef = tf.random.normal(
        (seasonality.shape[-1],), mean=seasonality_mean, stddev=seasonality_std, dtype=tf.dtypes.float32, seed=seed
        )
        seasonality = tf.reduce_sum(seasonality*seas_coef, axis=-1)

        # Seasonality coefficients added to ts
        time_serie = time_serie + seasonality

    if noise:
        noise = tf.random.normal((steps,), mean=0.0, stddev=1, dtype=tf.dtypes.float32, seed=seed)
        return time_serie + noise

    return time_serie
