"""
============================
Plotting NBEATS output
============================
An example plot of :class:`autopycoin.models.NBEATS`
"""

import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd

from autopycoin.data import random_ts
from autopycoin.dataset import WindowGenerator
from autopycoin.models import nbeats


tf.random.set_seed(0)

data = random_ts(n_steps=400, # Number of steps (second dimension)
                 trend_degree=2,
                 periods=[10], # We can combine multiple periods, period is the time length for a cyclical function to reproduce a similar output
                 fourier_orders=[10], # higher is this number, more complex is the output
                 trend_mean=0,
                 trend_std=1,
                 seasonality_mean=0,
                 seasonality_std=1,
                 batch_size=1, # Generate a batch of data (first dimension)
                 n_variables=1, # Number of variables (last dimension)
                 noise=True, # add normal centered noise
                 seed=42)


w = WindowGenerator(
        input_width=80,
        label_width=40,
        shift=40,
        test_size=50,
        valid_size=10,
        flat=True,
        batch_size=16,
        preprocessing = lambda x,y: (x, (x,y)) # NBEATS output
    )

data = pd.DataFrame(data.numpy()[0], columns=['test'])

w = w.from_array(data=data, # Has to be 2D array
        input_columns=['test'],
        known_columns=[],
        label_columns=['test'],
        date_columns=[],)

model1 = nbeats.create_interpretable_nbeats(
            label_width=40,
            forecast_periods=[10],
            backcast_periods=[10],
            forecast_fourier_order=[10],
            backcast_fourier_order=[10],
            p_degree=1,
            trend_n_neurons=200,
            seasonality_n_neurons=200,
            drop_rate=0.,
            share=True)

model1.compile(tf.keras.optimizers.Adam(
    learning_rate=0.0015, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True,
    name='Adam'),
    loss='mse',
    loss_weights=[1, 1], # In the paper = [0, 1]
    metrics=["mae"])

model1.fit(w.train, validation_data=w.valid, epochs=20)

iterator = iter(w.train)
x, y = iterator.get_next()

input_width = 80

plt.plot(range(input_width, input_width + 40), model1.predict(x)[1].values[0], label='forecast')
# Usefull only if stack = True
plt.plot(range(input_width), model1.predict(x)[0].values[0], label='backcast')
plt.plot(range(input_width, input_width + 40), y[1][0], label='labels')
plt.plot(range(input_width), x[0], label='inputs')
plt.legend()
