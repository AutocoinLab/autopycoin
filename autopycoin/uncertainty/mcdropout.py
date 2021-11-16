"""
Contains uncertainty estimation object based on mc dropout method
"""

from typing import Union, List, Tuple
from abc import abstractmethod

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

InputsType = Union[tf.Tensor, List[tf.Tensor], dict, tf.data.Dataset]


# We need to create base class which will be the abstract class
# UncertaintyEstimator will be then used to check if model and inputs is good.
class UncertaintyEstimator:
    """
    An uncertainty estimator needs to define a quantile attributes
    and a call method where logic calculation of uncertainty is defined.
    """

    def __init__(self, quantile):
        self._quantile = quantile

    @abstractmethod
    def call(self, inputs, model):
        """
        This method needs to be defined
        when class is inherited from UncertinatyEstimator.
        """
        raise NotImplementedError("Must be implemented in subclasses.")

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    @property
    def quantile(self):
        return self._quantile


class MCDropoutEstimator(UncertaintyEstimator):
    """
    Convenient object which estimate an epistemic error thanks to MC dropout method.
    It consists to predict an instance a multiple time with a dropout rate > 0.
    It support only numerical output for the moment.

    Parameters
    ----------
    n_preds : int
        Number of time a prediction is run.
    quantile : float or int
        quantile used to calculate the uncertainty interval. Default to 0.99.

    Returns
    -------
    mean : `tensor`
        Return the mean of the bench of outputs defined by n_preds.
    min_interval : `tensor`
        Return the maximum of the bench of outputs defined by n_preds.
    max_interval : `tensor`
        Return the minimum of the bench of outputs defined by n_preds.

    Attributes
    ----------
    n_preds : int
    quantile : float or int

    Examples
    --------
    >>> import pandas as pd
    >>> from autopycoin.models import create_interpretable_nbeats
    >>> from autopycoin.dataset import WindowGenerator
    >>> from autopycoin.losses import QuantileLossError
    >>> from autopycoin.data import random_ts
    >>> import tensorflow as tf
    >>> data = random_ts(
    ...    n_steps=400,
    ...    trend_degree=2,
    ...    periods=[10],
    ...    fourier_orders=[10],
    ...    trend_mean=0,
    ...    trend_std=1,
    ...    seasonality_mean=0,
    ...    seasonality_std=1,
    ...    batch_size=1,
    ...    n_variables=1,
    ...    noise=True,
    ...    seed=42,
    ... )
    ...
    >>> data = pd.DataFrame(data[0].numpy(), columns=['test'])
    >>> w = WindowGenerator(
    ...    input_width=50,
    ...    label_width=20,
    ...    shift=20,
    ...    test_size=10,
    ...    valid_size=10,
    ...    strategy="one_shot",
    ...    batch_size=32,
    ... )
    >>> w = w.from_dataframe(
    ...    data=data,
    ...    input_columns=["test"],
    ...    known_columns=[],
    ...    label_columns=["test"],
    ...    date_columns=[])
    >>> model = create_interpretable_nbeats(
    ...    input_width=20,
    ...    label_width=50,
    ...    periods=[10],
    ...    back_periods=[10],
    ...    forecast_fourier_order=[10],
    ...    backcast_fourier_order=[10],
    ...    p_degree=1,
    ...    trend_n_neurons=32,
    ...    seasonality_n_neurons=32,
    ...    drop_rate=0.1,
    ...    share=True,
    ... )
    ...
    >>> model.compile(
    ...   tf.keras.optimizers.Adam(
    ...        learning_rate=0.015,
    ...        beta_1=0.9,
    ...        beta_2=0.999,
    ...        epsilon=1e-07,
    ...        amsgrad=False,
    ...        name="Adam",
    ...    ),
    ...    loss=QuantileLossError([0.1, 0.3, 0.5]),
    ...    metrics=["mae"],
    ... )
    >>> model.build(tf.TensorShape((None, 50)))
    >>> estimator = MCDropoutEstimator(
    ...    n_preds=10, quantile=0.99
    ... )
    >>> [tensor.shape for tensor in estimator(w.train, model)]
    [TensorShape([5, 273, 20]), TensorShape([5, 273, 20]), TensorShape([5, 273, 20])]

    Notes
    -----
    *Input shape*:
    N-D tensor with shape: (batch_size, ..., input_dim).
    The most common situation would be a 2D input with shape (batch_size, input_dim).

    *Output shape*:
    3 N-D tensor with shape: (quantiles, batch_size, ..., units) or (batch_size, ..., units).
                             (quantiles, batch_size, ..., units) or (batch_size, ..., units).
                             (quantiles, batch_size, ..., units) or (batch_size, ..., units).
    For instance, for a 2D input with shape (batch_size, input_dim),
    the outputs would have shapes (batch_size, units).
    With a QuantileLossError with 2 quantiles or higher the outputs
    would have shapes (quantiles, batch_size, units).
    """

    def __init__(self, n_preds: int, quantile: float = 0.99):
        super().__init__(quantile)
        self.n_preds = n_preds

    @staticmethod
    def mc_dropout_estimation(
        inputs: InputsType, model: tf.keras.Model, n_preds: int
    ) -> tf.Tensor:
        """
        Run a prediction multiple time. the number of run is defined by n_preds.
        Then the outputs are stacked in the first axis.

        Parameters
        ----------
        inputs : `tensor, list of `tensor`, dict or dataset
            The inputs are used to produce prediction.
        model : `tensorflow model`
            Model which defines dropout layers.
        n_preds : int
            Number of time a prediction is run.

        Returns
        -------
        outputs : `tensor`
            tensor of shape (n_preds, quantiles, batch_size, dim) or
            (batch_size, dim).
        """
        outputs = []
        for _ in tf.range(n_preds):
            output = model.predict(inputs)
            outputs.append(output)
        return tf.stack(outputs, axis=0)

    def call(self, inputs: InputsType, model: tf.keras.Model) -> Tuple[tf.Tensor]:
        """
        Run a prediction multiple time and stack them in axis=0. Then the tensor is reduced
        by mean and standard deviation to create 3 tensors.
         - mean : Represents the average result over the n_preds run.
         - min_interval : Represents the minimum interval defined by quantile.
         - min_interval : Represents the maximum interval defined by quantile.

        Parameters
        ----------
        inputs : `tensor, list of `tensor`, dict or dataset
            The inputs are used to produce prediction.
        model : `tensorflow model`
            Model which defines dropout layers.

        Returns
        -------
        mean : `tensor`
            tensor of shape (quantiles, batch_size, dim) or
            (batch_size, dim).
        min_interval : `tensor`
            tensor of shape (quantiles, batch_size, dim) or
            (batch_size, dim).
        max_interval : `tensor`
            tensor of shape (quantiles, batch_size, dim) or
            (batch_size, dim).
        """

        outputs = self.mc_dropout_estimation(inputs, model, self.n_preds)
        mean = tf.reduce_mean(outputs, axis=0)
        standard_deviation = tf.math.reduce_std(outputs, axis=0)

        quantile = tfd.Normal(loc=0, scale=1).quantile(self.quantile)
        max_interval = mean + quantile * standard_deviation
        min_interval = mean - quantile * standard_deviation

        return mean, min_interval, max_interval
