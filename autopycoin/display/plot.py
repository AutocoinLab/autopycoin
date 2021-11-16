"""
This file defines the plot function to use with generator.
"""

from typing import List, Union, Tuple
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from ..dataset import WindowGenerator
from ..baseclass import AutopycoinBaseClass
from ..uncertainty import MCDropoutEstimator
from ..utils import example_handler



class PlotTs(AutopycoinBaseClass):
    """
    Display a model results with matplotlib.

    Parameters
    ----------
    dataset : `tf.data.Dataset with shape
            (inputs, date_inputs, date_labels), labels
            or (inputs, known, date_inputs, date_labels), labels` or str or `Tensor`
        Dataset from `WindowGenerator` or string value between ['train', 'valid', 'test'],
        a `Tensor` can be provided and needs to match the model input shape.
    plot_col : list[str]
        Columns to plot.
    model : `Regressor defining a predict method`
        The model needs to return a 3D output tensor, default to `None`.
        If `None` no predictions are displaying.
    plot_labels : bool
        Plot ground truth label data, default to `True`.
    plot_history : list[tensors or array] `with shape (historical date, historical values)`
        Plot the history.
    max_subplots : int
        Number of test instances to display, default to 3.
    fig_kwargs : dict
        Matplotlib figure's parameters.
    """

    def __init__(
        self,
        window_generator: WindowGenerator,
        dataset: Union[tf.Tensor, str],
        model: Union[tf.keras.Model, None] = None,
        interval_estimator: Union[MCDropoutEstimator, None] = None,
        **kwargs: dict,
    ):

        self.window_generator = window_generator
        self.window_generator.batch_size_cached = self.window_generator.batch_size
        self.window_generator.batch_size = None
        self.model = model
        self.interval_estimator = interval_estimator
        self.fig_kwargs = kwargs

        if isinstance(dataset, tf.Tensor):
            self._dataset = self.window_generator.production(dataset, None)
        else:
            self._dataset = getattr(self.window_generator, dataset)
        
        self.window_generator.batch_size = self.window_generator.batch_size_cached

    def plot_from_index(
        self,
        index: Union[str, List[str]],
        plot_col: str,
        plot_labels: bool=True,
        plot_history: np.array=None,
        plot_interval: bool=True,
        max_subplots: int=3,
    ):
        """Plot the results starting at the index provided."""

        (inputs, _, date_inputs, date_labels), labels = self._filtering_by_index(index)

        fig = plt.figure(**self.fig_kwargs)

        # We loop over max_subplots instances
        max_plots = min(max_subplots, len(inputs))
        for plot in range(max_plots):
            plt.subplot(max_plots, 1, plot + 1)
            plt.ylabel(f"{plot_col}")

            if plot_history is not None:
                plt.plot(
                    plot_history[plot, :, 0],
                    plot_history[plot, :, 1],
                    label="Inputs",
                    marker=".",
                    zorder=-10,
                )
            else:
                plt.plot(
                    date_inputs[plot],
                    inputs[plot],
                    label="Inputs",
                    marker=".",
                    zorder=-10,
                )

            if plot_labels is True:
                plt.plot(
                    date_labels[plot],
                    labels[plot],
                    color="#fe9929",
                    marker=".",
                    markeredgecolor="k",
                    label="Real values",
                )

            if self.model is not None:
                if plot_interval:
                    self._plot_intervals(
                        inputs, date_labels, plot
                    )

                # If we don't want to plot interval then plot only the quantile 0.5 if it exists else just the output
                else:
                    output = self.model.predict(inputs)
                    n_quantiles = output.shape[0]
                    middle = int(np.ceil(n_quantiles / 2))
                    if self.model.quantiles:
                        plt.plot(
                            date_labels[plot],
                            output[middle - 1, plot],
                            color="#fff7bc",
                            marker="X",
                            markeredgecolor="k",
                            label="Predictions",
                        )
                    else:
                        plt.plot(
                            date_labels[plot],
                            output[plot],
                            color="#fff7bc",
                            marker="X",
                            markeredgecolor="k",
                            label="Predictions",
                        )

            if plot == 0:
                plt.legend()
            plt.xticks(rotation="vertical")

        fig.tight_layout()
        plt.xlabel("Time")

    def _filtering_by_index(self, index: Union[str, List[str]]) -> Tuple[Tuple[tf.Tensor, ...], tf.Tensor]:
        """Return the dataset filtered by the index in `Tensor` format."""
        (inputs, known, date_inputs, date_labels), labels = example_handler(self._dataset)
        idx = np.where(date_labels[..., :, 0] == index)
        assert np.size(idx), f"""index {index} not found inside the dataset."""
        idx = np.squeeze(idx)
        return (inputs[..., idx:], known[..., idx:], date_inputs[..., idx:], date_labels[..., idx:]), labels[..., idx:]

    def _plot_intervals(self, inputs: tf.Tensor, date_labels: np.ndarray, instance: int) -> None:
        # If interval_estimator is not defined then we plot only one instance of prediction
        if self.interval_estimator:
            mean, min_interval, max_interval = self.interval_estimator(inputs, model=self.model)
            # quantile 0.5 is taken as reference
            n_quantiles = mean.shape[0]
            middle = int(np.ceil(n_quantiles / 2))

            # If mean doesn't contain quantiles i.e, quantiles = 1 or loss is not defining quantiles
            if hasattr(self.model, "quantiles"):
                min_interval = min_interval[middle - 1]
                max_interval = max_interval[middle - 1]

            plt.fill_between(
                date_labels[instance],
                min_interval[instance],
                max_interval[instance],
                alpha=0.5,
                color="#045a8d",
                label=f"Epistemic error: {self.interval_estimator.quantile} quantile",
            )

        else:
            mean = self.model.predict(inputs)
            # quantile 0.5 is taken as reference
            n_quantiles = mean.shape[0]
            middle = int(np.ceil(n_quantiles / 2))

        # if model is not defining quantiles then we don't plot them
        if self.model.quantiles:
            plt.plot(
                date_labels[instance],
                mean[middle - 1, instance],
                color="#fff7bc",
                marker="X",
                markeredgecolor="k",
                label="Predictions",
            )

            for i, k in zip(range(n_quantiles), self.model.quantiles[: middle - 1]):
                if i != middle - 1:
                    plt.fill_between(
                        date_labels[instance],
                        mean[mean.shape[0] - i - 1, instance],
                        mean[i, instance],
                        alpha=0.5,
                        label=f"Aleotoric error: {k:.2f} quantile",
                    )

        else:
            plt.plot(
                date_labels[instance],
                mean[instance],
                color="#fff7bc",
                marker="X",
                markeredgecolor="k",
                label="Predictions",
            )

        return plt.gca()