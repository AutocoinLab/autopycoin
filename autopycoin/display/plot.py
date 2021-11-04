"""
This file defines the plot function to use with generator.
"""

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from ..utils import example_handler


def plot_intervals(inputs, model, date_labels, instance, interval_estimator=None):
    # If interval_estimator is not defined then we plot only one isntance of prediction
    if interval_estimator is not None:
        mean, min_interval, max_interval = interval_estimator(inputs, model=model)
        # quantile 0.5 is taken as reference
        n_quantiles = mean.shape[0]
        middle = int(np.ceil(n_quantiles / 2))

        # If mean doesn't contain quantiles i.e, quantiles = 1 or loss is not defining quantiles
        if hasattr(model.loss, "quantiles"):
            min_interval = min_interval[middle - 1]
            max_interval = max_interval[middle - 1]

        plt.fill_between(
            date_labels[instance],
            min_interval[instance],
            max_interval[instance],
            alpha=0.5,
            color="#045a8d",
            label=f"Epistemic error: {interval_estimator.quantile} quantile",
        )

    else:
        mean = model.predict(inputs)
        # quantile 0.5 is taken as reference
        n_quantiles = mean.shape[0]
        middle = int(np.ceil(n_quantiles / 2))

    # if model is not defining quantiles then we don't plot them
    if hasattr(model.loss, "quantiles"):
        plt.plot(
            date_labels[instance],
            mean[middle - 1, instance],
            color="#fff7bc",
            marker="X",
            markeredgecolor="k",
            label="Predictions",
        )

        for i, k in zip(range(n_quantiles), model.loss.quantiles[: middle - 1]):
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


def plot_ts(
    window_generator,
    dataset,
    plot_col,
    model=None,
    plot_labels=True,
    plot_history=None,
    plot_interval=True,
    interval_estimator=None,
    max_subplots=3,
    fig_kwargs={"figsize": (20, 20)},
):
    """
    Display the results with matplotlib.

    Parameters
    ----------
    dataset : `tf.data.Dataset with shape
            (inputs, date_inputs, date_labels), labels
            or (inputs, known, date_inputs, date_labels), labels`
        Dataset from `WindowGenerator`.
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

    if isinstance(dataset, tf.Tensor):
        dataset = window_generator.forecast(dataset, None)

    (inputs, _, date_inputs, date_labels), labels = example_handler(dataset)
    fig = plt.figure(**fig_kwargs)

    # Get label and plot column indices
    if window_generator.label_columns:
        plot_col_index = window_generator.inputs_columns_indices.get(plot_col, None)
        label_col_index = window_generator.label_columns_indices.get(plot_col, None)
    else:
        plot_col_index = window_generator.column_indices[plot_col]
        label_col_index = window_generator.column_indices[plot_col]

    # We loop over max_subplots instances
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
        plt.subplot(max_n, 1, n + 1)
        plt.ylabel(f"{plot_col}")

        if plot_history is not None:
            plt.plot(
                plot_history[n, :, 0],
                plot_history[n, :, 1],
                label="Inputs",
                marker=".",
                zorder=-10,
            )
        else:
            plt.plot(
                date_inputs[n],
                inputs[n],
                label="Inputs",
                marker=".",
                zorder=-10,
            )

        if label_col_index is None:
            continue

        if plot_labels is True:
            plt.plot(
                date_labels[n],
                labels[n],
                color="#fe9929",
                marker=".",
                markeredgecolor="k",
                label="Real values",
            )

        if model is not None:
            if plot_interval:
                plot_intervals(inputs, model, date_labels, n, interval_estimator)

            # If we don't want to plot interval then plot only the quantile 0.5 if it exists else just the output
            else:
                output = model.predict(inputs)
                n_quantiles = output.shape[0]
                middle = int(np.ceil(n_quantiles / 2))
                if hasattr(model.loss, "quantiles"):
                    plt.plot(
                        date_labels[n],
                        output[middle - 1, n],
                        color="#fff7bc",
                        marker="X",
                        markeredgecolor="k",
                        label="Predictions",
                    )
                else:
                    plt.plot(
                        date_labels[n],
                        output[n],
                        color="#fff7bc",
                        marker="X",
                        markeredgecolor="k",
                        label="Predictions",
                    )

        if n == 0:
            plt.legend()
        plt.xticks(rotation="vertical")

    fig.tight_layout()
    plt.xlabel("Time")
