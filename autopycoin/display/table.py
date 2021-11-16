"""
This file defines the table format for models outputs.
"""

import tensorflow as tf

from ..utils import example_handler


def table_ts(
    window_generator,
    dataset,
    plot_col,
    model=None,
    plot_labels=True,
    plot_history=None,
    plot_interval=True,
    interval_estimator=None,
    max_subplots=3,
):

    if isinstance(dataset, tf.Tensor):
        dataset = window_generator.forecast(dataset, None)

    (inputs, _, date_inputs, date_labels), labels = example_handler(dataset)

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
