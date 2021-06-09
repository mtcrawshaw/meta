""" Plotting for performance metrics. """

from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

from meta.utils.metrics import Metrics


def plot(metrics: Metrics, plot_path: str) -> None:
    """
    Plot the metrics given in `metrics_history`, store image at `plot_path`.
    """

    # Count the number of plots. We create a plot for each unique basename among all
    # basenames from the given metrics with non-empty history.
    basenames = []
    for metric in metrics.metric_dict.values():
        if len(metric.history) > 0 and metric.basename not in basenames:
            basenames.append(metric.basename)

    # Add an extra plot which will just contain a table.
    num_plots = len(basenames) + 1

    # Create subplots.
    fig_width = 12.8
    plot_height = 4.8
    table_height_ratio = 0.3
    fig, axs = plt.subplots(
        num_plots,
        figsize=(fig_width, plot_height * (num_plots - (1.0 - table_height_ratio))),
        gridspec_kw={"height_ratios": [1.0] * (num_plots - 1) + [table_height_ratio]},
    )

    # Wrap a single axis in a list to make sure that axs is iterable.
    if not isinstance(axs, np.ndarray):
        axs = [axs]

    # Create figure title.
    fig.suptitle("Training Metrics")

    # Plot each metric set on a separate plot.
    plotted_metrics = []
    for i, basename in enumerate(basenames):

        # Construct metric names and find max metric length (scale of x-axis).
        legend = []
        metric_names = [
            metric_name
            for metric_name, metric in metrics.metric_dict.items()
            if metric.basename == basename
        ]
        max_metric_len = max(
            [len(metric.mean) for metric in metrics.metric_dict.values()]
        )

        for metric_name in metric_names:

            # Get metric state values.
            metric = metrics.metric_dict[metric_name]
            mean_array = np.array(metric.mean)
            stdev_array = np.array(metric.stdev)
            assert len(mean_array) == len(stdev_array)

            # Assign x-axis values to each data point.
            num_intervals = len(mean_array) - 1 if len(mean_array) > 1 else 1
            x_axis = [
                (i * max_metric_len) // num_intervals for i in range(len(mean_array))
            ]

            # Plot mean.
            axs[i].plot(x_axis, mean_array)
            legend.append(metric_name)

            # Fill space between mean and upper/lower std devs.
            upper_dev_array = mean_array + stdev_array
            lower_dev_array = mean_array - stdev_array
            axs[i].fill_between(x_axis, lower_dev_array, upper_dev_array, alpha=0.2)

            # Add x-axis label.
            axs[i].set_xlabel("Training episodes")

            plotted_metrics.append(metric_name)

        # Add legend to subplot.
        axs[i].legend(legend, loc="upper left")

    # Helper function for constructing cell text.
    possibly_none = lambda val: "%.5f" % val if val is not None else "None"

    # Write out table of final metrics.
    axs[-1].axis("off")
    row_labels = list(plotted_metrics)
    col_labels = ["Best", "Final"]
    cell_text = []
    for metric_name in plotted_metrics:
        metric = metrics.metric_dict[metric_name]
        row_text = []
        row_text.append(possibly_none(metric.best))
        if len(metric.mean) == 0:
            row_text.append("None")
        else:
            row_text.append(possibly_none(metric.mean[-1]))
        cell_text.append(list(row_text))
    axs[-1].table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        colWidths=[0.2] * len(col_labels),
        loc="upper center",
    )

    # Save out plot.
    plt.savefig(plot_path)
    plt.close()
