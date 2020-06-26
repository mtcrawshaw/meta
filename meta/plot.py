""" Plotting for performance metrics. """

from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt


def plot(metrics_state: Dict[str, Dict[str, List[float]]], plot_path: str) -> None:
    """
    Plot the metrics given in ``metrics_history``, store image at ``plot_path``.
    """

    # Count the number of plots. We create a plot for each (train, eval) pair of
    # metrics, and we don't plot any metrics whose histories are empty.
    basenames = []
    for metric_name, metric_state in metrics_state.items():
        if len(metric_state["history"]) > 0:
            if metric_name.startswith("train_") or metric_name.startswith("eval_"):
                start_pos = metric_name.index("_") + 1
                basename = metric_name[start_pos:]
                if basename not in basenames:
                    basenames.append(basename)
            else:
                raise ValueError(
                    "Can only plot metrics with names of the form 'train_*' or 'eval_*'"
                )

    # Add an extra plot which will just contain a table.
    num_plots = len(basenames) + 1

    # Create subplots.
    fig_width = 12.8
    plot_height = 4.8
    table_height_ratio = 0.2
    fig, axs = plt.subplots(
        num_plots,
        figsize=(fig_width, plot_height * (num_plots - (1.0 - table_height_ratio))),
        gridspec_kw={"height_ratios": [1] * (num_plots - 1) + [table_height_ratio]},
    )

    # Wrap a single axis in a list to make sure that axs is iterable.
    if not isinstance(axs, np.ndarray):
        axs = [axs]

    # Create figure title.
    fig.suptitle("Training Metrics")

    # Plot each metric on a separate plot.
    plotted_metrics = []
    for i, basename in enumerate(basenames):

        # Construct metric names and find max metric length (scale of x-axis).
        legend = []
        splits = ["train", "eval"]
        metric_names = ["%s_%s" % (split, basename) for split in splits]
        max_metric_len = max(
            [len(metrics_state[metric_name]["mean"]) for metric_name in metric_names]
        )

        for metric_name in metric_names:

            # Get metric state values.
            metric_state = metrics_state[metric_name]
            mean_array = np.array(metric_state["mean"])
            stdev_array = np.array(metric_state["stdev"])
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
    col_labels = ["Maximum", "Final"]
    cell_text = []
    for metric_name in plotted_metrics:
        metric_state = metrics_state[metric_name]
        row_text = []
        row_text.append(possibly_none(metric_state["maximum"]))
        if len(metric_state["mean"]) == 0:
            row_text.append("None")
        else:
            row_text.append(possibly_none(metric_state["mean"][-1]))
        cell_text.append(list(row_text))
    axs[-1].table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        colWidths=[0.2] * len(col_labels),
        loc="bottom",
    )

    # Save out plot.
    plt.savefig(plot_path)
