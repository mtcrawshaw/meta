""" Plotting for performance metrics. """

from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt


def plot(metrics_state: Dict[str, Dict[str, List[float]]], plot_path: str) -> None:
    """
    Plot the metrics given in ``metrics_history``, store image at ``plot_path``.
    """

    # Create a subplot for each metric. We only plot metrics whose corresponding lists
    # are nonempty, and we replace a singleton subplot with a list of a single subplot,
    # so that we can access it by index. The +1 on num_plots is done so that we have an
    # extra subplot on the end which will just contain a table.
    num_plots = 0
    for metric_name, metric_state in metrics_state.items():
        if len(metric_state["history"]) > 0:
            num_plots += 1
    num_plots += 1
    fig_width = 12.8
    plot_height = 4.8
    table_height_ratio = 0.2
    fig, axs = plt.subplots(
        num_plots,
        figsize=(fig_width, plot_height * (num_plots - (1.0 - table_height_ratio))),
        gridspec_kw={"height_ratios": [1] * (num_plots - 1) + [table_height_ratio]},
    )
    if not isinstance(axs, np.ndarray):
        axs = [axs]
    fig.suptitle("Training Metrics")

    # Plot each metric on a separate plot.
    plot_num = 0
    plotted_metrics = []
    for metric_name, metric_state in metrics_state.items():

        mean_array = np.array(metric_state["mean"])
        stdev_array = np.array(metric_state["stdev"])
        assert len(mean_array) == len(stdev_array)
        legend = []

        # Make sure that metric history is nonempty.
        if len(mean_array) == 0:
            continue
        plotted_metrics.append(metric_name)

        # Plot mean.
        axs[plot_num].plot(np.arange(len(mean_array)), mean_array)
        legend.append("%s mean" % metric_name)

        # Plot mean + stdev and mean - stdev.
        upper_dev_array = mean_array + stdev_array
        lower_dev_array = mean_array - stdev_array
        axs[plot_num].plot(np.arange(len(mean_array)), upper_dev_array)
        axs[plot_num].plot(np.arange(len(mean_array)), lower_dev_array)
        legend.append("%s mean + stdev" % metric_name)
        legend.append("%s mean - stdev" % metric_name)

        # Add legend to subplot.
        axs[plot_num].legend(legend, loc="upper left")

        plot_num += 1

    # Write out table of final metrics.
    axs[-1].axis("off")
    row_labels = list(plotted_metrics)
    col_labels = ["Evaluation mean", "Training max"]
    cell_text = []
    for metric_name in plotted_metrics:
        metric_state = metrics_state[metric_name]
        row_text = []
        row_text.append("%.5f" % metric_state["final"])
        row_text.append("%.5f" % metric_state["maximum"])
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
