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
    # so that we can access it by index.
    num_plots = 0
    for metric_name, metric_state in metrics_state.items():
        if len(metric_state["history"]) > 0:
            num_plots += 1
    fig, axs = plt.subplots(num_plots, figsize=(12.8, 4.8 * num_plots))
    if not isinstance(axs, np.ndarray):
        axs = [axs]
    fig.suptitle("Training Metrics")

    # Plot each metric on a separate curve.
    plot_num = 0
    for metric_name, metric_state in metrics_state.items():

        mean_array = np.array(metric_state["mean"])
        stdev_array = np.array(metric_state["stdev"])
        assert len(mean_array) == len(stdev_array)
        legend = []

        # Make sure that metric history is nonempty.
        if len(mean_array) == 0:
            continue

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

    # Save out plot.
    plt.savefig(plot_path)
