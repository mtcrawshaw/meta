""" Plotting for performance metrics. """

from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt


def plot(metrics_state: Dict[str, Dict[str, List[float]]], plot_path: str) -> None:
    """
    Plot the metrics given in ``metrics_history``, store image at ``plot_path``.
    """

    # Create a subplot for each metric.
    fig, axs = plt.subplots(len(metrics_state), figsize=(12.8, 9.6))
    fig.suptitle("Training Metrics")

    # Plot each metric on a separate curve.
    for i, (metric_name, metric_state) in enumerate(metrics_state.items()):

        mean_array = np.array(metric_state["mean"])
        stdev_array = np.array(metric_state["stdev"])
        assert len(mean_array) == len(stdev_array)
        legend = []

        # Plot mean.
        axs[i].plot(np.arange(len(mean_array)), mean_array)
        legend.append("%s mean" % metric_name)

        # Plot mean + stdev and mean - stdev.
        upper_dev_array = mean_array + stdev_array
        lower_dev_array = mean_array - stdev_array
        axs[i].plot(np.arange(len(mean_array)), upper_dev_array)
        axs[i].plot(np.arange(len(mean_array)), lower_dev_array)
        legend.append("%s mean + stdev" % metric_name)
        legend.append("%s mean - stdev" % metric_name)

        # Add legend to subplot.
        axs[i].legend(legend, loc="upper right")

    # Save out plot.
    plt.savefig(plot_path)
