""" Plotting for performance metrics. """

from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt


def plot(metrics_state: Dict[str, Dict[str, List[float]]], plot_path: str) -> None:
    """
    Plot the metrics given in ``metrics_history``, store image at ``plot_path``.
    """

    # Plot each metric on a separate curve.
    legend = []
    for metric_name, metric_state in metrics_state.items():

        mean_array = np.array(metric_state["mean"])
        stdev_array = np.array(metric_state["stdev"])
        assert len(mean_array) == len(stdev_array)

        # Plot mean.
        plt.plot(np.arange(len(mean_array)), mean_array)
        legend.append("%s mean" % metric_name)

        # Plot mean + stdev and mean - stdev.
        upper_dev_array = mean_array + stdev_array
        lower_dev_array = mean_array - stdev_array
        plt.plot(np.arange(len(mean_array)), upper_dev_array)
        plt.plot(np.arange(len(mean_array)), lower_dev_array)
        legend.append("%s mean + stdev" % metric_name)
        legend.append("%s mean - stdev" % metric_name)

    plt.legend(legend, loc="upper right")

    # Save out plot.
    plt.savefig(plot_path)
