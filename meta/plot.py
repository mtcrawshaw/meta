""" Plotting for performance metrics. """

import os
from functools import reduce
from typing import Dict, List, Union, Any, Tuple
import pickle

import torch
import torch.nn as nn
from gym.spaces import Space, Box, Discrete
import numpy as np
import matplotlib.pyplot as plt


def plot(metrics_history: Dict[str, List[float]], plot_path: str) -> None:
    """
    Plot the metrics given in ``metrics_history``, store image at ``plot_path``.
    """

    # Plot each metric on a separate curve.
    legend = []
    for metric_name, metric_history in metrics_history.items():
        plt.plot(np.arange(len(metric_history)), np.array(metric_history))
        legend.append(metric_name)
    plt.legend(legend, loc="upper right")

    # Save out plot.
    plt.savefig(plot_path)
