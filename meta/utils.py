""" Utility functions and objects for training pipeline. """

import os
from functools import reduce
from gym.spaces import Space, Box, Discrete
from typing import Dict, List, Union, Any
import pickle

import torch
import torch.nn as nn


METRICS_DIR = os.path.join("data", "metrics")


class AddBias(nn.Module):
    """ Hacky fix for Gaussian policies. """

    def __init__(self, bias: torch.Tensor) -> None:
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self._bias


def init(
    module: nn.Module, weight_init: Any, bias_init: Any, gain: Union[float, int] = 1
) -> nn.Module:
    """ Helper function to initialize network weights. """

    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def get_space_size(space: Space) -> int:
    """ Get the input/output size of an MLP whose input/output space is ``space``. """

    size: int = 0
    if isinstance(space, Discrete):
        size = space.n
    elif isinstance(space, Box):
        size = reduce(lambda a, b: a * b, space.shape)
    else:
        raise ValueError("Unsupported space type: %s." % type(space))

    return size


def compare_metrics(metrics: Dict[str, List[float]], metrics_filename: str) -> None:
    """ Compute diff of metrics against the most recently saved baseline. """

    # Load baseline metric values.
    with open(metrics_filename, "rb") as metrics_file:
        baseline_metrics = pickle.load(metrics_file)

    # Compare metrics against baseline.
    assert set(metrics.keys()) == set(baseline_metrics.keys())
    diff: Dict[str, List[Any]] = {key: [] for key in metrics}
    for key in metrics:
        assert len(metrics[key]) == len(baseline_metrics[key])

        for i in range(max(len(metrics[key]), len(baseline_metrics[key]))):

            if i >= len(metrics[key]):
                diff[key].append((i, None, baseline_metrics[key][i]))
            if i >= len(baseline_metrics[key]):
                diff[key].append((i, metrics[key][i], None))

            current_val = metrics[key][i]
            baseline_val = baseline_metrics[key][i]
            if current_val != baseline_val:
                diff[key].append((i, current_val, baseline_val))

    print("Metrics diff: %s" % diff)
    assert all(len(diff_values) == 0 for diff_values in diff.values())
