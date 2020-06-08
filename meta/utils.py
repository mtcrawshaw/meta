""" Utility functions and objects for training pipeline. """

import os
from functools import reduce
from typing import Dict, List, Union, Any
import pickle

import torch
import torch.nn as nn
from gym.spaces import Space, Box, Discrete


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

    # This is a somewhat gross way to handle both Linear/Conv modules and GRU modules.
    # It can probably be cleaned up.
    if hasattr(module, "weight") and hasattr(module, "bias"):
        weight_init(module.weight.data, gain=gain)
        bias_init(module.bias.data)
    else:
        for name, param in module.named_parameters():
            if "weight" in name:
                weight_init(param)
            elif "bias" in name:
                bias_init(param)

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
    diff: Dict[str, List[Any]] = {key: [] for key in metrics}
    for key in set(metrics.keys()).intersection(set(baseline_metrics.keys())):
        for i in range(min(len(metrics[key]), len(baseline_metrics[key]))):

            if i >= len(metrics[key]):
                diff[key].append((i, None, baseline_metrics[key][i]))
            if i >= len(baseline_metrics[key]):
                diff[key].append((i, metrics[key][i], None))

            current_val = metrics[key][i]
            baseline_val = baseline_metrics[key][i]
            if current_val != baseline_val:
                diff[key].append((i, current_val, baseline_val))

    print("Metrics diff: %s" % diff)
    assert set(metrics.keys()) == set(baseline_metrics.keys())
    for key in metrics:
        assert len(metrics[key]) == len(baseline_metrics[key])
    assert all(len(diff_values) == 0 for diff_values in diff.values())


def combine_first_two_dims(t: torch.Tensor):
    """ Flattens the first two dimensions of ``t`` into a single dimension. """

    if len(t.shape) < 2:
        raise ValueError(
            "Can't combine first two dimensions of tensor which has less than two "
            "dimensions: %s"
            % t
        )

    if len(t.shape) == 2:
        return t.view(t.shape[0] * t.shape[1])
    else:
        return t.view(t.shape[0] * t.shape[1], *t.shape[2:])
