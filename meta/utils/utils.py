""" Utility functions and objects for training pipeline. """

import os
from functools import reduce
from typing import Dict, List, Any, Tuple
import pickle

import torch
import torch.nn as nn
from gym.spaces import Space, Box, Discrete


METRICS_DIR = os.path.join("data", "metrics")
RESULTS_DIR = os.path.join("results")
DATA_DIR = os.path.join("data", "datasets")


class AddBias(nn.Module):
    """ Hacky fix for Gaussian policies. """

    def __init__(self, bias: torch.Tensor) -> None:
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self._bias


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


def get_space_shape(space: Space, space_name: str) -> Tuple[int, ...]:
    """ Get the tensor shape of a sample from ``space``. """

    shape: Tuple[int, ...] = (-1,)

    if isinstance(space, Discrete):
        if space_name == "obs":
            shape = (space.n,)
        elif space_name == "action":
            shape = (1,)
        else:
            raise ValueError("Unrecognized space '%s'." % space_name)
    elif isinstance(space, Box):
        shape = space.shape
    else:
        raise ValueError("'%r' not a supported %s space." % (type(space), space_name))

    return shape


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


def combine_first_two_dims(t: torch.Tensor) -> torch.Tensor:
    """ Flattens the first two dimensions of ``t`` into a single dimension. """

    if len(t.shape) < 2:
        raise ValueError(
            "Can't combine first two dimensions of tensor which has less than two "
            "dimensions: %s" % t
        )

    return t.view(t.shape[0] * t.shape[1], *t.shape[2:])


def save_dir_from_name(name: str) -> str:
    """
    Return the name of the directory to store results of training run with name
    ``name``.
    """
    return os.path.join(RESULTS_DIR, name)


def aligned_train_configs(config1: Dict[str, Any], config2: Dict[str, Any]) -> bool:
    """
    Check whether two training configurations are aligned, i.e. a training run with
    ``config1`` that was interrupted can be resumed with ``config2``.
    """

    aligned_settings = [
        "env_name",
        "lr_schedule_type",
        "initial_lr",
        "final_lr",
        "normalize_transition",
        "normalize_first_n",
        "architecture_config",
        "evaluation_freq",
        "evaluation_episodes",
        "time_limit",
    ]
    equal = True
    for setting in aligned_settings:
        equal = equal and config1[setting] == config2[setting]

    return equal


def aligned_tune_configs(config1: Dict[str, Any], config2: Dict[str, Any]) -> bool:
    """
    Check whether two tune configurations are aligned, i.e. a hyperparameter search run
    with ``config1`` that was interrupted can be resumed with ``config2``.
    """

    aligned_settings = [
        "search_type",
        "trials_per_config",
        "search_params",
        "fitness_metric_name",
        "fitness_metric_type",
    ]
    equal = True
    for setting in aligned_settings:
        equal = equal and config1[setting] == config2[setting]
    equal = equal and aligned_train_configs(
        config1["base_train_config"], config2["base_train_config"]
    )

    return equal
