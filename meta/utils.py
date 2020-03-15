import glob
import os
from functools import reduce
from typing import Dict
import pickle

import torch.nn as nn
from gym.spaces import Space, Box, Discrete


METRICS_DIR = os.path.join("data", "metrics")


# Get a render function
def get_render_func(venv):
    if hasattr(venv, "envs"):
        return venv.envs[0].render
    elif hasattr(venv, "venv"):
        return get_render_func(venv.venv)
    elif hasattr(venv, "env"):
        return get_render_func(venv.env)

    return None


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, "*.monitor.csv"))
        for f in files:
            os.remove(f)


def get_space_size(space: Space):
    """ Get the input/output size of an MLP whose input/output space is ``space``. """

    if isinstance(space, Discrete):
        size = space.n
    elif isinstance(space, Box):
        size = reduce(lambda a, b: a * b, space.shape)
    else:
        raise ValueError("Unsupported space type: %s." % type(space))

    return size


def compare_output_metrics(output_metrics: Dict[str, float], metrics_filename: str):
    """ Compare output_metrics against the most recently saved baseline. """

    # Load baseline metric values.
    with open(metrics_filename, "rb") as metrics_file:
        baseline_output_metrics = pickle.load(metrics_file)

    # Compare output_metrics against baseline.
    same_as_baseline = True
    same_as_baseline = same_as_baseline and set(output_metrics.keys()) == set(baseline_output_metrics.keys())
    for metric_name in baseline_output_metrics.keys():
        same_as_baseline = same_as_baseline and baseline_output_metrics[metric_name] == output_metrics[metric_name]

    return same_as_baseline
