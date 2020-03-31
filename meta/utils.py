import os
import pickle
import copy
from functools import reduce
from typing import Union, List, Dict

import numpy as np
import torch
import torch.nn as nn
import gym
from gym import Env
from gym.spaces import Space, Box, Discrete
from baselines import bench

from meta.tests.envs import ParityEnv, UniqueEnv


METRICS_DIR = os.path.join("data", "metrics")


# Hacky fix for Gaussian policies.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias)

    def forward(self, x):
        return x + self._bias


def convert_to_tensor(val: Union[np.ndarray, int, float]):
    """
    Converts a value (observation or action) from environment to a tensor.

    Arguments
    ---------
    val: np.ndarray or int
        Observation or action returned from the environment.
    """

    if isinstance(val, int) or isinstance(val, float):
        return torch.Tensor([val])
    elif isinstance(val, np.ndarray):
        return torch.Tensor(val)
    elif isinstance(val, torch.Tensor):
        return val
    else:
        raise ValueError(
            "Cannot convert value of type '%r' to torch.Tensor." % type(val)
        )


def init(module, weight_init, bias_init, gain=1):
    """ Helper function it initialize network weights. """

    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def get_space_size(space: Space):
    """ Get the input/output size of an MLP whose input/output space is ``space``. """

    if isinstance(space, Discrete):
        size = space.n
    elif isinstance(space, Box):
        size = reduce(lambda a, b: a * b, space.shape)
    else:
        raise ValueError("Unsupported space type: %s." % type(space))

    return size


def compare_metrics(metrics: Dict[str, List[float]], metrics_filename: str):
    """ Compute diff of metrics against the most recently saved baseline. """

    # Load baseline metric values.
    with open(metrics_filename, "rb") as metrics_file:
        baseline_metrics = pickle.load(metrics_file)

    # Compare metrics against baseline.
    assert set(metrics.keys()) == set(baseline_metrics.keys())
    diff = {key: [] for key in metrics}
    for key in metrics:
        assert len(metrics[key]) == len(baseline_metrics[key])

        for i in range(len(metrics[key])):
            current_val = metrics[key][i]
            baseline_val = baseline_metrics[key][i]
            if current_val != baseline_val:
                diff[key].append((i, current_val, baseline_val))

    same = all(len(diff_values) == 0 for diff_values in diff.values())
    return diff, same


def get_env(env_name: str, seed: int) -> Env:
    """ Return environment object from environment name. """

    metaworld_env_names = get_metaworld_env_names()
    if env_name in metaworld_env_names:

        # We import here so that we avoid importing metaworld if possible, since it is
        # dependent on mujoco.
        from metaworld.benchmarks import ML1

        env = ML1.get_train_tasks(env_name)
        tasks = env.sample_tasks(1)
        env.set_task(tasks[0])

    elif env_name == "parity-env":
        env = ParityEnv()

    elif env_name == "unique-env":
        env = UniqueEnv()

    else:
        env = gym.make(env_name)

    # Set environment seed.
    env.seed(seed)

    # Add environment wrappers.
    env = bench.Monitor(env, None)

    return env


def get_metaworld_env_names() -> List[str]:
    """ Returns a list of Metaworld environment names. """

    return HARD_MODE_CLS_DICT["train"] + HARD_MODE_CLS_DICT["test"]


# HARDCODE. This is copied from the metaworld repo to avoid the need to import metaworld
# unnencessarily. Since it relies on mujoco, we don't want to import it if we don't have
# to.
HARD_MODE_CLS_DICT = {
    "train": [
        "reach-v1",
        "push-v1",
        "pick-place-v1",
        "reach-wall-v1",
        "pick-place-wall-v1",
        "push-wall-v1",
        "door-open-v1",
        "door-close-v1",
        "drawer-open-v1",
        "drawer-close-v1",
        "button-press_topdown-v1",
        "button-press-v1",
        "button-press-topdown-wall-v1",
        "button-press-wall-v1",
        "peg-insert-side-v1",
        "peg-unplug-side-v1",
        "window-open-v1",
        "window-close-v1",
        "dissassemble-v1",
        "hammer-v1",
        "plate-slide-v1",
        "plate-slide-side-v1",
        "plate-slide-back-v1",
        "plate-slide-back-side-v1",
        "handle-press-v1",
        "handle-pull-v1",
        "handle-press-side-v1",
        "handle-pull-side-v1",
        "stick-push-v1",
        "stick-pull-v1",
        "basket-ball-v1",
        "soccer-v1",
        "faucet-open-v1",
        "faucet-close-v1",
        "coffee-push-v1",
        "coffee-pull-v1",
        "coffee-button-v1",
        "sweep-v1",
        "sweep-into-v1",
        "pick-out-of-hole-v1",
        "assembly-v1",
        "shelf-place-v1",
        "push-back-v1",
        "lever-pull-v1",
        "dial-turn-v1",
    ],
    "test": [
        "bin-picking-v1",
        "box-close-v1",
        "hand-insert-v1",
        "door-lock-v1",
        "door-unlock-v1",
    ],
}
