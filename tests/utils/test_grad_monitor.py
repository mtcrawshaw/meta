"""
Unit tests for meta/utils/grad_monitor.py.
"""

from typing import Dict, Any
from itertools import product

import numpy as np
import torch
from gym.spaces import Box

from meta.networks.mlp import MLPNetwork
from meta.utils.grad_monitor import GradMonitor
from tests.helpers import DEFAULT_SETTINGS


SETTINGS = {
    "obs_dim": 8,
    "num_processes": 8,
    "num_tasks": 4,
    "num_layers": 3,
    "device": torch.device("cpu"),
}


def test_task_grad_diffs_zero() -> None:
    """
    Test that `get_task_grad_diffs()` correctly computes the pairwise difference between
    task-specific gradients at each layer when these gradients are hard-coded to zero.
    """

    grad_diffs_template(SETTINGS, "zero")


def test_task_grad_diffs_rand_identical() -> None:
    """
    Test that `get_task_grad_diffs()` correctly computes the pairwise difference between
    task-specific gradients at each layer when these gradients are random, but
    identical across tasks.
    """

    grad_diffs_template(SETTINGS, "rand_identical")


def test_task_grad_diffs_rand() -> None:
    """
    Test that `get_task_grad_diffs()` correctly computes the pairwise difference between
    task-specific gradients at each layer when these gradients are random.
    """

    grad_diffs_template(SETTINGS, "rand")


def grad_diffs_template(settings: Dict[str, Any], grad_type: str) -> None:
    """ Template to test pairwise differences between task gradients. """

    # Set up case.
    dim = settings["obs_dim"] + settings["num_tasks"]
    observation_subspace = Box(low=-np.inf, high=np.inf, shape=(settings["obs_dim"],))
    observation_subspace.seed(DEFAULT_SETTINGS["seed"])
    hidden_size = dim

    # Construct network and gradient monitor.
    network = MLPNetwork(
        input_size=dim,
        output_size=dim,
        num_layers=settings["num_layers"],
        hidden_size=hidden_size,
        device=settings["device"],
    )
    monitor = GradMonitor(network, settings["num_tasks"])

    # Construct dummy task gradients.
    if grad_type == "zero":
        task_grads = torch.zeros(
            monitor.num_tasks, monitor.num_layers, monitor.max_layer_size
        )
    elif grad_type == "rand_identical":
        task_grads = torch.rand(1, monitor.num_layers, monitor.max_layer_size)
        task_grads = task_grads.expand(monitor.num_tasks, -1, -1)
    elif grad_type == "rand":
        task_grads = torch.rand(
            monitor.num_tasks, monitor.num_layers, monitor.max_layer_size
        )
    else:
        raise NotImplementedError

    # Compute pairwise differences of task gradients.
    task_grad_diffs = monitor.get_task_grad_diffs(task_grads)

    # Check computed differences.
    for task1, task2 in product(range(monitor.num_tasks), range(monitor.num_tasks)):
        for layer in range(monitor.num_layers):
            expected_diff = torch.sum(
                torch.pow(task_grads[task1, layer] - task_grads[task2, layer], 2)
            )
            assert torch.allclose(task_grad_diffs[task1, task2, layer], expected_diff)
