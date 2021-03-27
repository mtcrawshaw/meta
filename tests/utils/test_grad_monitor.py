"""
Unit tests for meta/utils/grad_monitor.py.
"""

from math import sqrt
from itertools import product
from typing import Dict, Any

import numpy as np
import torch
from gym.spaces import Box

from meta.networks.mlp import MLPNetwork
from meta.utils.grad_monitor import GradMonitor
from tests.helpers import DEFAULT_SETTINGS


SETTINGS = {
    "obs_dim": 4,
    "num_processes": 4,
    "num_tasks": 5,
    "metric": "sqeuclidean",
    "num_layers": 3,
    "num_steps": 20,
    "ema_alpha": 0.9,
    "device": torch.device("cpu"),
}

TOL = 1e-4


def test_task_grad_diffs_zero() -> None:
    """
    Test that `get_task_grad_diffs()` correctly computes the pairwise difference between
    task-specific gradients at each layer when these gradients are hard-coded to zero.
    """

    grad_diffs_template(SETTINGS, "zero")


def test_task_grad_diffs_rand_zero() -> None:
    """
    Test that `get_task_grad_diffs()` correctly computes the pairwise difference between
    task-specific gradients at each layer when these gradients are random, while some
    tasks randomly have gradients set to zero.
    """

    grad_diffs_template(SETTINGS, "rand_zero")


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


def test_task_grad_diffs_zero_cosine() -> None:
    """
    Test that `get_task_grad_diffs()` correctly computes the pairwise cosine difference
    between task-specific gradients at each layer when these gradients are hard-coded to
    zero.
    """

    settings = dict(SETTINGS)
    settings["metric"] = "cosine"
    grad_diffs_template(settings, "zero")


def test_task_grad_diffs_rand_zero_cosine() -> None:
    """
    Test that `get_task_grad_diffs()` correctly computes the pairwise cosine difference
    between task-specific gradients at each layer when these gradients are random, while
    some tasks randomly have gradients set to zero.
    """

    settings = dict(SETTINGS)
    settings["metric"] = "cosine"
    grad_diffs_template(settings, "rand_zero")


def test_task_grad_diffs_rand_identical_cosine() -> None:
    """
    Test that `get_task_grad_diffs()` correctly computes the pairwise cosine difference
    between task-specific gradients at each layer when these gradients are random, but
    identical across tasks.
    """

    settings = dict(SETTINGS)
    settings["metric"] = "cosine"
    grad_diffs_template(settings, "rand_identical")


def test_task_grad_diffs_rand_cosine() -> None:
    """
    Test that `get_task_grad_diffs()` correctly computes the pairwise cosine difference
    between task-specific gradients at each layer when these gradients are random.
    """

    settings = dict(SETTINGS)
    settings["metric"] = "cosine"
    grad_diffs_template(settings, "rand")


def test_task_grad_stats_zero() -> None:
    """
    Test that `update_grad_stats()` correctly computes gradient statistics over multiple
    steps when the gradients are always zero.
    """

    grad_stats_template(SETTINGS, "zero")


def test_task_grad_stats_rand_zero() -> None:
    """
    Test that `update_grad_stats()` correctly computes gradient statistics over multiple
    steps when the gradients are random, while some tasks randomly have gradients set to
    zero.
    """

    grad_stats_template(SETTINGS, "rand_zero")


def test_task_grad_stats_rand_identical() -> None:
    """
    Test that `update_grad_stats()` correctly computes gradient statistics over multiple
    steps when these gradients are random, but identical across tasks.
    """

    grad_stats_template(SETTINGS, "rand_identical")


def test_task_grad_stats_rand() -> None:
    """
    Test that `update_grad_stats()` correctly computes gradient statistics over multiple
    steps when these gradients are random.
    """

    grad_stats_template(SETTINGS, "rand")


def test_task_grad_stats_zero_cosine() -> None:
    """
    Test that `update_grad_stats()` correctly computes gradient statistics over multiple
    steps when the gradients are always zero, with a cosine metric.
    """

    settings = dict(SETTINGS)
    settings["metric"] = "cosine"
    grad_stats_template(settings, "zero")


def test_task_grad_stats_rand_zero_cosine() -> None:
    """
    Test that `update_grad_stats()` correctly computes gradient statistics over multiple
    steps when the gradients are random, while some tasks randomly have gradients set to
    zero, with a cosine metric.
    """

    settings = dict(SETTINGS)
    settings["metric"] = "cosine"
    grad_stats_template(settings, "rand_zero")


def test_task_grad_stats_rand_identical_cosine() -> None:
    """
    Test that `update_grad_stats()` correctly computes gradient statistics over multiple
    steps when these gradients are random, but identical across tasks, with a cosine
    metric.
    """

    settings = dict(SETTINGS)
    settings["metric"] = "cosine"
    grad_stats_template(settings, "rand_identical")


def test_task_grad_stats_rand_cosine() -> None:
    """
    Test that `update_grad_stats()` correctly computes gradient statistics over multiple
    steps when these gradients are random, with a cosine metric.
    """

    settings = dict(SETTINGS)
    settings["metric"] = "cosine"
    grad_stats_template(settings, "rand")


def grad_diffs_template(settings: Dict[str, Any], grad_type: str) -> None:
    """ Template to test pairwise differences between task gradients. """

    # Set up case.
    dim = settings["obs_dim"] + settings["num_tasks"]
    observation_subspace = Box(low=-np.inf, high=np.inf, shape=(settings["obs_dim"],))
    observation_subspace.seed(DEFAULT_SETTINGS["seed"])

    # Construct network and gradient monitor.
    network = MLPNetwork(
        input_size=dim,
        output_size=dim,
        num_layers=settings["num_layers"],
        hidden_size=dim,
        device=settings["device"],
    )
    monitor = GradMonitor(network, settings["num_tasks"], metric=settings["metric"])

    # Construct dummy task gradients.
    task_grads = get_task_gradients(
        grad_type, monitor.num_tasks, monitor.num_layers, monitor.max_layer_size
    )

    # Compute pairwise differences of task gradients.
    task_grad_diffs = monitor.get_task_grad_diffs(task_grads)

    # Check computed differences.
    for task1, task2 in product(range(monitor.num_tasks), range(monitor.num_tasks)):
        for layer in range(monitor.num_layers):
            if settings["metric"] == "sqeuclidean":
                expected = torch.sum(
                    torch.pow(task_grads[task1, layer] - task_grads[task2, layer], 2)
                )
            else:
                grad1 = task_grads[task1, layer] / sqrt(
                    torch.sum(task_grads[task1, layer] ** 2)
                )
                grad2 = task_grads[task2, layer] / sqrt(
                    torch.sum(task_grads[task2, layer] ** 2)
                )
                if (
                    torch.isinf(grad1)
                    + torch.isnan(grad1)
                    + torch.isinf(grad2)
                    + torch.isnan(grad2)
                ).any():
                    expected = torch.Tensor([0.0])
                else:
                    expected = torch.sum(grad1 * grad2)

            assert torch.allclose(task_grad_diffs[task1, task2, layer], expected)


def grad_stats_template(settings: Dict[str, Any], grad_type: str) -> None:
    """
    Test that `update_grad_stats()` correctly computes pairwise differences of gradients
    over multiple steps.
    """

    dim = settings["obs_dim"] + settings["num_tasks"]

    # Construct network and gradient monitor.
    network = MLPNetwork(
        input_size=dim,
        output_size=dim,
        num_layers=settings["num_layers"],
        hidden_size=dim,
        device=settings["device"],
    )
    monitor = GradMonitor(
        network,
        settings["num_tasks"],
        ema_alpha=settings["ema_alpha"],
        metric=settings["metric"],
    )
    ema_threshold = monitor.grad_stats.ema_threshold

    # Update the monitor's gradient statistics with dummy task gradients,
    task_grads = torch.zeros(
        settings["num_steps"],
        monitor.num_tasks,
        monitor.num_layers,
        monitor.max_layer_size,
    )
    task_flags = torch.zeros(settings["num_steps"], monitor.num_tasks)
    task_pair_flags = torch.zeros(
        settings["num_steps"], monitor.num_tasks, monitor.num_tasks
    )
    for step in range(settings["num_steps"]):

        # Construct task gradients and update monitor.
        task_grads[step] = get_task_gradients(
            grad_type, monitor.num_tasks, monitor.num_layers, monitor.max_layer_size
        )
        monitor.update_grad_stats(task_grads=task_grads[step])

        # Set task flags, i.e. indicators for whether or not each task is included in
        # the current batch, and compute sample sizes for each task and task pair.
        task_flags[step] = torch.any(
            task_grads[step].view(monitor.num_tasks, -1) != 0, dim=1
        )
        task_flags[step] = task_flags[step] * 1
        task_pair_flags[step] = task_flags[step].unsqueeze(0) * task_flags[
            step
        ].unsqueeze(1)
        sample_sizes = torch.sum(task_flags[: step + 1], dim=0)
        pair_sample_sizes = torch.sum(task_pair_flags[: step + 1], dim=0)

        def grad_diff(grad1: torch.Tensor, grad2: torch.Tensor) -> float:
            """ Compute diff between two gradients based on `metric`. """

            if settings["metric"] == "sqeuclidean":
                diff = torch.sum((grad1 - grad2) ** 2)
            elif settings["metric"] == "cosine":
                grad1 /= sqrt(torch.sum(grad1 ** 2))
                grad2 /= sqrt(torch.sum(grad2 ** 2))
                diff = torch.sum(grad1 * grad2)
            else:
                raise NotImplementedError

            return diff

        # Compare monitor's gradients stats to the expected value for each `(task1,
        # task2, layer)`.
        for task1, task2, layer in product(
            range(monitor.num_tasks),
            range(monitor.num_tasks),
            range(monitor.num_layers),
        ):
            layer_size = int(monitor.layer_sizes[layer])

            # Computed the expected value of the mean of gradient differences between
            # `task1, task2` at layer `layer`.
            steps = task_pair_flags[:, task1, task2].bool()
            if not torch.any(steps):
                continue
            task1_grads = task_grads[steps, task1, layer, :layer_size]
            task2_grads = task_grads[steps, task2, layer, :layer_size]
            num_steps = len(steps.nonzero())
            if pair_sample_sizes[task1, task2] <= ema_threshold:
                diffs = torch.Tensor(
                    [
                        grad_diff(task1_grads[i], task2_grads[i])
                        for i in range(num_steps)
                    ]
                )
                exp_mean = torch.mean(diffs)
            else:
                initial_task1_grads = task1_grads[:ema_threshold]
                initial_task2_grads = task2_grads[:ema_threshold]
                diffs = torch.Tensor(
                    [
                        grad_diff(initial_task1_grads[i], initial_task2_grads[i])
                        for i in range(ema_threshold)
                    ]
                )
                exp_mean = torch.mean(diffs)
                for i in range(ema_threshold, int(pair_sample_sizes[task1, task2])):
                    task1_grad = task1_grads[i]
                    task2_grad = task2_grads[i]
                    diff = grad_diff(task1_grad, task2_grad)
                    exp_mean = exp_mean * settings["ema_alpha"] + diff * (
                        1.0 - settings["ema_alpha"]
                    )

            # Compare expected mean to monitor's mean.
            actual_mean = monitor.grad_stats.mean[task1, task2, layer]
            assert abs(actual_mean - exp_mean) < TOL


def get_task_gradients(
    grad_type: str, num_tasks: int, num_layers: int, max_layer_size: int
) -> torch.Tensor:
    """ Construct dummy task gradients. """

    if grad_type == "zero":
        task_grads = torch.zeros(num_tasks, num_layers, max_layer_size)
    elif grad_type == "rand_zero":
        task_grads = torch.rand(num_tasks, num_layers, max_layer_size)
        task_grads *= (torch.rand(num_tasks) < 0.5).unsqueeze(-1).unsqueeze(-1)
    elif grad_type == "rand_identical":
        task_grads = torch.rand(1, num_layers, max_layer_size)
        task_grads = task_grads.expand(num_tasks, -1, -1)
    elif grad_type == "rand":
        task_grads = torch.rand(num_tasks, num_layers, max_layer_size)
    else:
        raise NotImplementedError

    return task_grads
