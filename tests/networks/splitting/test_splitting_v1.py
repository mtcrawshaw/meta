"""
Unit tests for meta/networks/splitting/splitting_v1.py.
"""

import math
import random
from itertools import product
from typing import Dict, Any, List

import numpy as np
from scipy import stats
import torch
import torch.nn.functional as F
from gym.spaces import Box

from meta.networks.initialize import init_base
from meta.networks.splitting import MultiTaskSplittingNetworkV1
from meta.utils.estimate import alpha_to_threshold
from tests.helpers import DEFAULT_SETTINGS, get_obs_batch
from tests.networks.templates import (
    TOL,
    gradients_template,
    backward_template,
    grad_diffs_template,
    split_stats_template,
    split_template,
    score_template,
)


SETTINGS = {
    "obs_dim": 8,
    "num_processes": 8,
    "num_tasks": 4,
    "num_layers": 3,
    "split_alpha": 0.05,
    "grad_var": None,
    "split_step_threshold": 30,
    "cap_sample_size": True,
    "ema_alpha": 0.999,
    "include_task_index": True,
    "device": torch.device("cpu"),
}


def test_forward_shared() -> None:
    """
    Test forward() when all regions of the splitting network are fully shared. The
    function computed by the network should be f(x) = 3 * tanh(2 * tanh(x + 1) + 2) + 3.
    """

    # Set up case.
    dim = SETTINGS["obs_dim"] + SETTINGS["num_tasks"]
    observation_subspace = Box(low=-np.inf, high=np.inf, shape=(SETTINGS["obs_dim"],))
    observation_subspace.seed(DEFAULT_SETTINGS["seed"])
    hidden_size = dim

    # Construct network.
    network = MultiTaskSplittingNetworkV1(
        input_size=dim,
        output_size=dim,
        init_base=init_base,
        init_final=init_base,
        num_tasks=SETTINGS["num_tasks"],
        num_layers=SETTINGS["num_layers"],
        hidden_size=hidden_size,
        device=SETTINGS["device"],
    )

    # Set network weights.
    state_dict = network.state_dict()
    for i in range(SETTINGS["num_layers"]):
        weight_name = "regions.%d.0.0.weight" % i
        bias_name = "regions.%d.0.0.bias" % i
        state_dict[weight_name] = torch.Tensor((i + 1) * np.identity(dim))
        state_dict[bias_name] = torch.Tensor((i + 1) * np.ones(dim))
    network.load_state_dict(state_dict)

    # Construct batch of observations concatenated with one-hot task vectors.
    obs, task_indices = get_obs_batch(
        batch_size=SETTINGS["num_processes"],
        obs_space=observation_subspace,
        num_tasks=SETTINGS["num_tasks"],
    )

    # Get output of network.
    output = network(obs, task_indices)

    # Computed expected output of network.
    expected_output = 3 * torch.tanh(2 * torch.tanh(obs + 1) + 2) + 3

    # Test output of network.
    assert torch.allclose(output, expected_output)


def test_forward_single() -> None:
    """
    Test forward() when all regions of the splitting network are fully shared except
    one. The function computed by the network should be f(x) = 3 * tanh(2 * tanh(x + 1)
    + 2) + 3 for tasks 0 and 1 and f(x) = 3 * tanh(-2 * tanh(x + 1) - 2) + 3 for tasks 2
    and 3.
    """

    # Set up case.
    dim = SETTINGS["obs_dim"] + SETTINGS["num_tasks"]
    observation_subspace = Box(low=-np.inf, high=np.inf, shape=(SETTINGS["obs_dim"],))
    observation_subspace.seed(DEFAULT_SETTINGS["seed"])
    hidden_size = dim

    # Construct network.
    network = MultiTaskSplittingNetworkV1(
        input_size=dim,
        output_size=dim,
        init_base=init_base,
        init_final=init_base,
        num_tasks=SETTINGS["num_tasks"],
        num_layers=SETTINGS["num_layers"],
        hidden_size=hidden_size,
        device=SETTINGS["device"],
    )

    # Split the network at the second layer. Tasks 0 and 1 stay assigned to the original
    # copy and tasks 2 and 3 are assigned to the new copy.
    network.split(1, 0, [0, 1], [2, 3])

    # Set network weights.
    state_dict = network.state_dict()
    for i in range(SETTINGS["num_layers"]):
        weight_name = "regions.%d.0.0.weight" % i
        bias_name = "regions.%d.0.0.bias" % i
        state_dict[weight_name] = torch.Tensor((i + 1) * np.identity(dim))
        state_dict[bias_name] = torch.Tensor((i + 1) * np.ones(dim))
    weight_name = "regions.1.1.0.weight"
    bias_name = "regions.1.1.0.bias"
    state_dict[weight_name] = torch.Tensor(-2 * np.identity(dim))
    state_dict[bias_name] = torch.Tensor(-2 * np.ones(dim))
    network.load_state_dict(state_dict)

    # Construct batch of observations concatenated with one-hot task vectors.
    obs, task_indices = get_obs_batch(
        batch_size=SETTINGS["num_processes"],
        obs_space=observation_subspace,
        num_tasks=SETTINGS["num_tasks"],
    )

    # Get output of network.
    output = network(obs, task_indices)

    # Computed expected output of network.
    expected_output = torch.zeros(obs.shape)
    for i, (ob, task) in enumerate(zip(obs, task_indices)):
        if task in [0, 1]:
            expected_output[i] = 3 * torch.tanh(2 * torch.tanh(ob + 1) + 2) + 3
        elif task in [2, 3]:
            expected_output[i] = 3 * torch.tanh(-2 * torch.tanh(ob + 1) - 2) + 3
        else:
            raise NotImplementedError

    # Test output of network.
    assert torch.allclose(output, expected_output)


def test_forward_multiple() -> None:
    """
    Test forward() when none of the layers are fully shared. The function computed by
    the network should be:
    - f(x) = 3 * tanh(2 * tanh(x + 1) + 2) + 3 for task 0
    - f(x) = -3 * tanh(-2 * tanh(x + 1) - 2) - 3 for task 1
    - f(x) = -3 * tanh(1/2 * tanh(-x - 1) + 1/2) - 3 for task 2
    - f(x) = 3 * tanh(-2 * tanh(-x - 1) - 2) + 3 for task 3
    """

    # Set up case.
    dim = SETTINGS["obs_dim"] + SETTINGS["num_tasks"]
    observation_subspace = Box(low=-np.inf, high=np.inf, shape=(SETTINGS["obs_dim"],))
    observation_subspace.seed(DEFAULT_SETTINGS["seed"])
    hidden_size = dim

    # Construct network.
    network = MultiTaskSplittingNetworkV1(
        input_size=dim,
        output_size=dim,
        init_base=init_base,
        init_final=init_base,
        num_tasks=SETTINGS["num_tasks"],
        num_layers=SETTINGS["num_layers"],
        hidden_size=hidden_size,
        device=SETTINGS["device"],
    )

    # Split the network at the second layer. Tasks 0 and 1 stay assigned to the original
    # copy and tasks 2 and 3 are assigned to the new copy.
    network.split(0, 0, [0, 1], [2, 3])
    network.split(1, 0, [0, 2], [1, 3])
    network.split(1, 0, [0], [2])
    network.split(2, 0, [0, 3], [1, 2])

    # Set network weights.
    state_dict = network.state_dict()
    for i in range(SETTINGS["num_layers"]):
        for j in range(3):
            weight_name = "regions.%d.%d.0.weight" % (i, j)
            bias_name = "regions.%d.%d.0.bias" % (i, j)
            if weight_name not in state_dict:
                continue

            if j == 0:
                state_dict[weight_name] = torch.Tensor((i + 1) * np.identity(dim))
                state_dict[bias_name] = torch.Tensor((i + 1) * np.ones(dim))
            elif j == 1:
                state_dict[weight_name] = torch.Tensor(-(i + 1) * np.identity(dim))
                state_dict[bias_name] = torch.Tensor(-(i + 1) * np.ones(dim))
            elif j == 2:
                state_dict[weight_name] = torch.Tensor(1 / (i + 1) * np.identity(dim))
                state_dict[bias_name] = torch.Tensor(1 / (i + 1) * np.ones(dim))
            else:
                raise NotImplementedError

    network.load_state_dict(state_dict)

    # Construct batch of observations concatenated with one-hot task vectors.
    obs, task_indices = get_obs_batch(
        batch_size=SETTINGS["num_processes"],
        obs_space=observation_subspace,
        num_tasks=SETTINGS["num_tasks"],
    )

    # Get output of network.
    output = network(obs, task_indices)

    # Computed expected output of network.
    expected_output = torch.zeros(obs.shape)
    for i, (ob, task) in enumerate(zip(obs, task_indices)):
        if task == 0:
            expected_output[i] = 3 * torch.tanh(2 * torch.tanh(ob + 1) + 2) + 3
        elif task == 1:
            expected_output[i] = -3 * torch.tanh(-2 * torch.tanh(ob + 1) - 2) - 3
        elif task == 2:
            expected_output[i] = (
                -3 * torch.tanh(1 / 2 * torch.tanh(-ob - 1) + 1 / 2) - 3
            )
        elif task == 3:
            expected_output[i] = 3 * torch.tanh(-2 * torch.tanh(-ob - 1) - 2) + 3
        else:
            raise NotImplementedError

    # Test output of network.
    assert torch.allclose(output, expected_output)


def test_split_single() -> None:
    """
    Test that split() correctly sets new parameters when we perform a single split.
    """

    # Set up case.
    dim = SETTINGS["obs_dim"] + SETTINGS["num_tasks"]
    observation_subspace = Box(low=-np.inf, high=np.inf, shape=(SETTINGS["obs_dim"],))
    observation_subspace.seed(DEFAULT_SETTINGS["seed"])
    hidden_size = dim

    # Construct network.
    network = MultiTaskSplittingNetworkV1(
        input_size=dim,
        output_size=dim,
        init_base=init_base,
        init_final=init_base,
        num_tasks=SETTINGS["num_tasks"],
        num_layers=SETTINGS["num_layers"],
        hidden_size=hidden_size,
        device=SETTINGS["device"],
    )

    # Split the network at the last layer, so that tasks 0 and 2 stay assigned to the
    # original copy and tasks 1 and 3 are assigned to the new copy.
    network.split(2, 0, [0, 2], [1, 3])

    # Check the parameters of the network.
    param_names = [name for name, param in network.named_parameters()]

    # Construct expected parameters of network.
    region_copies = {i: [0] for i in range(SETTINGS["num_layers"])}
    region_copies[2].append(1)
    expected_params = []
    for region, copies in region_copies.items():
        for copy in copies:
            expected_params.append("regions.%d.%d.0.weight" % (region, copy))
            expected_params.append("regions.%d.%d.0.bias" % (region, copy))

    # Test actual parameter names.
    assert set(param_names) == set(expected_params)


def test_split_multiple() -> None:
    """
    Test that split() correctly sets new parameters when we perform multiple splits.
    """

    # Set up case.
    dim = SETTINGS["obs_dim"] + SETTINGS["num_tasks"]
    observation_subspace = Box(low=-np.inf, high=np.inf, shape=(SETTINGS["obs_dim"],))
    observation_subspace.seed(DEFAULT_SETTINGS["seed"])
    hidden_size = dim

    # Construct network.
    network = MultiTaskSplittingNetworkV1(
        input_size=dim,
        output_size=dim,
        init_base=init_base,
        init_final=init_base,
        num_tasks=SETTINGS["num_tasks"],
        num_layers=SETTINGS["num_layers"],
        hidden_size=hidden_size,
        device=SETTINGS["device"],
    )

    # Split the network at the first layer once and the last layer twice.
    network.split(0, 0, [0, 1], [2, 3])
    network.split(2, 0, [0, 2], [1, 3])
    network.split(2, 1, [1], [3])

    # Check the parameters of the network.
    param_names = [name for name, param in network.named_parameters()]

    # Construct expected parameters of network.
    region_copies = {i: [0] for i in range(SETTINGS["num_layers"])}
    region_copies[0].extend([1])
    region_copies[2].extend([1, 2])
    expected_params = []
    for region, copies in region_copies.items():
        for copy in copies:
            expected_params.append("regions.%d.%d.0.weight" % (region, copy))
            expected_params.append("regions.%d.%d.0.bias" % (region, copy))

    # Test actual parameter names.
    assert set(param_names) == set(expected_params)


def test_backward_shared() -> None:
    """
    Test that the backward() function correctly computes gradients in the case of a
    fully shared network.
    """

    splits_args = []
    backward_template(SETTINGS, splits_args)


def test_backward_single() -> None:
    """
    Test that the backward() function correctly computes gradients in the case of a
    single split.
    """

    splits_args = [
        {"region": 1, "copy": 0, "group1": [0, 3], "group2": [1, 2]},
    ]
    backward_template(SETTINGS, splits_args)


def test_backward_multiple() -> None:
    """
    Test that the backward() function correctly computes gradients in the case of
    multiple splits.
    """

    splits_args = [
        {"region": 0, "copy": 0, "group1": [0, 1], "group2": [2, 3]},
        {"region": 1, "copy": 0, "group1": [0, 2], "group2": [1, 3]},
        {"region": 1, "copy": 0, "group1": [0], "group2": [2]},
        {"region": 2, "copy": 0, "group1": [0, 3], "group2": [1, 2]},
    ]
    backward_template(SETTINGS, splits_args)


def test_task_grads_shared() -> None:
    """
    Test that `get_task_grads()` correctly computes task-specific gradients at each
    region of the network in the case of a fully shared network.
    """

    splits_args = []
    gradients_template(SETTINGS, splits_args)


def test_task_grads_single() -> None:
    """
    Test that `get_task_grads()` correctly computes task-specific gradients at each
    region of the network in the case of a single split network.
    """

    splits_args = [
        {"region": 1, "copy": 0, "group1": [0, 3], "group2": [1, 2]},
    ]
    gradients_template(SETTINGS, splits_args)


def test_task_grads_multiple() -> None:
    """
    Test that `get_task_grads()` correctly computes task-specific gradients at each
    region of the network in the case of a multiple split network.
    """

    splits_args = [
        {"region": 0, "copy": 0, "group1": [0, 1], "group2": [2, 3]},
        {"region": 1, "copy": 0, "group1": [0, 2], "group2": [1, 3]},
        {"region": 1, "copy": 0, "group1": [0], "group2": [2]},
        {"region": 2, "copy": 0, "group1": [0, 3], "group2": [1, 2]},
    ]
    gradients_template(SETTINGS, splits_args)


def test_task_grad_diffs_zero() -> None:
    """
    Test that `get_task_grad_diffs()` correctly computes the pairwise difference between
    task-specific gradients at each region when these gradients are hard-coded to zero.
    """

    grad_diffs_template(SETTINGS, "zero")


def test_task_grad_diffs_rand_identical() -> None:
    """
    Test that `get_task_grad_diffs()` correctly computes the pairwise difference between
    task-specific gradients at each region when these gradients are random, but
    identical across tasks.
    """

    grad_diffs_template(SETTINGS, "rand_identical")


def test_task_grad_diffs_rand() -> None:
    """
    Test that `get_task_grad_diffs()` correctly computes the pairwise difference between
    task-specific gradients at each region when these gradients are random.
    """

    grad_diffs_template(SETTINGS, "rand")


def test_split_stats_arithmetic_simple_shared() -> None:
    """
    Test that `get_split_statistics()` correctly computes the z-score over the pairwise
    differences in task gradients in the case of (nearly) identical gradients at each
    time step, a fully shared network, and only using arithmetic means to keep track of
    gradient statistics.
    """

    # Set up case.
    settings = dict(SETTINGS)
    settings["obs_dim"] = 2
    settings["num_tasks"] = 4
    settings["hidden_size"] = settings["obs_dim"] + settings["num_tasks"] + 2

    # Construct series of splits.
    splits_args = []

    # Construct a sequence of task gradients. The network gradient statistics will be
    # updated with these task gradients, and the z-scores will be computed from these
    # statistics.
    total_steps = settings["split_step_threshold"] + 10
    dim = settings["obs_dim"] + settings["num_tasks"]
    max_region_size = settings["hidden_size"] ** 2 + settings["hidden_size"]
    task_grad_vals = [[-2, 1, 0, -1], [2, -1, 1, 1]]
    task_grads = torch.zeros(
        total_steps, settings["num_tasks"], settings["num_layers"], max_region_size
    )
    for task, region in product(
        range(settings["num_tasks"]), range(settings["num_layers"])
    ):
        if region == 0:
            region_size = settings["hidden_size"] * (dim + 1)
        elif region == settings["num_layers"] - 1:
            region_size = dim * (settings["hidden_size"] + 1)
        else:
            region_size = max_region_size

        task_grads[:, task, region, : region_size // 2] = task_grad_vals[0][task]
        task_grads[:, task, region, region_size // 2 : region_size] = task_grad_vals[1][
            task
        ]

    # Run test.
    split_stats_template(settings, task_grads, splits_args)


def test_split_stats_arithmetic_simple_split() -> None:
    """
    Test that `get_split_statistics()` correctly computes the z-score over the pairwise
    differences in task gradients in the case of (nealry) identical gradients at each
    time step, a split network, and only using arithmetic means to keep track of
    gradient statistics.
    """

    # Set up case.
    settings = dict(SETTINGS)
    settings["obs_dim"] = 2
    settings["num_tasks"] = 4
    settings["hidden_size"] = settings["obs_dim"] + settings["num_tasks"] + 2

    # Construct series of splits.
    splits_args = [
        {"region": 0, "copy": 0, "group1": [0, 1], "group2": [2, 3]},
        {"region": 1, "copy": 0, "group1": [0, 2], "group2": [1, 3]},
        {"region": 1, "copy": 1, "group1": [1], "group2": [3]},
        {"region": 2, "copy": 0, "group1": [0, 3], "group2": [1, 2]},
    ]

    # Construct a sequence of task gradients. The network gradient statistics will be
    # updated with these task gradients, and the z-scores will be computed from these
    # statistics.
    total_steps = settings["split_step_threshold"] + 10
    dim = settings["obs_dim"] + settings["num_tasks"]
    max_region_size = settings["hidden_size"] ** 2 + settings["hidden_size"]
    task_grad_vals = [[-2, 1, 0, -1], [2, -1, 1, 1]]
    task_grads = torch.zeros(
        total_steps, settings["num_tasks"], settings["num_layers"], max_region_size
    )
    for task, region in product(
        range(settings["num_tasks"]), range(settings["num_layers"])
    ):
        if region == 0:
            region_size = settings["hidden_size"] * (dim + 1)
        elif region == settings["num_layers"] - 1:
            region_size = dim * (settings["hidden_size"] + 1)
        else:
            region_size = max_region_size

        task_grads[:, task, region, : region_size // 2] = task_grad_vals[0][task]
        task_grads[:, task, region, region_size // 2 : region_size] = task_grad_vals[1][
            task
        ]

    # Run test.
    split_stats_template(settings, task_grads, splits_args)


def test_split_stats_arithmetic_random_shared() -> None:
    """
    Test that `get_split_statistics()` correctly computes the z-score over the pairwise
    differences in task gradients in the case of random gradients at each time step,
    a fully shared network, and only using arithmetic means to keep track of gradient
    statistics.
    """

    # Set up case.
    settings = dict(SETTINGS)
    settings["obs_dim"] = 2
    settings["num_tasks"] = 4
    settings["hidden_size"] = settings["obs_dim"] + settings["num_tasks"] + 2

    # Construct series of splits.
    splits_args = []

    # Construct a sequence of task gradients. The network gradient statistics will be
    # updated with these task gradients, and the z-scores will be computed from these
    # statistics.
    total_steps = settings["split_step_threshold"] + 10
    dim = settings["obs_dim"] + settings["num_tasks"]
    max_region_size = settings["hidden_size"] ** 2 + settings["hidden_size"]
    task_grads = torch.zeros(
        total_steps, settings["num_tasks"], settings["num_layers"], max_region_size
    )
    for region in product(range(settings["num_layers"])):
        if region == 0:
            region_size = settings["hidden_size"] * (dim + 1)
        elif region == settings["num_layers"] - 1:
            region_size = dim * (settings["hidden_size"] + 1)
        else:
            region_size = max_region_size

        task_grads[:, :, region, :region_size] = torch.rand(
            total_steps, settings["num_tasks"], 1, region_size
        )

    # Run test.
    split_stats_template(settings, task_grads, splits_args)


def test_split_stats_arithmetic_random_split() -> None:
    """
    Test that `get_split_statistics()` correctly computes the z-score over the pairwise
    differences in task gradients in the case of identical gradients at each time step,
    a split network, and only using arithmetic means to keep track of gradient
    statistics.
    """

    # Set up case.
    settings = dict(SETTINGS)
    settings["obs_dim"] = 2
    settings["num_tasks"] = 4
    settings["hidden_size"] = settings["obs_dim"] + settings["num_tasks"] + 2

    # Construct series of splits.
    splits_args = [
        {"region": 0, "copy": 0, "group1": [0, 1], "group2": [2, 3]},
        {"region": 1, "copy": 0, "group1": [0, 2], "group2": [1, 3]},
        {"region": 1, "copy": 1, "group1": [1], "group2": [3]},
        {"region": 2, "copy": 0, "group1": [0, 3], "group2": [1, 2]},
    ]

    # Construct a sequence of task gradients. The network gradient statistics will be
    # updated with these task gradients, and the z-scores will be computed from these
    # statistics.
    total_steps = settings["split_step_threshold"] + 10
    dim = settings["obs_dim"] + settings["num_tasks"]
    max_region_size = settings["hidden_size"] ** 2 + settings["hidden_size"]
    task_grads = torch.zeros(
        total_steps, settings["num_tasks"], settings["num_layers"], max_region_size
    )
    for region in product(range(settings["num_layers"])):
        if region == 0:
            region_size = settings["hidden_size"] * (dim + 1)
        elif region == settings["num_layers"] - 1:
            region_size = dim * (settings["hidden_size"] + 1)
        else:
            region_size = max_region_size

        task_grads[:, :, region, :region_size] = torch.rand(
            total_steps, settings["num_tasks"], 1, region_size
        )

    # Run test.
    split_stats_template(settings, task_grads, splits_args)


def test_split_stats_EMA_random_shared() -> None:
    """
    Test that `get_split_statistics()` correctly computes the z-score over the pairwise
    differences in task gradients in the case of random gradients at each time step, a
    fully shared network, and using both arithmetic mean and EMA to keep track of
    gradient statistics.
    """

    # Set up case.
    settings = dict(SETTINGS)
    settings["obs_dim"] = 2
    settings["num_tasks"] = 4
    settings["ema_alpha"] = 0.99
    settings["hidden_size"] = settings["obs_dim"] + settings["num_tasks"] + 2
    ema_threshold = alpha_to_threshold(settings["ema_alpha"])

    # Construct series of splits.
    splits_args = []

    # Construct a sequence of task gradients. The network gradient statistics will be
    # updated with these task gradients, and the z-scores will be computed from these
    # statistics.
    total_steps = ema_threshold + 20
    dim = settings["obs_dim"] + settings["num_tasks"]
    max_region_size = settings["hidden_size"] ** 2 + settings["hidden_size"]
    task_grads = torch.zeros(
        total_steps, settings["num_tasks"], settings["num_layers"], max_region_size
    )
    for region in product(range(settings["num_layers"])):
        if region == 0:
            region_size = settings["hidden_size"] * (dim + 1)
        elif region == settings["num_layers"] - 1:
            region_size = dim * (settings["hidden_size"] + 1)
        else:
            region_size = max_region_size

        task_grads[:, :, region, :region_size] = torch.rand(
            total_steps, settings["num_tasks"], 1, region_size
        )

    # Run test.
    split_stats_template(settings, task_grads, splits_args)


def test_split_stats_EMA_random_split() -> None:
    """
    Test that `get_split_statistics()` correctly computes the z-score over the pairwise
    differences in task gradients in the case of random gradients at each time step, a
    split network, and using both arithmetic mean and EMA to keep track of gradient
    statistics.
    """

    # Set up case.
    settings = dict(SETTINGS)
    settings["obs_dim"] = 2
    settings["num_tasks"] = 4
    settings["ema_alpha"] = 0.99
    settings["hidden_size"] = settings["obs_dim"] + settings["num_tasks"] + 2
    ema_threshold = alpha_to_threshold(settings["ema_alpha"])

    # Construct series of splits.
    splits_args = [
        {"region": 0, "copy": 0, "group1": [0, 1], "group2": [2, 3]},
        {"region": 1, "copy": 0, "group1": [0, 2], "group2": [1, 3]},
        {"region": 1, "copy": 1, "group1": [1], "group2": [3]},
        {"region": 2, "copy": 0, "group1": [0, 3], "group2": [1, 2]},
    ]

    # Construct a sequence of task gradients. The network gradient statistics will be
    # updated with these task gradients, and the z-scores will be computed from these
    # statistics.
    total_steps = ema_threshold + 20
    dim = settings["obs_dim"] + settings["num_tasks"]
    max_region_size = settings["hidden_size"] ** 2 + settings["hidden_size"]
    task_grads = torch.zeros(
        total_steps, settings["num_tasks"], settings["num_layers"], max_region_size
    )
    for region in product(range(settings["num_layers"])):
        if region == 0:
            region_size = settings["hidden_size"] * (dim + 1)
        elif region == settings["num_layers"] - 1:
            region_size = dim * (settings["hidden_size"] + 1)
        else:
            region_size = max_region_size

        task_grads[:, :, region, :region_size] = torch.rand(
            total_steps, settings["num_tasks"], 1, region_size
        )

    # Run test.
    split_stats_template(settings, task_grads, splits_args)


def test_split_stats_EMA_random_split_batch() -> None:
    """
    Test that `get_split_statistics()` correctly computes the z-score over the pairwise
    differences in task gradients in the case of random gradients at each time step, a
    split network, using both arithmetic mean and EMA to keep track of gradient
    statistics, and when the gradient batches each contain gradients for only a subset
    of all tasks.
    """

    # Set up case.
    settings = dict(SETTINGS)
    settings["obs_dim"] = 2
    settings["num_tasks"] = 4
    settings["ema_alpha"] = 0.99
    settings["hidden_size"] = settings["obs_dim"] + settings["num_tasks"] + 2
    ema_threshold = alpha_to_threshold(settings["ema_alpha"])

    # Construct series of splits.
    splits_args = [
        {"region": 0, "copy": 0, "group1": [0, 1], "group2": [2, 3]},
        {"region": 1, "copy": 0, "group1": [0, 2], "group2": [1, 3]},
        {"region": 1, "copy": 1, "group1": [1], "group2": [3]},
        {"region": 2, "copy": 0, "group1": [0, 3], "group2": [1, 2]},
    ]

    # Construct a sequence of task gradients. The network gradient statistics will be
    # updated with these task gradients, and the z-scores will be computed from these
    # statistics.
    total_steps = ema_threshold + 20
    dim = settings["obs_dim"] + settings["num_tasks"]
    max_region_size = settings["hidden_size"] ** 2 + settings["hidden_size"]
    task_grads = torch.zeros(
        total_steps, settings["num_tasks"], settings["num_layers"], max_region_size
    )
    for step in range(total_steps):

        # Generate tasks for each batch. Each task has a 50-50 chance of being included
        # in each batch.
        batch_tasks = torch.rand(settings["num_tasks"]) < 0.5
        batch_tasks = batch_tasks.view(settings["num_tasks"], 1, 1)
        for region in product(range(settings["num_layers"])):
            if region == 0:
                region_size = settings["hidden_size"] * (dim + 1)
            elif region == settings["num_layers"] - 1:
                region_size = dim * (settings["hidden_size"] + 1)
            else:
                region_size = max_region_size

            local_grad = torch.rand(settings["num_tasks"], 1, region_size)
            local_grad *= batch_tasks
            task_grads[step, :, region, :region_size] = local_grad

    # Run test.
    split_stats_template(settings, task_grads, splits_args)


def test_split_stats_EMA_random_capped() -> None:
    """
    Test that `get_split_statistics()` correctly computes the z-score over the pairwise
    differences in task gradients in the case of random gradients at each time step,
    using both arithmetic mean and EMA to keep track of gradient statistics, and not
    capping the gradient statistic sample size.
    """

    # Set up case.
    settings = dict(SETTINGS)
    settings["obs_dim"] = 2
    settings["num_tasks"] = 4
    settings["cap_sample_size"] = False
    settings["ema_alpha"] = 0.99
    settings["hidden_size"] = settings["obs_dim"] + settings["num_tasks"] + 2
    ema_threshold = alpha_to_threshold(settings["ema_alpha"])

    # Construct series of splits.
    splits_args = []

    # Construct a sequence of task gradients. The network gradient statistics will be
    # updated with these task gradients, and the z-scores will be computed from these
    # statistics.
    total_steps = ema_threshold + 20
    dim = settings["obs_dim"] + settings["num_tasks"]
    max_region_size = settings["hidden_size"] ** 2 + settings["hidden_size"]
    task_grads = torch.zeros(
        total_steps, settings["num_tasks"], settings["num_layers"], max_region_size
    )
    for region in product(range(settings["num_layers"])):
        if region == 0:
            region_size = settings["hidden_size"] * (dim + 1)
        elif region == settings["num_layers"] - 1:
            region_size = dim * (settings["hidden_size"] + 1)
        else:
            region_size = max_region_size

        task_grads[:, :, region, :region_size] = torch.rand(
            total_steps, settings["num_tasks"], 1, region_size
        )

    # Run test.
    split_stats_template(settings, task_grads, splits_args)


def test_split_stats_EMA_random_split_grad_var() -> None:
    """
    Test that `get_split_statistics()` correctly computes the z-score over the pairwise
    differences in task gradients in the case of random gradients at each time step, a
    split network, using both arithmetic mean and EMA to keep track of gradient
    statistics, when the standard deviation of task-gradients is given as a
    hyperparameter instead of measured online.
    """

    # Set up case.
    settings = dict(SETTINGS)
    settings["obs_dim"] = 2
    settings["num_tasks"] = 4
    settings["grad_var"] = 0.01
    settings["ema_alpha"] = 0.99
    settings["hidden_size"] = settings["obs_dim"] + settings["num_tasks"] + 2
    ema_threshold = alpha_to_threshold(settings["ema_alpha"])

    # Construct series of splits.
    splits_args = [
        {"region": 0, "copy": 0, "group1": [0, 1], "group2": [2, 3]},
        {"region": 1, "copy": 0, "group1": [0, 2], "group2": [1, 3]},
        {"region": 1, "copy": 1, "group1": [1], "group2": [3]},
        {"region": 2, "copy": 0, "group1": [0, 3], "group2": [1, 2]},
    ]

    # Construct a sequence of task gradients. The network gradient statistics will be
    # updated with these task gradients, and the z-scores will be computed from these
    # statistics.
    total_steps = ema_threshold + 20
    dim = settings["obs_dim"] + settings["num_tasks"]
    max_region_size = settings["hidden_size"] ** 2 + settings["hidden_size"]
    task_grads = torch.zeros(
        total_steps, settings["num_tasks"], settings["num_layers"], max_region_size
    )
    for region in product(range(settings["num_layers"])):
        if region == 0:
            region_size = settings["hidden_size"] * (dim + 1)
        elif region == settings["num_layers"] - 1:
            region_size = dim * (settings["hidden_size"] + 1)
        else:
            region_size = max_region_size

        task_grads[:, :, region, :region_size] = torch.rand(
            total_steps, settings["num_tasks"], 1, region_size
        )

    # Run test.
    split_stats_template(settings, task_grads, splits_args)


def test_split_stats_manual() -> None:
    """
    Test that `get_split_statistics()` correctly computes the z-score over the pairwise
    differences in task gradients for manually computed values.
    """

    # Set up case.
    settings = dict(SETTINGS)
    settings["num_layers"] = 1
    settings["num_tasks"] = 4
    settings["ema_alpha"] = 0.8
    input_size = 1
    output_size = 2
    settings["hidden_size"] = 2
    ema_threshold = alpha_to_threshold(settings["ema_alpha"])

    # Construct a sequence of task gradients. The network gradient statistics will be
    # updated with these task gradients, and the z-scores will be computed from these
    # statistics.
    task_grads = torch.Tensor(
        [
            [
                [[-0.117, 0.08, -0.091, -0.008]],
                [[0, 0, 0, 0]],
                [[-0.053, 0.078, -0.046, 0.017]],
                [[0, 0, 0, 0]],
            ],
            [
                [[-0.006, 0.083, -0.065, -0.095]],
                [[0.037, 0.051, 0.009, -0.075]],
                [[0.107, 0.264, -0.072, 0.143]],
                [[0.049, 0.03, -0.051, -0.012]],
            ],
            [
                [[0.106, -0.092, -0.015, 0.159]],
                [[0, 0, 0, 0]],
                [[0.055, 0.115, -0.096, 0.032]],
                [[-0.21, 0.11, -0.091, -0.014]],
            ],
            [
                [[-0.116, 0.079, 0.087, 0.041]],
                [[0.094, 0.143, -0.015, -0.008]],
                [[-0.056, -0.054, 0.01, 0.073]],
                [[0.103, -0.085, -0.008, -0.018]],
            ],
            [
                [[-0.147, -0.067, -0.063, -0.022]],
                [[-0.098, 0.059, 0.064, 0.045]],
                [[-0.037, 0.138, 0.06, -0.056]],
                [[0, 0, 0, 0]],
            ],
            [
                [[-0.062, 0.001, 0.106, -0.176]],
                [[-0.007, 0.013, -0.095, 0.082]],
                [[-0.003, 0.066, 0.106, -0.17]],
                [[-0.035, -0.027, -0.105, 0.058]],
            ],
            [
                [[0.114, -0.191, -0.054, -0.122]],
                [[0.053, 0.004, -0.019, 0.053]],
                [[0.155, -0.027, 0.054, -0.015]],
                [[0.073, 0.042, -0.08, 0.056]],
            ],
            [
                [[0.094, 0.002, 0.078, -0.049]],
                [[-0.116, 0.205, 0.175, -0.026]],
                [[-0.178, 0.013, -0.012, 0.136]],
                [[-0.05, 0.105, 0.114, -0.053]],
            ],
            [
                [[0, 0, 0, 0]],
                [[-0.171, -0.001, 0.069, -0.077]],
                [[0.11, 0.053, 0.039, -0.005]],
                [[-0.097, 0.046, 0.124, 0.072]],
            ],
        ]
    )
    total_steps = len(task_grads)

    # Set expected values of gradient statistics.
    expected_grad_diff_mean = torch.Tensor(
        [
            [[0, 0, 0.00675, 0], [0, 0, 0, 0], [0.00675, 0, 0, 0], [0, 0, 0, 0]],
            [
                [0, 0.008749, 0.0544865, 0.012919],
                [0.008749, 0, 0.104354, 0.008154],
                [0.0544865, 0.104354, 0, 0.082586],
                [0.012919, 0.008154, 0.082586, 0],
            ],
            [
                [0, 0.008749, 0.05903766667, 0.094642],
                [0.008749, 0, 0.104354, 0.008154],
                [0.05903766667, 0.104354, 0, 0.0774885],
                [0.094642, 0.008154, 0.0774885, 0],
            ],
            [
                [0, 0.034875, 0.05133875, 0.09221566667],
                [0.034875, 0, 0.0864245, 0.030184],
                [0.05133875, 0.0864245, 0, 0.06327466667],
                [0.09221566667, 0.030184, 0.06327466667, 0],
            ],
            [
                [0, 0.036215, 0.055153, 0.09221566667],
                [0.036215, 0, 0.06434266667, 0.030184],
                [0.055153, 0.06434266667, 0, 0.06327466667],
                [0.09221566667, 0.030184, 0.06327466667, 0],
            ],
            [
                [0, 0.05469475, 0.0456708, 0.09435925],
                [0.05469475, 0, 0.0749395, 0.02114266667],
                [0.0456708, 0.0749395, 0, 0.0740005],
                [0.09435925, 0.02114266667, 0.0740005, 0],
            ],
            [
                [0, 0.058475, 0.04687464, 0.0931534],
                [0.058475, 0, 0.0642152, 0.0172505],
                [0.04687464, 0.0642152, 0, 0.0660968],
                [0.0931534, 0.0172505, 0.0660968, 0],
            ],
            [
                [0, 0.0658294, 0.060785712, 0.08105412],
                [0.0658294, 0, 0.07175636, 0.0175616],
                [0.060785712, 0.07175636, 0, 0.06816644],
                [0.08105412, 0.0175616, 0.06816644, 0],
            ],
            [
                [0, 0.0658294, 0.060785712, 0.08105412],
                [0.0658294, 0, 0.074997288, 0.02063148],
                [0.060785712, 0.074997288, 0, 0.065743552],
                [0.08105412, 0.02063148, 0.065743552, 0],
            ],
        ]
    )
    expected_grad_diff_mean = expected_grad_diff_mean.unsqueeze(-1)

    expected_grad_mean = torch.Tensor(
        [
            [-0.034, 0, -0.001, 0],
            [-0.027375, 0.0055, 0.05475, 0.004],
            [-0.005083333333, 0.0055, 0.04533333333, -0.023625],
            [0.001875, 0.0295, 0.0323125, -0.01641666667],
            [-0.01345, 0.0255, 0.0311, -0.01641666667],
            [-0.01731, 0.0186875, 0.02483, -0.019125],
            [-0.026498, 0.0195, 0.028214, -0.01075],
            [-0.0149484, 0.0275, 0.0205212, -0.0028],
            [-0.0149484, 0.013, 0.02626696, 0.00501],
        ]
    )

    expected_grad_var = torch.Tensor(
        [
            [0.0059525, 0, 0.0028235, 0],
            [0.005326734375, 0.00238875, 0.0117619375, 0.0014955],
            [0.007792076389, 0.00238875, 0.009992055556, 0.008282234375],
            [0.007669109375, 0.004036, 0.008708839844, 0.007142576389],
            [0.0074847475, 0.004221083333, 0.00819259, 0.007142576389],
            [0.0081357339, 0.004302214844, 0.0089363611, 0.006214734375],
            [0.009410001996, 0.00364065, 0.008241032204, 0.0059802875],
            [0.008732512137, 0.00679957, 0.009333179951, 0.00633534],
            [0.008732512137, 0.007872256, 0.007936236492, 0.0066536939],
        ]
    )

    expected_z = torch.Tensor(
        [
            [
                [-1.414213562, 0, -1.142280405, 0],
                [0, 0, 0, 0],
                [-1.142280405, 0, -1.414213562, 0],
                [0, 0, 0, 0],
            ],
            [
                [-2, -1.170405699, 0.1473023578, -1.054200557],
                [-1.170405699, -1.414213562, 1.493813156, -1.186986529],
                [0.1473023578, 1.493813156, -2, 0.8872055932],
                [-1.054200557, -1.186986529, 0.8872055932, -1.414213562],
            ],
            [
                [-2.449489743, -1.221703287, -0.1994752765, 0.9450617525],
                [-1.221703287, -1.414213562, 0.8819594016, -1.234795482],
                [-0.1994752765, 0.8819594016, -2.449489743, 0.4112805901],
                [0.9450617525, -1.234795482, 0.4112805901, -2],
            ],
            [
                [-2.828427125, -0.8070526312, -0.3449088766, 1.413801122],
                [-0.8070526312, -2, 0.9562689571, -0.9675147418],
                [-0.3449088766, 0.9562689571, -2.828427125, 0.2013444437],
                [1.413801122, -0.9675147418, 0.2013444437, -2.449489743],
            ],
            [
                [-3.16227766, -0.8721406803, -0.06105579169, 1.566975682],
                [-0.8721406803, -2.449489743, 0.3529635209, -0.926578017],
                [-0.06105579169, 0.3529635209, -3.16227766, 0.3064466412],
                [1.566975682, -0.926578017, 0.3064466412, -2.449489743],
            ],
            [
                [-3.16227766, -0.09688841312, -0.6121886259, 1.884016833],
                [-0.09688841312, -2.828427125, 0.9141650853, -1.535056275],
                [-0.6121886259, 0.9141650853, -3.16227766, 0.8672700021],
                [1.884016833, -1.535056275, 0.8672700021, -2.828427125],
            ],
            [
                [-3.16227766, 0.227909676, -0.4446408767, 2.238448753],
                [0.227909676, -3.16227766, 0.56070751, -1.933886337],
                [-0.4446408767, 0.56070751, -3.16227766, 0.6697964624],
                [2.238448753, -1.933886337, 0.6697964624, -3.16227766],
            ],
            [
                [-3.16227766, 0.1737291325, -0.08186756787, 0.9452653965],
                [0.1737291325, -3.16227766, 0.4740870094, -2.272316383],
                [-0.08186756787, 0.4740870094, -3.16227766, 0.2921622538],
                [0.9452653965, -2.272316383, 0.2921622538, -3.16227766],
            ],
            [
                [-3.16227766, 0.1743604677, -0.08128460407, 0.946042744],
                [0.1743604677, -3.16227766, 0.6390453145, -2.116547594],
                [-0.08128460407, 0.6390453145, -3.16227766, 0.170009164],
                [0.946042744, -2.116547594, 0.170009164, -3.16227766],
            ],
        ]
    )
    expected_z = expected_z.unsqueeze(-1)

    expected_sample_size = torch.Tensor(
        [
            [1, 0, 1, 0],
            [2, 1, 2, 1],
            [3, 1, 3, 2],
            [4, 2, 4, 3],
            [5, 3, 5, 3],
            [5, 4, 5, 4],
            [5, 5, 5, 5],
            [5, 5, 5, 5],
            [5, 5, 5, 5],
        ]
    )
    expected_pair_sample_size = torch.Tensor(
        [
            [[1, 0, 1, 0], [0, 0, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0]],
            [[2, 1, 2, 1], [1, 1, 1, 1], [2, 1, 2, 1], [1, 1, 1, 1]],
            [[3, 1, 3, 2], [1, 1, 1, 1], [3, 1, 3, 2], [2, 1, 2, 2]],
            [[4, 2, 4, 3], [2, 2, 2, 2], [4, 2, 4, 3], [3, 2, 3, 3]],
            [[5, 3, 5, 3], [3, 3, 3, 2], [5, 3, 5, 3], [3, 2, 3, 3]],
            [[5, 4, 5, 4], [4, 4, 4, 3], [5, 4, 5, 4], [4, 3, 4, 4]],
            [[5, 5, 5, 5], [5, 5, 5, 4], [5, 5, 5, 5], [5, 4, 5, 5]],
            [[5, 5, 5, 5], [5, 5, 5, 5], [5, 5, 5, 5], [5, 5, 5, 5]],
            [[5, 5, 5, 5], [5, 5, 5, 5], [5, 5, 5, 5], [5, 5, 5, 5]],
        ]
    )
    expected_pair_sample_size = expected_pair_sample_size.unsqueeze(-1)

    # Instantiate network.
    network = MultiTaskSplittingNetworkV1(
        input_size=input_size,
        output_size=output_size,
        init_base=init_base,
        init_final=init_base,
        num_tasks=settings["num_tasks"],
        num_layers=settings["num_layers"],
        hidden_size=settings["hidden_size"],
        ema_alpha=settings["ema_alpha"],
    )

    # Update gradient statistics for each step.
    for step in range(total_steps):
        network.num_steps += 1
        network.update_grad_stats(task_grads[step])
        z = network.get_split_statistics()

        # Compare network statistics to expected values.
        assert torch.all(network.grad_stats.sample_size == expected_sample_size[step])
        assert torch.all(
            network.grad_diff_stats.sample_size == expected_pair_sample_size[step]
        )
        assert torch.allclose(
            network.grad_diff_stats.mean, expected_grad_diff_mean[step]
        )
        assert torch.allclose(network.grad_stats.mean, expected_grad_mean[step])
        assert torch.allclose(network.grad_stats.var, expected_grad_var[step])
        assert torch.allclose(z, expected_z[step], atol=TOL)


def test_split_rand_all_tasks() -> None:
    """
    Test whether splitting decisions are made correctly when given z-scores for the
    pairwise difference in gradient distributions at each region, when z-scores are
    generated randomly and all tasks are included in each batch.
    """

    # Set up case.
    settings = dict(SETTINGS)
    total_steps = settings["split_step_threshold"] + 20
    splits_args = []

    # Generate z-scores, ensuring that `z[:, task1, task2, :] == z[:, task2, task1, :]`.
    mean = torch.zeros(
        total_steps,
        settings["num_tasks"],
        settings["num_tasks"],
        settings["num_layers"],
    )
    std = torch.ones(
        total_steps,
        settings["num_tasks"],
        settings["num_tasks"],
        settings["num_layers"],
    )
    z = torch.normal(mean, std)
    for task1 in range(settings["num_tasks"] - 1):
        for task2 in range(task1 + 1, settings["num_tasks"]):
            z[:, task1, task2, :] = z[:, task2, task1, :]

    # Generate task grads. Since all tasks are included in each batch, none of these
    # will be zero.
    hidden_size = settings["obs_dim"] + settings["num_tasks"]
    max_region_size = hidden_size ** 2 + hidden_size
    task_grad_shape = (settings["num_tasks"], settings["num_layers"], max_region_size)
    task_grads = torch.ones(total_steps, *task_grad_shape)

    # Call template.
    split_template(settings, z, task_grads, splits_args)


def test_split_rand_some_tasks() -> None:
    """
    Test whether splitting decisions are made correctly when given z-scores for the
    pairwise difference in gradient distributions at each region, when z-scores are
    generated randomly and only some tasks are included in each batch.
    """

    # Set up case.
    settings = dict(SETTINGS)
    total_steps = 4 * settings["split_step_threshold"]
    splits_args = []

    # Generate z-scores, ensuring that `z[:, task1, task2, :] == z[:, task2, task1, :]`.
    mean = torch.zeros(
        total_steps,
        settings["num_tasks"],
        settings["num_tasks"],
        settings["num_layers"],
    )
    std = torch.ones(
        total_steps,
        settings["num_tasks"],
        settings["num_tasks"],
        settings["num_layers"],
    )
    z = torch.normal(mean, std)
    for task1 in range(settings["num_tasks"] - 1):
        for task2 in range(task1 + 1, settings["num_tasks"]):
            z[:, task1, task2, :] = z[:, task2, task1, :]

    # Generate task grads. Since only some tasks are included in each batch, some grads
    # in each batch will be set to zero.
    hidden_size = settings["obs_dim"] + settings["num_tasks"]
    max_region_size = hidden_size ** 2 + hidden_size
    task_grad_shape = (settings["num_tasks"], settings["num_layers"], max_region_size)
    task_grads = torch.ones(total_steps, *task_grad_shape)
    task_probs = [1.0, 0.1, 1.0, 0.5]
    for step in range(total_steps):
        for task in range(settings["num_tasks"]):
            if random.random() >= task_probs[task]:
                task_grads[step, task] = 0

    # Call template.
    split_template(settings, z, task_grads, splits_args)


def test_split_always() -> None:
    """
    Test whether splitting decisions are made correctly when given z-scores for the
    pairwise difference in gradient distributions at each region, when z-scores are
    always above the critical value.
    """

    # Set up case.
    settings = dict(SETTINGS)
    total_steps = settings["split_step_threshold"] + 20
    splits_args = []

    # Generate z-scores, ensuring that `z[:, task1, task2, :] == z[:, task2, task1, :]`.
    critical_z = stats.norm.ppf(1 - settings["split_alpha"])
    z = torch.ones(
        total_steps,
        settings["num_tasks"],
        settings["num_tasks"],
        settings["num_layers"],
    )
    z *= critical_z + 1

    # Generate task grads.
    hidden_size = settings["obs_dim"] + settings["num_tasks"]
    max_region_size = hidden_size ** 2 + hidden_size
    task_grad_shape = (settings["num_tasks"], settings["num_layers"], max_region_size)
    task_grads = torch.ones(total_steps, *task_grad_shape)

    # Call template.
    split_template(settings, z, task_grads, splits_args)


def test_split_never() -> None:
    """
    Test whether splitting decisions are made correctly when given z-scores for the
    pairwise difference in gradient distributions at each region, when z-scores are
    never above the critical value.
    """

    # Set up case.
    settings = dict(SETTINGS)
    total_steps = settings["split_step_threshold"] + 20
    splits_args = []

    # Generate z-scores, ensuring that `z[:, task1, task2, :] == z[:, task2, task1, :]`.
    z = torch.zeros(
        total_steps,
        settings["num_tasks"],
        settings["num_tasks"],
        settings["num_layers"],
    )

    # Generate task grads.
    hidden_size = settings["obs_dim"] + settings["num_tasks"]
    max_region_size = hidden_size ** 2 + hidden_size
    task_grad_shape = (settings["num_tasks"], settings["num_layers"], max_region_size)
    task_grads = torch.ones(total_steps, *task_grad_shape)

    # Call template.
    split_template(settings, z, task_grads, splits_args)


def test_sharing_score_shared() -> None:
    """
    Test that the sharing score is correctly computed for a fully shared network.
    """

    # Set up case.
    settings = dict(SETTINGS)
    dim = settings["obs_dim"] + settings["num_tasks"]
    settings["input_size"] = dim
    settings["output_size"] = dim
    settings["hidden_size"] = dim
    splits_args = []
    expected_score = 1.0

    # Call template.
    score_template(settings, splits_args, expected_score)


def test_sharing_score_separate() -> None:
    """
    Test that the sharing score is correctly computed for a fully separated network,
    i.e. a network with no sharing.
    """

    # Set up case.
    settings = dict(SETTINGS)
    dim = settings["obs_dim"] + settings["num_tasks"]
    settings["input_size"] = dim
    settings["output_size"] = dim
    settings["hidden_size"] = dim
    splits_args = [
        {"region": 0, "copy": 0, "group1": [0, 1], "group2": [2, 3]},
        {"region": 0, "copy": 0, "group1": [0], "group2": [1]},
        {"region": 0, "copy": 1, "group1": [2], "group2": [3]},
        {"region": 1, "copy": 0, "group1": [0, 1], "group2": [2, 3]},
        {"region": 1, "copy": 0, "group1": [0], "group2": [1]},
        {"region": 1, "copy": 1, "group1": [2], "group2": [3]},
        {"region": 2, "copy": 0, "group1": [0, 1], "group2": [2, 3]},
        {"region": 2, "copy": 0, "group1": [0], "group2": [1]},
        {"region": 2, "copy": 1, "group1": [2], "group2": [3]},
    ]
    expected_score = 0.0

    # Call template.
    score_template(settings, splits_args, expected_score)


def test_sharing_score_split_1() -> None:
    """
    Test that the sharing score is correctly computed for a network with half of each
    region shared.
    """

    # Set up case.
    settings = dict(SETTINGS)
    dim = settings["obs_dim"] + settings["num_tasks"]
    settings["input_size"] = dim
    settings["output_size"] = dim
    settings["hidden_size"] = dim
    splits_args = [
        {"region": 0, "copy": 0, "group1": [0, 1], "group2": [2, 3]},
        {"region": 1, "copy": 0, "group1": [0, 2], "group2": [1, 3]},
        {"region": 2, "copy": 0, "group1": [0, 3], "group2": [1, 2]},
    ]
    expected_score = 2.0 / 3.0

    # Call template.
    score_template(settings, splits_args, expected_score)


def test_sharing_score_split_2() -> None:
    """
    Test that the sharing score is correctly computed for a network with half of each
    region shared.
    """

    # Set up case.
    settings = dict(SETTINGS)
    dim = settings["obs_dim"] + settings["num_tasks"]
    settings["input_size"] = settings["obs_dim"]
    settings["output_size"] = dim
    settings["hidden_size"] = dim
    splits_args = [
        {"region": 1, "copy": 0, "group1": [0, 2], "group2": [1, 3]},
        {"region": 2, "copy": 0, "group1": [0], "group2": [1, 2, 3]},
        {"region": 2, "copy": 1, "group1": [1], "group2": [2, 3]},
    ]
    dim = settings["obs_dim"] + settings["num_tasks"]
    region_sizes = [
        settings["obs_dim"] * dim + dim,
        dim ** 2 + dim,
        dim ** 2 + dim,
    ]
    region_scores = [1.0, 2.0 / 3.0, 1.0 / 3.0]
    expected_score = sum(
        [score * size for score, size in zip(region_scores, region_sizes)]
    ) / sum(region_sizes)

    # Call template.
    score_template(settings, splits_args, expected_score)


def split_stats_distribution() -> None:
    """
    IMPORTANT: This test is currently disabled because it can't pass with the current
    implementation. Our statistical test isn't truly accurate (we use some heuristics)
    so we can't exactly know the distribution of the z-scores. To re-enable the test,
    just add "test_" to the beginning of the function name.

    This is a sanity check on our computation of z-scores in `split_statistics()`. By
    randomly generating task gradients according to the distribution of the
    null-hypothesis, the resulting z-scores should follow a standard normal
    distribution. We check this condition for a varying number of tasks, number of
    layers, splitting configurations, etc. It should also be noted that `num_tasks`
    should probably stay at 2. If it's higher, then the pool of z-scores will not have
    been independently sampled, so the apparent distribution may not look like it's
    supposed to.
    """

    ALPHA = 0.05
    TOTAL_STEPS = 250
    START_STEP = 200
    NUM_TASKS = 2
    NUM_TRIALS = 100
    load = None

    # Construct list of options for each setting.
    settings_vals = {
        "obs_dim": [2, 4, 10],
        "hidden_size": [2, 5, 10],
        "num_layers": [3, 20, 100],
        "splits_args": [[]],
        "grad_sigma": [0.001, 0.1, 1.0],
    }
    used_settings = []
    reject_probs = []

    def get_settings() -> Dict[str, Any]:
        """ Helper function to construct settings dictionary. """
        return {name: random.choice(options) for name, options in settings_vals.items()}

    for i in range(NUM_TRIALS):

        # Construct unique settings.
        settings = get_settings()
        while settings in used_settings:
            settings = get_settings()
        used_settings.append(dict(settings))

        # Construct network.
        dim = settings["obs_dim"] + NUM_TASKS
        network = MultiTaskSplittingNetworkV1(
            input_size=dim,
            output_size=dim,
            init_base=init_base,
            init_final=init_base,
            num_tasks=NUM_TASKS,
            num_layers=settings["num_layers"],
            hidden_size=settings["hidden_size"],
        )

        # Construct a sequence of task gradients according to the distribution of the null
        # hypothesis. According to this distribution, each element has mean 0 and standard
        # deviation `settings["grad_sigma"]`.
        region_sizes = network.region_sizes.tolist()
        task_grads = torch.zeros(
            TOTAL_STEPS,
            network.num_tasks,
            network.num_regions,
            network.max_region_size,
        )

        if load is None:
            for region in range(network.num_regions):
                mean = torch.zeros(TOTAL_STEPS, network.num_tasks, region_sizes[region])
                std = torch.ones(TOTAL_STEPS, network.num_tasks, region_sizes[region])
                std *= settings["grad_sigma"]
                task_grads[:, :, region, : region_sizes[region]] = torch.normal(
                    mean, std
                )
        else:
            # Load in generated values saved by scripts/stats_test.py, for debugging.
            with open(load, "rb") as f:
                grads_arr = np.load(f)
            grads_arr = np.transpose(grads_arr, (0, 3, 1, 2))
            task_grads = torch.Tensor(grads_arr)

        # Update the network's gradient statistics with our constructed task gradients, and
        # compute the split statistics (z-scores) along the way.
        reject_count = 0
        for step in range(TOTAL_STEPS):
            network.num_steps += 1
            network.update_grad_stats(task_grads[step])

            if step >= START_STEP:
                z = network.get_split_statistics().numpy()

                # Before we check the distribution of z-scores, we have to remove all the
                # scores outside of the upper triangle. This will remove duplicates
                # (`(task1, task2, region)` vs `(task2, task1, region)`) as well as trivial
                # scores of a task against itself (`(task, task, region)`).
                filtered_z = []
                for task in range(network.num_tasks):
                    filtered_z.append(z[task, task + 1 :, :].reshape(-1))
                filtered_z = np.concatenate(filtered_z)
                assert (
                    len(filtered_z)
                    == network.num_tasks
                    * (network.num_tasks - 1)
                    * network.num_regions
                    / 2
                )
                z = filtered_z

                # Check that the set of computed z-scores follows a standard normal
                # distribution using the Kolgomorov-Smirnov test.
                s, p = stats.kstest(z, "norm")
                if p < ALPHA:
                    reject_count += 1

        reject_prob = reject_count / (TOTAL_STEPS - START_STEP)
        reject_probs.append(reject_prob)

    avg_reject_prob = sum(reject_probs) / len(reject_probs)
    print("reject_probs: %s" % str(reject_probs))
    print("avg reject_prob: %f" % avg_reject_prob)
    print(
        "num rejects: %d/%d"
        % (len([prob for prob in reject_probs if prob != 0.0]), len(reject_probs))
    )
    assert abs(avg_reject_prob - ALPHA) < ALPHA * 0.1
