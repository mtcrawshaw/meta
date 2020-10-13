"""
Unit tests for meta/networks/splitting.py.
"""

import math
import random
from itertools import product
from typing import Dict, Any

import numpy as np
from scipy import stats
import torch
import torch.nn.functional as F
from gym.spaces import Box

from meta.networks.initialize import init_base
from meta.networks.splitting import SplittingMLPNetwork
from meta.utils.estimate import alpha_to_threshold
from tests.helpers import DEFAULT_SETTINGS, get_obs_batch
from tests.networks.templates import (
    gradients_template,
    backward_template,
    grad_diffs_template,
    split_stats_template,
)


SETTINGS = {
    "obs_dim": 8,
    "num_processes": 8,
    "num_tasks": 4,
    "num_layers": 3,
    "split_alpha": 0.05,
    "split_step_threshold": 30,
    "ema_alpha": 0.999,
    "include_task_index": True,
    "device": torch.device("cpu"),
}
EMA_ALPHA = 0.999
EMA_THRESHOLD = alpha_to_threshold(EMA_ALPHA)


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
    network = SplittingMLPNetwork(
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
    network = SplittingMLPNetwork(
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
    network = SplittingMLPNetwork(
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
    network = SplittingMLPNetwork(
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
    network = SplittingMLPNetwork(
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
    differences in task gradients in the case of identical gradients at each time step,
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
    task_grad_vals = [0, -1, 1, 0]
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

        task_grads[:, task, region, :region_size] = task_grad_vals[task]

    # Run test.
    split_stats_template(settings, task_grads, splits_args)


def test_split_stats_arithmetic_simple_split() -> None:
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
    task_grad_vals = [-2, 1, 0, -1]
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

        task_grads[:, task, region, :region_size] = task_grad_vals[task]

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
    settings["hidden_size"] = settings["obs_dim"] + settings["num_tasks"] + 2

    # Construct series of splits.
    splits_args = []

    # Construct a sequence of task gradients. The network gradient statistics will be
    # updated with these task gradients, and the z-scores will be computed from these
    # statistics.
    total_steps = EMA_THRESHOLD + 20
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
    total_steps = EMA_THRESHOLD + 20
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


def split_stats_distribution() -> None:
    """
    IMPORTANT: This test is currently disabled because it can't pass with the current
    implementation. Our statistical test isn't truly accurate (we use some heuristics)
    so we can't exactly know the distribution of the z-scores.

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
        network = SplittingMLPNetwork(
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
                task_grads[:, :, region, : region_sizes[region]] = torch.normal(mean, std)
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
    print("num rejects: %d/%d" % (len([prob for prob in reject_probs if prob != 0.0]), len(reject_probs)))
    # assert abs(avg_reject_prob - ALPHA) < ALPHA * 0.1
    assert False
