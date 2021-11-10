"""
Unit tests for meta/networks/splitting/multi_splitting_base.py.
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

from meta.networks.utils import init_base
from meta.networks.splitting import BaseMultiTaskSplittingNetwork
from meta.utils.estimate import alpha_to_threshold
from tests.helpers import DEFAULT_SETTINGS, get_obs_batch
from tests.networks.splitting import BASE_SETTINGS
from tests.networks.splitting.templates import (
    TOL,
    gradients_template,
    backward_template,
    grad_diffs_template,
    grad_stats_template,
    score_template,
    load_template,
)


def test_forward_shared() -> None:
    """
    Test forward() when all regions of the splitting network are fully shared. The
    function computed by the network should be f(x) = 3 * tanh(2 * tanh(x + 1) + 2) + 3.
    """

    # Set up case.
    dim = BASE_SETTINGS["obs_dim"] + BASE_SETTINGS["num_tasks"]
    observation_subspace = Box(
        low=-np.inf, high=np.inf, shape=(BASE_SETTINGS["obs_dim"],)
    )
    observation_subspace.seed(DEFAULT_SETTINGS["seed"])
    hidden_size = dim

    # Construct network.
    network = BaseMultiTaskSplittingNetwork(
        input_size=dim,
        output_size=dim,
        num_tasks=BASE_SETTINGS["num_tasks"],
        num_layers=BASE_SETTINGS["num_layers"],
        hidden_size=hidden_size,
        device=BASE_SETTINGS["device"],
    )

    # Set network weights.
    state_dict = network.state_dict()
    for i in range(BASE_SETTINGS["num_layers"]):
        weight_name = "regions.%d.0.0.weight" % i
        bias_name = "regions.%d.0.0.bias" % i
        state_dict[weight_name] = torch.Tensor((i + 1) * np.identity(dim))
        state_dict[bias_name] = torch.Tensor((i + 1) * np.ones(dim))
    network.load_state_dict(state_dict)

    # Construct batch of observations concatenated with one-hot task vectors.
    obs, task_indices = get_obs_batch(
        batch_size=BASE_SETTINGS["num_processes"],
        obs_space=observation_subspace,
        num_tasks=BASE_SETTINGS["num_tasks"],
    )

    # Get output of network.
    output = network(obs, task_indices)

    # Computed expected output of network.
    expected_output = 3 * torch.tanh(2 * torch.tanh(obs + 1) + 2) + 3

    # Test output of network.
    assert torch.allclose(output, expected_output)


def test_forward_shared_all_tasks() -> None:
    """
    Test forward() when all regions of the splitting network are fully shared, and a
    prediction for each task is made for all inputs. The function computed by the
    network should be f(x) = 3 * tanh(2 * tanh(x + 1) + 2) + 3.
    """

    # Set up case.
    dim = BASE_SETTINGS["obs_dim"]
    hidden_size = dim

    # Construct network.
    network = BaseMultiTaskSplittingNetwork(
        input_size=dim,
        output_size=dim,
        num_tasks=BASE_SETTINGS["num_tasks"],
        num_layers=BASE_SETTINGS["num_layers"],
        hidden_size=hidden_size,
        device=BASE_SETTINGS["device"],
    )

    # Set network weights.
    state_dict = network.state_dict()
    for i in range(BASE_SETTINGS["num_layers"]):
        weight_name = "regions.%d.0.0.weight" % i
        bias_name = "regions.%d.0.0.bias" % i
        state_dict[weight_name] = torch.Tensor((i + 1) * np.identity(dim))
        state_dict[bias_name] = torch.Tensor((i + 1) * np.ones(dim))
    network.load_state_dict(state_dict)

    # Construct batch of inputs.
    batch_size = BASE_SETTINGS["batch_size"]
    inputs = torch.rand((batch_size, dim))

    # Get output of network.
    output = network(inputs)

    # Computed expected output of network.
    expected_output = 3 * torch.tanh(2 * torch.tanh(inputs + 1) + 2) + 3
    expected_output = torch.stack([expected_output] * BASE_SETTINGS["num_tasks"], dim=0)

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
    dim = BASE_SETTINGS["obs_dim"] + BASE_SETTINGS["num_tasks"]
    observation_subspace = Box(
        low=-np.inf, high=np.inf, shape=(BASE_SETTINGS["obs_dim"],)
    )
    observation_subspace.seed(DEFAULT_SETTINGS["seed"])
    hidden_size = dim

    # Construct network.
    network = BaseMultiTaskSplittingNetwork(
        input_size=dim,
        output_size=dim,
        num_tasks=BASE_SETTINGS["num_tasks"],
        num_layers=BASE_SETTINGS["num_layers"],
        hidden_size=hidden_size,
        device=BASE_SETTINGS["device"],
    )

    # Split the network at the second layer. Tasks 0 and 1 stay assigned to the original
    # copy and tasks 2 and 3 are assigned to the new copy.
    network.split(1, 0, [0, 1], [2, 3])

    # Set network weights.
    state_dict = network.state_dict()
    for i in range(BASE_SETTINGS["num_layers"]):
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
        batch_size=BASE_SETTINGS["num_processes"],
        obs_space=observation_subspace,
        num_tasks=BASE_SETTINGS["num_tasks"],
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


def test_forward_single_all_tasks() -> None:
    """
    Test forward() when all regions of the splitting network are fully shared except
    one, and when all inputs get a prediction for all tasks. The function computed by
    the network should be f(x) = 3 * tanh(2 * tanh(x + 1) + 2) + 3 for tasks 0 and 1 and
    f(x) = 3 * tanh(-2 * tanh(x + 1) - 2) + 3 for tasks 2 and 3.
    """

    # Set up case.
    dim = BASE_SETTINGS["obs_dim"]
    hidden_size = dim

    # Construct network.
    network = BaseMultiTaskSplittingNetwork(
        input_size=dim,
        output_size=dim,
        num_tasks=BASE_SETTINGS["num_tasks"],
        num_layers=BASE_SETTINGS["num_layers"],
        hidden_size=hidden_size,
        device=BASE_SETTINGS["device"],
    )

    # Split the network at the second layer. Tasks 0 and 1 stay assigned to the original
    # copy and tasks 2 and 3 are assigned to the new copy.
    network.split(1, 0, [0, 1], [2, 3])

    # Set network weights.
    state_dict = network.state_dict()
    for i in range(BASE_SETTINGS["num_layers"]):
        weight_name = "regions.%d.0.0.weight" % i
        bias_name = "regions.%d.0.0.bias" % i
        state_dict[weight_name] = torch.Tensor((i + 1) * np.identity(dim))
        state_dict[bias_name] = torch.Tensor((i + 1) * np.ones(dim))
    weight_name = "regions.1.1.0.weight"
    bias_name = "regions.1.1.0.bias"
    state_dict[weight_name] = torch.Tensor(-2 * np.identity(dim))
    state_dict[bias_name] = torch.Tensor(-2 * np.ones(dim))
    network.load_state_dict(state_dict)

    # Construct batch of inputs.
    batch_size = BASE_SETTINGS["batch_size"]
    inputs = torch.rand((batch_size, dim))

    # Get output of network.
    output = network(inputs)

    # Computed expected output of network.
    expected_output = torch.zeros(BASE_SETTINGS["num_tasks"], *inputs.shape)
    for task in range(BASE_SETTINGS["num_tasks"]):
        if task in [0, 1]:
            expected_output[task] = 3 * torch.tanh(2 * torch.tanh(inputs + 1) + 2) + 3
        elif task in [2, 3]:
            expected_output[task] = 3 * torch.tanh(-2 * torch.tanh(inputs + 1) - 2) + 3
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
    dim = BASE_SETTINGS["obs_dim"] + BASE_SETTINGS["num_tasks"]
    observation_subspace = Box(
        low=-np.inf, high=np.inf, shape=(BASE_SETTINGS["obs_dim"],)
    )
    observation_subspace.seed(DEFAULT_SETTINGS["seed"])
    hidden_size = dim

    # Construct network.
    network = BaseMultiTaskSplittingNetwork(
        input_size=dim,
        output_size=dim,
        num_tasks=BASE_SETTINGS["num_tasks"],
        num_layers=BASE_SETTINGS["num_layers"],
        hidden_size=hidden_size,
        device=BASE_SETTINGS["device"],
    )

    # Split the network at the second layer. Tasks 0 and 1 stay assigned to the original
    # copy and tasks 2 and 3 are assigned to the new copy.
    network.split(0, 0, [0, 1], [2, 3])
    network.split(1, 0, [0, 2], [1, 3])
    network.split(1, 0, [0], [2])
    network.split(2, 0, [0, 3], [1, 2])

    # Set network weights.
    state_dict = network.state_dict()
    for i in range(BASE_SETTINGS["num_layers"]):
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
        batch_size=BASE_SETTINGS["num_processes"],
        obs_space=observation_subspace,
        num_tasks=BASE_SETTINGS["num_tasks"],
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


def test_forward_multiple_all_tasks() -> None:
    """
    Test forward() when none of the layers are fully shared, and when all inputs receive
    a prediction for all tasks. The function computed by the network should be:
    - f(x) = 3 * tanh(2 * tanh(x + 1) + 2) + 3 for task 0
    - f(x) = -3 * tanh(-2 * tanh(x + 1) - 2) - 3 for task 1
    - f(x) = -3 * tanh(1/2 * tanh(-x - 1) + 1/2) - 3 for task 2
    - f(x) = 3 * tanh(-2 * tanh(-x - 1) - 2) + 3 for task 3
    """

    # Set up case.
    dim = BASE_SETTINGS["obs_dim"]
    hidden_size = dim

    # Construct network.
    network = BaseMultiTaskSplittingNetwork(
        input_size=dim,
        output_size=dim,
        num_tasks=BASE_SETTINGS["num_tasks"],
        num_layers=BASE_SETTINGS["num_layers"],
        hidden_size=hidden_size,
        device=BASE_SETTINGS["device"],
    )

    # Split the network at the second layer. Tasks 0 and 1 stay assigned to the original
    # copy and tasks 2 and 3 are assigned to the new copy.
    network.split(0, 0, [0, 1], [2, 3])
    network.split(1, 0, [0, 2], [1, 3])
    network.split(1, 0, [0], [2])
    network.split(2, 0, [0, 3], [1, 2])

    # Set network weights.
    state_dict = network.state_dict()
    for i in range(BASE_SETTINGS["num_layers"]):
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

    # Construct batch of inputs.
    batch_size = BASE_SETTINGS["batch_size"]
    inputs = torch.rand((batch_size, dim))

    # Get output of network.
    output = network(inputs)

    # Computed expected output of network.
    expected_output = torch.zeros(BASE_SETTINGS["num_tasks"], *inputs.shape)
    for task in range(BASE_SETTINGS["num_tasks"]):
        if task == 0:
            expected_output[task] = 3 * torch.tanh(2 * torch.tanh(inputs + 1) + 2) + 3
        elif task == 1:
            expected_output[task] = -3 * torch.tanh(-2 * torch.tanh(inputs + 1) - 2) - 3
        elif task == 2:
            expected_output[task] = (
                -3 * torch.tanh(1 / 2 * torch.tanh(-inputs - 1) + 1 / 2) - 3
            )
        elif task == 3:
            expected_output[task] = 3 * torch.tanh(-2 * torch.tanh(-inputs - 1) - 2) + 3
        else:
            raise NotImplementedError

    # Test output of network.
    print(output.shape)
    print(expected_output.shape)
    assert torch.allclose(output, expected_output)


def test_split_single() -> None:
    """
    Test that split() correctly sets new parameters when we perform a single split.
    """

    # Set up case.
    dim = BASE_SETTINGS["obs_dim"] + BASE_SETTINGS["num_tasks"]
    observation_subspace = Box(
        low=-np.inf, high=np.inf, shape=(BASE_SETTINGS["obs_dim"],)
    )
    observation_subspace.seed(DEFAULT_SETTINGS["seed"])
    hidden_size = dim

    # Construct network.
    network = BaseMultiTaskSplittingNetwork(
        input_size=dim,
        output_size=dim,
        num_tasks=BASE_SETTINGS["num_tasks"],
        num_layers=BASE_SETTINGS["num_layers"],
        hidden_size=hidden_size,
        device=BASE_SETTINGS["device"],
    )

    # Split the network at the last layer, so that tasks 0 and 2 stay assigned to the
    # original copy and tasks 1 and 3 are assigned to the new copy.
    network.split(2, 0, [0, 2], [1, 3])

    # Check the parameters of the network.
    param_names = [name for name, param in network.named_parameters()]

    # Construct expected parameters of network.
    region_copies = {i: [0] for i in range(BASE_SETTINGS["num_layers"])}
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
    dim = BASE_SETTINGS["obs_dim"] + BASE_SETTINGS["num_tasks"]
    observation_subspace = Box(
        low=-np.inf, high=np.inf, shape=(BASE_SETTINGS["obs_dim"],)
    )
    observation_subspace.seed(DEFAULT_SETTINGS["seed"])
    hidden_size = dim

    # Construct network.
    network = BaseMultiTaskSplittingNetwork(
        input_size=dim,
        output_size=dim,
        num_tasks=BASE_SETTINGS["num_tasks"],
        num_layers=BASE_SETTINGS["num_layers"],
        hidden_size=hidden_size,
        device=BASE_SETTINGS["device"],
    )

    # Split the network at the first layer once and the last layer twice.
    network.split(0, 0, [0, 1], [2, 3])
    network.split(2, 0, [0, 2], [1, 3])
    network.split(2, 1, [1], [3])

    # Check the parameters of the network.
    param_names = [name for name, param in network.named_parameters()]

    # Construct expected parameters of network.
    region_copies = {i: [0] for i in range(BASE_SETTINGS["num_layers"])}
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
    backward_template(BASE_SETTINGS, splits_args)


def test_backward_shared_all_tasks() -> None:
    """
    Test that the backward() function correctly computes gradients in the case of a
    fully shared network, when inference on each input is run for all tasks.
    """
    splits_args = []
    backward_template(BASE_SETTINGS, splits_args, all_tasks=True)


def test_backward_single() -> None:
    """
    Test that the backward() function correctly computes gradients in the case of a
    single split.
    """
    splits_args = [
        {"region": 1, "copy": 0, "group1": [0, 3], "group2": [1, 2]},
    ]
    backward_template(BASE_SETTINGS, splits_args)


def test_backward_single_all_tasks() -> None:
    """
    Test that the backward() function correctly computes gradients in the case of a
    single split, when inference on each input is run for all tasks.
    """
    splits_args = [
        {"region": 1, "copy": 0, "group1": [0, 3], "group2": [1, 2]},
    ]
    backward_template(BASE_SETTINGS, splits_args, all_tasks=True)


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
    backward_template(BASE_SETTINGS, splits_args)


def test_backward_multiple_all_tasks() -> None:
    """
    Test that the backward() function correctly computes gradients in the case of
    multiple splits, when inference on each input is run for all tasks.
    """
    splits_args = [
        {"region": 0, "copy": 0, "group1": [0, 1], "group2": [2, 3]},
        {"region": 1, "copy": 0, "group1": [0, 2], "group2": [1, 3]},
        {"region": 1, "copy": 0, "group1": [0], "group2": [2]},
        {"region": 2, "copy": 0, "group1": [0, 3], "group2": [1, 2]},
    ]
    backward_template(BASE_SETTINGS, splits_args, all_tasks=True)


def test_task_grads_shared() -> None:
    """
    Test that `get_task_grads()` correctly computes task-specific gradients at each
    region of the network in the case of a fully shared network.
    """
    splits_args = []
    gradients_template(BASE_SETTINGS, splits_args)


def test_task_grads_shared_all_tasks() -> None:
    """
    Test that `get_task_grads()` correctly computes task-specific gradients at each
    region of the network in the case of a fully shared network, when inference for each
    input is run for all tasks.
    """
    splits_args = []
    gradients_template(BASE_SETTINGS, splits_args, all_tasks=True)


def test_task_grads_single() -> None:
    """
    Test that `get_task_grads()` correctly computes task-specific gradients at each
    region of the network in the case of a single split network.
    """
    splits_args = [
        {"region": 1, "copy": 0, "group1": [0, 3], "group2": [1, 2]},
    ]
    gradients_template(BASE_SETTINGS, splits_args)


def test_task_grads_single_all_tasks() -> None:
    """
    Test that `get_task_grads()` correctly computes task-specific gradients at each
    region of the network in the case of a single split network, when inference is run
    for all tasks.
    """
    splits_args = [
        {"region": 1, "copy": 0, "group1": [0, 3], "group2": [1, 2]},
    ]
    gradients_template(BASE_SETTINGS, splits_args, all_tasks=True)


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
    gradients_template(BASE_SETTINGS, splits_args)


def test_task_grads_multiple_all_tasks() -> None:
    """
    Test that `get_task_grads()` correctly computes task-specific gradients at each
    region of the network in the case of a multiple split network, when inference is run
    for all tasks.
    """
    splits_args = [
        {"region": 0, "copy": 0, "group1": [0, 1], "group2": [2, 3]},
        {"region": 1, "copy": 0, "group1": [0, 2], "group2": [1, 3]},
        {"region": 1, "copy": 0, "group1": [0], "group2": [2]},
        {"region": 2, "copy": 0, "group1": [0, 3], "group2": [1, 2]},
    ]
    gradients_template(BASE_SETTINGS, splits_args, all_tasks=True)


def test_task_grad_diffs_zero_euclidean() -> None:
    """
    Test that `get_task_grad_diffs()` correctly computes the pairwise Euclidean distance
    between task-specific gradients at each region when these gradients are hard-coded
    to zero.
    """
    settings = dict(BASE_SETTINGS)
    settings["metric"] = "sqeuclidean"
    grad_diffs_template(settings, "zero")


def test_task_grad_diffs_rand_identical_euclidean() -> None:
    """
    Test that `get_task_grad_diffs()` correctly computes the pairwise Euclidean distance
    between task-specific gradients at each region when these gradients are random, but
    identical across tasks.
    """
    settings = dict(BASE_SETTINGS)
    settings["metric"] = "sqeuclidean"
    grad_diffs_template(settings, "rand_identical")


def test_task_grad_diffs_rand_euclidean() -> None:
    """
    Test that `get_task_grad_diffs()` correctly computes the pairwise Euclidean distance
    between task-specific gradients at each region when these gradients are random.
    """
    settings = dict(BASE_SETTINGS)
    settings["metric"] = "sqeuclidean"
    grad_diffs_template(settings, "rand")


def test_task_grad_diffs_zero_cosine() -> None:
    """
    Test that `get_task_grad_diffs()` correctly computes the pairwise cosine distance
    between task-specific gradients at each region when these gradients are hard-coded
    to zero.
    """
    settings = dict(BASE_SETTINGS)
    settings["metric"] = "cosine"
    grad_diffs_template(settings, "zero")


def test_task_grad_diffs_rand_identical_cosine() -> None:
    """
    Test that `get_task_grad_diffs()` correctly computes the pairwise cosine distance
    between task-specific gradients at each region when these gradients are random, but
    identical across tasks.
    """
    settings = dict(BASE_SETTINGS)
    settings["metric"] = "cosine"
    grad_diffs_template(settings, "rand_identical")


def test_task_grad_diffs_rand_cosine() -> None:
    """
    Test that `get_task_grad_diffs()` correctly computes the pairwise cosine distance
    between task-specific gradients at each region when these gradients are random.
    """
    settings = dict(BASE_SETTINGS)
    settings["metric"] = "cosine"
    grad_diffs_template(settings, "rand")


def test_task_grad_stats_zero_euclidean_shared() -> None:
    """
    Test that `update_grad_stats()` correctly computes gradient statistics over multiple
    steps when the gradients are always zero, with a fully shared network.
    """

    # Set up case.
    settings = dict(BASE_SETTINGS)
    settings["metric"] = "sqeuclidean"
    settings["hidden_size"] = settings["obs_dim"] + settings["num_tasks"] + 2
    ema_threshold = alpha_to_threshold(settings["ema_alpha"])

    # Construct series of splits.
    splits_args = []

    # Construct a sequence of task gradients.
    settings["num_steps"] = max(settings["split_step_threshold"], ema_threshold) + 20
    dim = settings["obs_dim"] + settings["num_tasks"]
    region_sizes = get_region_sizes(settings)
    task_grads = make_task_gradients(
        "zero",
        settings["num_steps"],
        settings["num_tasks"],
        settings["num_layers"],
        region_sizes,
    )

    # Run test.
    grad_stats_template(settings, task_grads, splits_args)


def test_task_grad_stats_rand_zero_euclidean_shared() -> None:
    """
    Test that `update_grad_stats()` correctly computes gradient statistics over multiple
    steps when the gradients are random, while some tasks randomly have gradients set to
    zero, with a fully shared network.
    """

    # Set up case.
    settings = dict(BASE_SETTINGS)
    settings["metric"] = "sqeuclidean"
    settings["hidden_size"] = settings["obs_dim"] + settings["num_tasks"] + 2
    ema_threshold = alpha_to_threshold(settings["ema_alpha"])

    # Construct series of splits.
    splits_args = []

    # Construct a sequence of task gradients.
    settings["num_steps"] = max(settings["split_step_threshold"], ema_threshold) + 20
    dim = settings["obs_dim"] + settings["num_tasks"]
    region_sizes = get_region_sizes(settings)
    task_grads = make_task_gradients(
        "rand_zero",
        settings["num_steps"],
        settings["num_tasks"],
        settings["num_layers"],
        region_sizes,
    )

    # Run test.
    grad_stats_template(settings, task_grads, splits_args)


def test_task_grad_stats_rand_euclidean_shared() -> None:
    """
    Test that `update_grad_stats()` correctly computes gradient statistics over multiple
    steps when these gradients are random, with a fully shared network.
    """

    # Set up case.
    settings = dict(BASE_SETTINGS)
    settings["metric"] = "sqeuclidean"
    settings["hidden_size"] = settings["obs_dim"] + settings["num_tasks"] + 2
    ema_threshold = alpha_to_threshold(settings["ema_alpha"])

    # Construct series of splits.
    splits_args = []

    # Construct a sequence of task gradients.
    settings["num_steps"] = max(settings["split_step_threshold"], ema_threshold) + 20
    dim = settings["obs_dim"] + settings["num_tasks"]
    region_sizes = get_region_sizes(settings)
    task_grads = make_task_gradients(
        "rand",
        settings["num_steps"],
        settings["num_tasks"],
        settings["num_layers"],
        region_sizes,
    )

    # Run test.
    grad_stats_template(settings, task_grads, splits_args)


def test_task_grad_stats_zero_euclidean_split() -> None:
    """
    Test that `update_grad_stats()` correctly computes gradient statistics over multiple
    steps when the gradients are always zero, with a split network.
    """

    # Set up case.
    settings = dict(BASE_SETTINGS)
    settings["metric"] = "sqeuclidean"
    settings["hidden_size"] = settings["obs_dim"] + settings["num_tasks"] + 2
    ema_threshold = alpha_to_threshold(settings["ema_alpha"])

    # Construct series of splits.
    splits_args = [
        {"region": 0, "copy": 0, "group1": [0, 1], "group2": [2, 3]},
        {"region": 1, "copy": 0, "group1": [0, 2], "group2": [1, 3]},
        {"region": 1, "copy": 1, "group1": [1], "group2": [3]},
        {"region": 2, "copy": 0, "group1": [0, 3], "group2": [1, 2]},
    ]

    # Construct a sequence of task gradients.
    settings["num_steps"] = max(settings["split_step_threshold"], ema_threshold) + 20
    dim = settings["obs_dim"] + settings["num_tasks"]
    region_sizes = get_region_sizes(settings)
    task_grads = make_task_gradients(
        "zero",
        settings["num_steps"],
        settings["num_tasks"],
        settings["num_layers"],
        region_sizes,
    )

    # Run test.
    grad_stats_template(settings, task_grads, splits_args)


def test_task_grad_stats_rand_zero_euclidean_split() -> None:
    """
    Test that `update_grad_stats()` correctly computes gradient statistics over multiple
    steps when the gradients are random, while some tasks randomly have gradients set to
    zero, with a split network.
    """

    # Set up case.
    settings = dict(BASE_SETTINGS)
    settings["metric"] = "sqeuclidean"
    settings["hidden_size"] = settings["obs_dim"] + settings["num_tasks"] + 2
    ema_threshold = alpha_to_threshold(settings["ema_alpha"])

    # Construct series of splits.
    splits_args = [
        {"region": 0, "copy": 0, "group1": [0, 1], "group2": [2, 3]},
        {"region": 1, "copy": 0, "group1": [0, 2], "group2": [1, 3]},
        {"region": 1, "copy": 1, "group1": [1], "group2": [3]},
        {"region": 2, "copy": 0, "group1": [0, 3], "group2": [1, 2]},
    ]

    # Construct a sequence of task gradients.
    settings["num_steps"] = max(settings["split_step_threshold"], ema_threshold) + 20
    dim = settings["obs_dim"] + settings["num_tasks"]
    region_sizes = get_region_sizes(settings)
    task_grads = make_task_gradients(
        "rand_zero",
        settings["num_steps"],
        settings["num_tasks"],
        settings["num_layers"],
        region_sizes,
    )

    # Run test.
    grad_stats_template(settings, task_grads, splits_args)


def test_task_grad_stats_rand_euclidean_split() -> None:
    """
    Test that `update_grad_stats()` correctly computes gradient statistics over multiple
    steps when these gradients are random, with a split network.
    """

    # Set up case.
    settings = dict(BASE_SETTINGS)
    settings["metric"] = "sqeuclidean"
    settings["hidden_size"] = settings["obs_dim"] + settings["num_tasks"] + 2
    ema_threshold = alpha_to_threshold(settings["ema_alpha"])

    # Construct series of splits.
    splits_args = [
        {"region": 0, "copy": 0, "group1": [0, 1], "group2": [2, 3]},
        {"region": 1, "copy": 0, "group1": [0, 2], "group2": [1, 3]},
        {"region": 1, "copy": 1, "group1": [1], "group2": [3]},
        {"region": 2, "copy": 0, "group1": [0, 3], "group2": [1, 2]},
    ]

    # Construct a sequence of task gradients.
    settings["num_steps"] = max(settings["split_step_threshold"], ema_threshold) + 20
    dim = settings["obs_dim"] + settings["num_tasks"]
    region_sizes = get_region_sizes(settings)
    task_grads = make_task_gradients(
        "rand",
        settings["num_steps"],
        settings["num_tasks"],
        settings["num_layers"],
        region_sizes,
    )

    # Run test.
    grad_stats_template(settings, task_grads, splits_args)


def test_task_grad_stats_zero_cosine_shared() -> None:
    """
    Test that `update_grad_stats()` correctly computes gradient statistics over multiple
    steps when the gradients are always zero, with a fully shared network using cosine
    distance.
    """

    # Set up case.
    settings = dict(BASE_SETTINGS)
    settings["metric"] = "cosine"
    settings["hidden_size"] = settings["obs_dim"] + settings["num_tasks"] + 2
    ema_threshold = alpha_to_threshold(settings["ema_alpha"])

    # Construct series of splits.
    splits_args = []

    # Construct a sequence of task gradients.
    settings["num_steps"] = max(settings["split_step_threshold"], ema_threshold) + 20
    dim = settings["obs_dim"] + settings["num_tasks"]
    region_sizes = get_region_sizes(settings)
    task_grads = make_task_gradients(
        "zero",
        settings["num_steps"],
        settings["num_tasks"],
        settings["num_layers"],
        region_sizes,
    )

    # Run test.
    grad_stats_template(settings, task_grads, splits_args)


def test_task_grad_stats_rand_zero_cosine_shared() -> None:
    """
    Test that `update_grad_stats()` correctly computes gradient statistics over multiple
    steps when the gradients are random, while some tasks randomly have gradients set to
    zero, with a fully shared network using cosine distance.
    """

    # Set up case.
    settings = dict(BASE_SETTINGS)
    settings["metric"] = "cosine"
    settings["hidden_size"] = settings["obs_dim"] + settings["num_tasks"] + 2
    ema_threshold = alpha_to_threshold(settings["ema_alpha"])

    # Construct series of splits.
    splits_args = []

    # Construct a sequence of task gradients.
    settings["num_steps"] = max(settings["split_step_threshold"], ema_threshold) + 20
    dim = settings["obs_dim"] + settings["num_tasks"]
    region_sizes = get_region_sizes(settings)
    task_grads = make_task_gradients(
        "rand_zero",
        settings["num_steps"],
        settings["num_tasks"],
        settings["num_layers"],
        region_sizes,
    )

    # Run test.
    grad_stats_template(settings, task_grads, splits_args)


def test_task_grad_stats_rand_cosine_shared() -> None:
    """
    Test that `update_grad_stats()` correctly computes gradient statistics over multiple
    steps when these gradients are random, with a fully shared network using cosine
    distance.
    """

    # Set up case.
    settings = dict(BASE_SETTINGS)
    settings["metric"] = "cosine"
    settings["hidden_size"] = settings["obs_dim"] + settings["num_tasks"] + 2
    ema_threshold = alpha_to_threshold(settings["ema_alpha"])

    # Construct series of splits.
    splits_args = []

    # Construct a sequence of task gradients.
    settings["num_steps"] = max(settings["split_step_threshold"], ema_threshold) + 20
    dim = settings["obs_dim"] + settings["num_tasks"]
    region_sizes = get_region_sizes(settings)
    task_grads = make_task_gradients(
        "rand",
        settings["num_steps"],
        settings["num_tasks"],
        settings["num_layers"],
        region_sizes,
    )

    # Run test.
    grad_stats_template(settings, task_grads, splits_args)


def test_task_grad_stats_zero_cosine_split() -> None:
    """
    Test that `update_grad_stats()` correctly computes gradient statistics over multiple
    steps when the gradients are always zero, with a split network using cosine
    distance.
    """

    # Set up case.
    settings = dict(BASE_SETTINGS)
    settings["metric"] = "cosine"
    settings["hidden_size"] = settings["obs_dim"] + settings["num_tasks"] + 2
    ema_threshold = alpha_to_threshold(settings["ema_alpha"])

    # Construct series of splits.
    splits_args = [
        {"region": 0, "copy": 0, "group1": [0, 1], "group2": [2, 3]},
        {"region": 1, "copy": 0, "group1": [0, 2], "group2": [1, 3]},
        {"region": 1, "copy": 1, "group1": [1], "group2": [3]},
        {"region": 2, "copy": 0, "group1": [0, 3], "group2": [1, 2]},
    ]

    # Construct a sequence of task gradients.
    settings["num_steps"] = max(settings["split_step_threshold"], ema_threshold) + 20
    dim = settings["obs_dim"] + settings["num_tasks"]
    region_sizes = get_region_sizes(settings)
    task_grads = make_task_gradients(
        "zero",
        settings["num_steps"],
        settings["num_tasks"],
        settings["num_layers"],
        region_sizes,
    )

    # Run test.
    grad_stats_template(settings, task_grads, splits_args)


def test_task_grad_stats_rand_zero_cosine_split() -> None:
    """
    Test that `update_grad_stats()` correctly computes gradient statistics over multiple
    steps when the gradients are random, while some tasks randomly have gradients set to
    zero, with a split network using cosine distance.
    """

    # Set up case.
    settings = dict(BASE_SETTINGS)
    settings["metric"] = "cosine"
    settings["hidden_size"] = settings["obs_dim"] + settings["num_tasks"] + 2
    ema_threshold = alpha_to_threshold(settings["ema_alpha"])

    # Construct series of splits.
    splits_args = [
        {"region": 0, "copy": 0, "group1": [0, 1], "group2": [2, 3]},
        {"region": 1, "copy": 0, "group1": [0, 2], "group2": [1, 3]},
        {"region": 1, "copy": 1, "group1": [1], "group2": [3]},
        {"region": 2, "copy": 0, "group1": [0, 3], "group2": [1, 2]},
    ]

    # Construct a sequence of task gradients.
    settings["num_steps"] = max(settings["split_step_threshold"], ema_threshold) + 20
    dim = settings["obs_dim"] + settings["num_tasks"]
    region_sizes = get_region_sizes(settings)
    task_grads = make_task_gradients(
        "rand_zero",
        settings["num_steps"],
        settings["num_tasks"],
        settings["num_layers"],
        region_sizes,
    )

    # Run test.
    grad_stats_template(settings, task_grads, splits_args)


def test_task_grad_stats_rand_cosine_split() -> None:
    """
    Test that `update_grad_stats()` correctly computes gradient statistics over multiple
    steps when these gradients are random, with a split network using cosine distance.
    """

    # Set up case.
    settings = dict(BASE_SETTINGS)
    settings["metric"] = "cosine"
    settings["hidden_size"] = settings["obs_dim"] + settings["num_tasks"] + 2
    ema_threshold = alpha_to_threshold(settings["ema_alpha"])

    # Construct series of splits.
    splits_args = [
        {"region": 0, "copy": 0, "group1": [0, 1], "group2": [2, 3]},
        {"region": 1, "copy": 0, "group1": [0, 2], "group2": [1, 3]},
        {"region": 1, "copy": 1, "group1": [1], "group2": [3]},
        {"region": 2, "copy": 0, "group1": [0, 3], "group2": [1, 2]},
    ]

    # Construct a sequence of task gradients.
    settings["num_steps"] = max(settings["split_step_threshold"], ema_threshold) + 20
    dim = settings["obs_dim"] + settings["num_tasks"]
    region_sizes = get_region_sizes(settings)
    task_grads = make_task_gradients(
        "rand",
        settings["num_steps"],
        settings["num_tasks"],
        settings["num_layers"],
        region_sizes,
    )

    # Run test.
    grad_stats_template(settings, task_grads, splits_args)


def test_sharing_score_shared() -> None:
    """
    Test that the sharing score is correctly computed for a fully shared network.
    """

    # Set up case.
    settings = dict(BASE_SETTINGS)
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
    settings = dict(BASE_SETTINGS)
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
    settings = dict(BASE_SETTINGS)
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
    settings = dict(BASE_SETTINGS)
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


def test_shared_regions_shared() -> None:
    """
    Test that the shared regions are correctly computed by
    `SplittingMap.shared_regions()` in the case of a fully shared network.
    """

    # Construct network.
    dim = BASE_SETTINGS["obs_dim"] + BASE_SETTINGS["num_tasks"]
    network = BaseMultiTaskSplittingNetwork(
        input_size=dim,
        output_size=dim,
        num_tasks=BASE_SETTINGS["num_tasks"],
        num_layers=BASE_SETTINGS["num_layers"],
        hidden_size=dim,
        device=BASE_SETTINGS["device"],
    )

    # Compute expected shared regions.
    expected_is_shared = torch.zeros(
        network.num_tasks, network.num_tasks, network.num_regions
    )
    for task1, task2 in product(range(network.num_tasks), range(network.num_tasks)):
        if task1 == task2:
            continue
        for region in range(network.num_regions):
            expected_is_shared[task1, task2, region] = 1

    # Compare expected to actual.
    assert torch.all(expected_is_shared == network.splitting_map.shared_regions())


def test_shared_regions_single() -> None:
    """
    Test that the shared regions are correctly computed by
    `SplittingMap.shared_regions()` in the case of a network with a single split.
    """

    # Construct network.
    dim = BASE_SETTINGS["obs_dim"] + BASE_SETTINGS["num_tasks"]
    network = BaseMultiTaskSplittingNetwork(
        input_size=dim,
        output_size=dim,
        num_tasks=BASE_SETTINGS["num_tasks"],
        num_layers=BASE_SETTINGS["num_layers"],
        hidden_size=dim,
        device=BASE_SETTINGS["device"],
    )

    # Perform splits.
    network.split(1, 0, [0, 1], [2, 3])

    # Compute expected shared regions.
    expected_is_shared = torch.zeros(
        network.num_tasks, network.num_tasks, network.num_regions
    )
    for task1, task2 in product(range(network.num_tasks), range(network.num_tasks)):
        if task1 == task2:
            continue
        for region in range(network.num_regions):
            if region == 1 and (task1 // 2) != (task2 // 2):
                expected_is_shared[task1, task2, region] = 0
            else:
                expected_is_shared[task1, task2, region] = 1

    # Compare expected to actual.
    print(expected_is_shared)
    print(network.splitting_map.shared_regions())
    assert torch.all(expected_is_shared == network.splitting_map.shared_regions())


def test_shared_regions_multiple() -> None:
    """
    Test that the shared regions are correctly computed by
    `SplittingMap.shared_regions()` in the case of a network with a single split.
    """

    # Construct network.
    dim = BASE_SETTINGS["obs_dim"] + BASE_SETTINGS["num_tasks"]
    network = BaseMultiTaskSplittingNetwork(
        input_size=dim,
        output_size=dim,
        num_tasks=BASE_SETTINGS["num_tasks"],
        num_layers=BASE_SETTINGS["num_layers"],
        hidden_size=dim,
        device=BASE_SETTINGS["device"],
    )

    # Perform splits.
    network.split(0, 0, [0, 1], [2, 3])
    network.split(1, 0, [0, 2], [1, 3])
    network.split(1, 0, [0], [2])
    network.split(2, 0, [0, 3], [1, 2])

    # Compute expected shared regions.
    expected_is_shared = torch.zeros(
        network.num_tasks, network.num_tasks, network.num_regions
    )
    for task1, task2 in product(range(network.num_tasks), range(network.num_tasks)):
        if task1 == task2:
            continue
        for region in range(network.num_regions):
            val = 1
            if region == 0 and (task1 // 2) != (task2 // 2):
                val = 0
            elif region == 1 and (task1, task2) not in [(1, 3), (3, 1)]:
                val = 0
            elif region == 2 and task1 + task2 != 3:
                val = 0

            expected_is_shared[task1, task2, region] = val

    # Compare expected to actual.
    assert torch.all(expected_is_shared == network.splitting_map.shared_regions())


def test_load_architecture() -> None:
    """ Test that the architecture of a saved splitting network is correctly loaded. """
    checkpoint_path = "results/test_load_architecture/checkpoint.pkl"
    load_template(BASE_SETTINGS, checkpoint_path, load_initialization=False)


def test_load_architecture_initialization() -> None:
    """
    Test that the architecture and initialization of a saved splitting network is
    correctly loaded.
    """
    checkpoint_path = "results/test_load_architecture_initialization/checkpoint.pkl"
    load_template(BASE_SETTINGS, checkpoint_path, load_initialization=True)


def test_load_architecture_initialization_gpu() -> None:
    """
    Test that the architecture and initialization of a saved splitting network is
    correctly loaded when networks are stored on GPU.
    """
    checkpoint_path = "results/test_load_architecture_initialization/checkpoint.pkl"
    BASE_SETTINGS["device"] = torch.device("cuda:0")
    load_template(BASE_SETTINGS, checkpoint_path, load_initialization=True)


def make_task_gradients(
    grad_type: str,
    num_steps: int,
    num_tasks: int,
    num_layers: int,
    region_sizes: List[int],
) -> torch.Tensor:
    """ Construct dummy task gradients. """

    # Generate gradients.
    max_layer_size = max(region_sizes)
    if grad_type == "zero":
        task_grads = torch.zeros(num_steps, num_tasks, num_layers, max_layer_size)
    elif grad_type == "rand_zero":
        task_grads = torch.rand(num_steps, num_tasks, num_layers, max_layer_size)
        task_grads *= (
            (torch.rand(num_steps, num_tasks) < 0.5).unsqueeze(-1).unsqueeze(-1)
        )
    elif grad_type == "rand_identical":
        task_grads = torch.rand(num_steps, 1, num_layers, max_layer_size)
        task_grads = task_grads.expand(-1, num_tasks, -1, -1)
    elif grad_type == "rand":
        task_grads = torch.rand(num_steps, num_tasks, num_layers, max_layer_size)
    else:
        raise NotImplementedError

    # Zero out values that don't correspond to a parameter (this happens since layers
    # have different sizes).
    for layer in range(num_layers):
        task_grads[:, :, layer, region_sizes[layer] :] = 0.0

    return task_grads


def get_region_sizes(settings: Dict[str, Any]) -> List[int]:
    """ Compute size of each layer in network specified by `settings`. """

    dim = settings["num_tasks"] + settings["obs_dim"]
    region_sizes = []
    for region in range(settings["num_layers"]):
        if region == 0:
            region_size = settings["hidden_size"] * (dim + 1)
        elif region == settings["num_layers"] - 1:
            region_size = dim * (settings["hidden_size"] + 1)
        else:
            region_size = settings["hidden_size"] ** 2 + settings["hidden_size"]
        region_sizes.append(region_size)

    return region_sizes
