"""
Unit tests for meta/networks/splitting/meta_splitting.py.
"""

import numpy as np
import torch
from gym.spaces import Box

from meta.networks.initialize import init_base
from meta.networks.splitting import BaseMultiTaskSplittingNetwork, MetaSplittingNetwork
from tests.helpers import DEFAULT_SETTINGS, get_obs_batch
from tests.networks.splitting import BASE_SETTINGS


def test_forward_shared() -> None:
    """ Test forward() when all regions of the network are fully shared. """

    # Set up case.
    dim = BASE_SETTINGS["obs_dim"] + BASE_SETTINGS["num_tasks"]
    observation_subspace = Box(
        low=-np.inf, high=np.inf, shape=(BASE_SETTINGS["obs_dim"],)
    )
    observation_subspace.seed(DEFAULT_SETTINGS["seed"])
    hidden_size = dim

    # Construct multi-task network.
    multitask_network = BaseMultiTaskSplittingNetwork(
        input_size=dim,
        output_size=dim,
        init_base=init_base,
        init_final=init_base,
        num_tasks=BASE_SETTINGS["num_tasks"],
        num_layers=BASE_SETTINGS["num_layers"],
        hidden_size=hidden_size,
        device=BASE_SETTINGS["device"],
    )

    # Set multi-task network weights.
    state_dict = multitask_network.state_dict()
    for layer in range(multitask_network.num_layers):
        weight_name = "regions.%d.0.0.weight" % layer
        bias_name = "regions.%d.0.0.bias" % layer
        state_dict[weight_name] = torch.Tensor((layer + 1) * np.identity(dim))
        state_dict[bias_name] = torch.Tensor((layer + 1) * np.ones(dim))
    multitask_network.load_state_dict(state_dict)

    # Construct MetaSplittingNetwork from BaseMultiTaskSplittingNetwork.
    meta_network = MetaSplittingNetwork(
        multitask_network, device=BASE_SETTINGS["device"]
    )

    # Set alpha weights in the meta network.
    for layer in range(meta_network.num_layers):
        for task in range(meta_network.num_tasks):
            meta_network.alpha[layer][0, task] = layer + task + 1

    # Construct batch of observations concatenated with one-hot task vectors.
    obs, task_indices = get_obs_batch(
        batch_size=BASE_SETTINGS["num_processes"],
        obs_space=observation_subspace,
        num_tasks=BASE_SETTINGS["num_tasks"],
    )

    # Get output of network.
    output = meta_network(obs, task_indices)

    # Compute expected output of network.
    expected_output = torch.zeros(BASE_SETTINGS["num_processes"], dim)
    for i, (ob, task) in enumerate(zip(obs, task_indices)):
        x = ob
        for layer in range(meta_network.num_layers):
            x = (layer + 1) * x + (layer + 1)
            if layer != meta_network.num_layers - 1:
                x = torch.tanh(x)
            x *= layer + task + 1
        expected_output[i] = x

    # Test output of network.
    assert torch.allclose(output, expected_output)


def test_forward_single() -> None:
    """
    Test forward() when all regions of the splitting network are fully shared except
    one.
    """

    # Set up case.
    dim = BASE_SETTINGS["obs_dim"] + BASE_SETTINGS["num_tasks"]
    observation_subspace = Box(
        low=-np.inf, high=np.inf, shape=(BASE_SETTINGS["obs_dim"],)
    )
    observation_subspace.seed(DEFAULT_SETTINGS["seed"])
    hidden_size = dim

    # Construct multi-task network.
    multitask_network = BaseMultiTaskSplittingNetwork(
        input_size=dim,
        output_size=dim,
        init_base=init_base,
        init_final=init_base,
        num_tasks=BASE_SETTINGS["num_tasks"],
        num_layers=BASE_SETTINGS["num_layers"],
        hidden_size=hidden_size,
        device=BASE_SETTINGS["device"],
    )

    # Split the multi-task network at the second layer. Tasks 0 and 1 stay assigned to
    # the original copy and tasks 2 and 3 are assigned to the new copy.
    multitask_network.split(1, 0, [0, 1], [2, 3])

    # Set multi-task network weights.
    state_dict = multitask_network.state_dict()
    for layer in range(BASE_SETTINGS["num_layers"]):
        weight_name = "regions.%d.0.0.weight" % layer
        bias_name = "regions.%d.0.0.bias" % layer
        state_dict[weight_name] = torch.Tensor((layer + 1) * np.identity(dim))
        state_dict[bias_name] = torch.Tensor((layer + 1) * np.ones(dim))
    weight_name = "regions.1.1.0.weight"
    bias_name = "regions.1.1.0.bias"
    state_dict[weight_name] = torch.Tensor(-2 * np.identity(dim))
    state_dict[bias_name] = torch.Tensor(-2 * np.ones(dim))
    multitask_network.load_state_dict(state_dict)

    # Construct MetaSplittingNetwork from BaseMultiTaskSplittingNetwork.
    meta_network = MetaSplittingNetwork(
        multitask_network, device=BASE_SETTINGS["device"]
    )

    # Set alpha weights in the meta network.
    for task in range(meta_network.num_tasks):
        for layer in range(meta_network.num_layers):
            meta_network.alpha[layer][0, task] = layer + task + 1
        meta_network.alpha[1][1, task] = 3

    # Construct batch of observations concatenated with one-hot task vectors.
    obs, task_indices = get_obs_batch(
        batch_size=BASE_SETTINGS["num_processes"],
        obs_space=observation_subspace,
        num_tasks=BASE_SETTINGS["num_tasks"],
    )

    # Get output of network.
    output = meta_network(obs, task_indices)

    # Compute expected output of network.
    expected_output = torch.zeros(BASE_SETTINGS["num_processes"], dim)
    for i, (ob, task) in enumerate(zip(obs, task_indices)):
        x = ob
        for layer in range(meta_network.num_layers):
            x = (layer + 1) * x + (layer + 1)
            if layer != meta_network.num_layers - 1:
                x = torch.tanh(x)
            x *= layer + task + 1

            if layer == 1:
                y = 3.0 * torch.tanh(-2.0 * x - 2.0)
                x += y

        expected_output[i] = x

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

    """
    # Set up case.
    dim = BASE_SETTINGS["obs_dim"] + BASE_SETTINGS["num_tasks"]
    observation_subspace = Box(low=-np.inf, high=np.inf, shape=(BASE_SETTINGS["obs_dim"],))
    observation_subspace.seed(DEFAULT_SETTINGS["seed"])
    hidden_size = dim

    # Construct network.
    network = BaseMultiTaskSplittingNetwork(
        input_size=dim,
        output_size=dim,
        init_base=init_base,
        init_final=init_base,
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
    """
    raise NotImplementedError


def test_task_grads_shared() -> None:
    """
    Test that `get_task_grads()` correctly computes task-specific gradients at each
    region of the network in the case of a fully shared network.
    """

    """
    splits_args = []
    gradients_template(SETTINGS, splits_args)
    """
    raise NotImplementedError


def test_task_grads_single() -> None:
    """
    Test that `get_task_grads()` correctly computes task-specific gradients at each
    region of the network in the case of a single split network.
    """

    """
    splits_args = [
        {"region": 1, "copy": 0, "group1": [0, 3], "group2": [1, 2]},
    ]
    gradients_template(SETTINGS, splits_args)
    """
    raise NotImplementedError


def test_task_grads_multiple() -> None:
    """
    Test that `get_task_grads()` correctly computes task-specific gradients at each
    region of the network in the case of a multiple split network.
    """

    """
    splits_args = [
        {"region": 0, "copy": 0, "group1": [0, 1], "group2": [2, 3]},
        {"region": 1, "copy": 0, "group1": [0, 2], "group2": [1, 3]},
        {"region": 1, "copy": 0, "group1": [0], "group2": [2]},
        {"region": 2, "copy": 0, "group1": [0, 3], "group2": [1, 2]},
    ]
    gradients_template(SETTINGS, splits_args)
    """
    raise NotImplementedError
