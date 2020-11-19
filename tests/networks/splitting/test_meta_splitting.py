"""
Unit tests for meta/networks/splitting/meta_splitting.py.
"""

import numpy as np
import torch
from gym.spaces import Box

from meta.networks.utils import init_base
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
        meta_network.alpha[1][1, task] = 4

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

            layer_input = x.clone().detach()
            x = (layer + 1) * layer_input + (layer + 1)
            if layer != meta_network.num_layers - 1:
                x = torch.tanh(x)
            x *= layer + task + 1

            if layer == 1:
                y = 4 * torch.tanh(-2.0 * layer_input - 2.0)
                x += y

        expected_output[i] = x

    # Test output of network.
    assert torch.allclose(output, expected_output)


def test_forward_multiple() -> None:
    """ Test forward() when none of the layers are fully shared. """

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

    # Split the network at multiple layers.
    multitask_network.split(0, 0, [0, 1], [2, 3])
    multitask_network.split(1, 0, [0, 2], [1, 3])
    multitask_network.split(1, 0, [0], [2])
    multitask_network.split(2, 0, [0, 3], [1, 2])

    # Set multi-task network weights.
    state_dict = multitask_network.state_dict()
    for layer in range(BASE_SETTINGS["num_layers"]):
        for copy in range(3):
            weight_name = "regions.%d.%d.0.weight" % (layer, copy)
            bias_name = "regions.%d.%d.0.bias" % (layer, copy)
            if weight_name not in state_dict:
                continue

            if copy == 0:
                state_dict[weight_name] = torch.Tensor((layer + 1) * np.identity(dim))
                state_dict[bias_name] = torch.Tensor((layer + 1) * np.ones(dim))
            elif copy == 1:
                state_dict[weight_name] = torch.Tensor(-(layer + 1) * np.identity(dim))
                state_dict[bias_name] = torch.Tensor(-(layer + 1) * np.ones(dim))
            elif copy == 2:
                state_dict[weight_name] = torch.Tensor(
                    1 / (layer + 1) * np.identity(dim)
                )
                state_dict[bias_name] = torch.Tensor(1 / (layer + 1) * np.ones(dim))
            else:
                raise NotImplementedError
    multitask_network.load_state_dict(state_dict)

    # Construct MetaSplittingNetwork from BaseMultiTaskSplittingNetwork.
    meta_network = MetaSplittingNetwork(
        multitask_network, device=BASE_SETTINGS["device"]
    )

    # Set alpha weights in the meta network.
    alphas = [0.1, -0.5, 1.0]
    for task in range(meta_network.num_tasks):
        for layer in range(meta_network.num_layers):
            for copy in range(int(meta_network.splitting_map.num_copies[layer])):
                idx = (task + layer + copy) % 3
                meta_network.alpha[layer][copy, task] = alphas[idx]

    # Construct batch of observations concatenated with one-hot task vectors.
    obs, task_indices = get_obs_batch(
        batch_size=BASE_SETTINGS["num_processes"],
        obs_space=observation_subspace,
        num_tasks=BASE_SETTINGS["num_tasks"],
    )

    # Get output of network.
    output = meta_network(obs, task_indices)

    # Computed expected output of network.
    expected_output = torch.zeros(obs.shape)
    for i, (ob, task) in enumerate(zip(obs, task_indices)):
        x = ob
        for layer in range(meta_network.num_layers):

            copy_outputs = []

            # Copies 1 and 2
            copy_outputs.append((layer + 1) * x + (layer + 1))
            copy_outputs.append(-(layer + 1) * x - (layer + 1))

            # Copy 3, if necessary.
            if meta_network.splitting_map.num_copies[layer] > 2:
                copy_outputs.append(x / (layer + 1) + (1 / (layer + 1)))

            # Activation functions.
            if layer != meta_network.num_layers - 1:
                for j in range(len(copy_outputs)):
                    copy_outputs[j] = torch.tanh(copy_outputs[j])

            # Compute layer output by combining copy outputs.
            layer_output = alphas[(task + layer) % 3] * copy_outputs[0]
            layer_output += alphas[(task + layer + 1) % 3] * copy_outputs[1]
            if meta_network.splitting_map.num_copies[layer] > 2:
                layer_output += alphas[(task + layer + 2) % 3] * copy_outputs[2]

            x = layer_output

        expected_output[i] = x

    # Test output of network.
    assert torch.allclose(output, expected_output)


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
