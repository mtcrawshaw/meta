"""
Unit tests for meta/networks/splitting/meta_splitting.py.
"""

import numpy as np
import torch
from gym.spaces import Box

from tests.helpers import DEFAULT_SETTINGS, get_obs_batch
from tests.networks.splitting import BASE_SETTINGS
from tests.networks.splitting.templates import (
    meta_forward_template,
    meta_backward_template,
)


def test_forward_shared() -> None:
    """ Test forward() when all regions of the network are fully shared. """

    # Set up case.
    dim = BASE_SETTINGS["obs_dim"] + BASE_SETTINGS["num_tasks"]
    settings = {}
    settings["obs_dim"] = BASE_SETTINGS["obs_dim"]
    settings["num_processes"] = BASE_SETTINGS["num_processes"]
    settings["input_size"] = dim
    settings["output_size"] = dim
    settings["num_tasks"] = BASE_SETTINGS["num_tasks"]
    settings["num_layers"] = BASE_SETTINGS["num_layers"]
    settings["hidden_size"] = dim
    settings["device"] = BASE_SETTINGS["device"]
    settings["seed"] = DEFAULT_SETTINGS["seed"]

    # Construct list of splits to perform (none in this case).
    splits_args = []

    # Construct network state dict.
    state_dict = {}
    for layer in range(settings["num_layers"]):
        weight_name = "regions.%d.0.0.weight" % layer
        bias_name = "regions.%d.0.0.bias" % layer
        state_dict[weight_name] = torch.Tensor((layer + 1) * np.identity(dim))
        state_dict[bias_name] = torch.Tensor((layer + 1) * np.ones(dim))

    # Set alpha weights.
    num_copies = [1, 1, 1]
    alpha = [
        torch.zeros(num_copies[layer], settings["num_tasks"])
        for layer in range(settings["num_layers"])
    ]
    for layer in range(settings["num_layers"]):
        for task in range(settings["num_tasks"]):
            alpha[layer][0, task] = layer + task + 1

    # Compute expected output of network.
    def get_expected_output(
        obs: torch.Tensor, task_indices: torch.Tensor
    ) -> torch.Tensor:
        expected_output = torch.zeros(
            settings["num_processes"], settings["output_size"]
        )
        for i, (ob, task) in enumerate(zip(obs, task_indices)):
            x = ob
            for layer in range(settings["num_layers"]):
                x = (layer + 1) * x + (layer + 1)
                if layer != settings["num_layers"] - 1:
                    x = torch.tanh(x)
                x *= layer + task + 1
            expected_output[i] = x
        return expected_output

    # Call test template.
    meta_forward_template(settings, state_dict, splits_args, alpha, get_expected_output)


def test_forward_single() -> None:
    """
    Test forward() when all regions of the splitting network are fully shared except
    one.
    """

    # Set up case.
    dim = BASE_SETTINGS["obs_dim"] + BASE_SETTINGS["num_tasks"]
    settings = {}
    settings["obs_dim"] = BASE_SETTINGS["obs_dim"]
    settings["num_processes"] = BASE_SETTINGS["num_processes"]
    settings["input_size"] = dim
    settings["output_size"] = dim
    settings["num_tasks"] = BASE_SETTINGS["num_tasks"]
    settings["num_layers"] = BASE_SETTINGS["num_layers"]
    settings["hidden_size"] = dim
    settings["device"] = BASE_SETTINGS["device"]
    settings["seed"] = DEFAULT_SETTINGS["seed"]

    # Split the multi-task network at the second layer. Tasks 0 and 1 stay assigned to
    # the original copy and tasks 2 and 3 are assigned to the new copy.
    splits_args = [{"region": 1, "copy": 0, "group1": [0, 1], "group2": [2, 3]}]

    # Construct network state dict.
    state_dict = {}
    for layer in range(settings["num_layers"]):
        weight_name = "regions.%d.0.0.weight" % layer
        bias_name = "regions.%d.0.0.bias" % layer
        state_dict[weight_name] = torch.Tensor((layer + 1) * np.identity(dim))
        state_dict[bias_name] = torch.Tensor((layer + 1) * np.ones(dim))
    weight_name = "regions.1.1.0.weight"
    bias_name = "regions.1.1.0.bias"
    state_dict[weight_name] = torch.Tensor(-2 * np.identity(dim))
    state_dict[bias_name] = torch.Tensor(-2 * np.ones(dim))

    # Set alpha weights in the meta network.
    num_copies = [1, 2, 1]
    alpha = [
        torch.zeros(num_copies[layer], settings["num_tasks"])
        for layer in range(settings["num_layers"])
    ]
    for task in range(settings["num_tasks"]):
        for layer in range(settings["num_layers"]):
            alpha[layer][0, task] = layer + task + 1
        alpha[1][1, task] = 4

    # Compute expected output of network.
    def get_expected_output(
        obs: torch.Tensor, task_indices: torch.Tensor
    ) -> torch.Tensor:
        expected_output = torch.zeros(BASE_SETTINGS["num_processes"], dim)
        for i, (ob, task) in enumerate(zip(obs, task_indices)):
            x = ob
            for layer in range(settings["num_layers"]):

                layer_input = x.clone().detach()
                x = (layer + 1) * layer_input + (layer + 1)
                if layer != settings["num_layers"] - 1:
                    x = torch.tanh(x)
                x *= layer + task + 1

                if layer == 1:
                    y = 4 * torch.tanh(-2.0 * layer_input - 2.0)
                    x += y

            expected_output[i] = x
        return expected_output

    # Call test template.
    meta_forward_template(settings, state_dict, splits_args, alpha, get_expected_output)


def test_forward_multiple() -> None:
    """ Test forward() when none of the layers are fully shared. """

    # Set up case.
    dim = BASE_SETTINGS["obs_dim"] + BASE_SETTINGS["num_tasks"]
    settings = {}
    settings["obs_dim"] = BASE_SETTINGS["obs_dim"]
    settings["num_processes"] = BASE_SETTINGS["num_processes"]
    settings["input_size"] = dim
    settings["output_size"] = dim
    settings["num_tasks"] = BASE_SETTINGS["num_tasks"]
    settings["num_layers"] = BASE_SETTINGS["num_layers"]
    settings["hidden_size"] = dim
    settings["device"] = BASE_SETTINGS["device"]
    settings["seed"] = DEFAULT_SETTINGS["seed"]

    # Split the network at multiple layers.
    splits_args = [
        {"region": 0, "copy": 0, "group1": [0, 1], "group2": [2, 3]},
        {"region": 1, "copy": 0, "group1": [0, 2], "group2": [1, 3]},
        {"region": 1, "copy": 0, "group1": [0], "group2": [2]},
        {"region": 2, "copy": 0, "group1": [0, 3], "group2": [1, 2]},
    ]

    # Construct network state dict.
    num_copies = [2, 3, 2]
    state_dict = {}
    for layer in range(settings["num_layers"]):
        for copy in range(num_copies[layer]):
            weight_name = "regions.%d.%d.0.weight" % (layer, copy)
            bias_name = "regions.%d.%d.0.bias" % (layer, copy)

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

    # Set alpha weights in the meta network.
    alpha = [
        torch.zeros(num_copies[layer], settings["num_tasks"])
        for layer in range(settings["num_layers"])
    ]
    alphas = [0.1, -0.5, 1.0]
    for task in range(settings["num_tasks"]):
        for layer in range(settings["num_layers"]):
            for copy in range(num_copies[layer]):
                idx = (task + layer + copy) % 3
                alpha[layer][copy, task] = alphas[idx]

    # Computed expected output of network.
    def get_expected_output(
        obs: torch.Tensor, task_indices: torch.Tensor
    ) -> torch.Tensor:
        expected_output = torch.zeros(obs.shape)
        for i, (ob, task) in enumerate(zip(obs, task_indices)):
            x = ob
            for layer in range(settings["num_layers"]):

                copy_outputs = []

                # Copies 1 and 2
                copy_outputs.append((layer + 1) * x + (layer + 1))
                copy_outputs.append(-(layer + 1) * x - (layer + 1))

                # Copy 3, if necessary.
                if num_copies[layer] > 2:
                    copy_outputs.append(x / (layer + 1) + (1 / (layer + 1)))

                # Activation functions.
                if layer != settings["num_layers"] - 1:
                    for j in range(len(copy_outputs)):
                        copy_outputs[j] = torch.tanh(copy_outputs[j])

                # Compute layer output by combining copy outputs.
                layer_output = alphas[(task + layer) % 3] * copy_outputs[0]
                layer_output += alphas[(task + layer + 1) % 3] * copy_outputs[1]
                if num_copies[layer] > 2:
                    layer_output += alphas[(task + layer + 2) % 3] * copy_outputs[2]

                x = layer_output

            expected_output[i] = x
        return expected_output

    # Call test template.
    meta_forward_template(settings, state_dict, splits_args, alpha, get_expected_output)


def test_backward_shared() -> None:
    """
    Test that calling `backward()` correctly computes gradients for each parameter of
    the meta splitting network in the case of a fully shared network.
    """

    # Set up case.
    dim = BASE_SETTINGS["obs_dim"] + BASE_SETTINGS["num_tasks"]
    settings = {}
    settings["obs_dim"] = BASE_SETTINGS["obs_dim"]
    settings["num_processes"] = BASE_SETTINGS["num_processes"]
    settings["input_size"] = dim
    settings["output_size"] = dim
    settings["num_tasks"] = BASE_SETTINGS["num_tasks"]
    settings["num_layers"] = BASE_SETTINGS["num_layers"]
    settings["hidden_size"] = dim
    settings["device"] = BASE_SETTINGS["device"]
    settings["seed"] = DEFAULT_SETTINGS["seed"]

    # Split the network at multiple layers.
    num_copies = [1, 1, 1]
    splits_args = []

    # Set alpha weights in the meta network.
    alpha = [
        torch.zeros(num_copies[layer], settings["num_tasks"])
        for layer in range(settings["num_layers"])
    ]
    for layer in range(settings["num_layers"]):
        for task in range(settings["num_tasks"]):
            alpha[layer][0, task] = layer + task + 1

    # Call test template.
    meta_backward_template(settings, splits_args, alpha)


def test_backward_single() -> None:
    """
    Test that calling `backward()` correctly computes gradients for each parameter of
    the meta splitting network in the case of a single split network.
    """

    # Set up case.
    dim = BASE_SETTINGS["obs_dim"] + BASE_SETTINGS["num_tasks"]
    settings = {}
    settings["obs_dim"] = BASE_SETTINGS["obs_dim"]
    settings["num_processes"] = BASE_SETTINGS["num_processes"]
    settings["input_size"] = dim
    settings["output_size"] = dim
    settings["num_tasks"] = BASE_SETTINGS["num_tasks"]
    settings["num_layers"] = BASE_SETTINGS["num_layers"]
    settings["hidden_size"] = dim
    settings["device"] = BASE_SETTINGS["device"]
    settings["seed"] = DEFAULT_SETTINGS["seed"]

    # Split the network at multiple layers.
    num_copies = [1, 2, 1]
    splits_args = [
        {"region": 1, "copy": 0, "group1": [0, 3], "group2": [1, 2]},
    ]

    # Set alpha weights in the meta network.
    alpha = [
        torch.zeros(num_copies[layer], settings["num_tasks"])
        for layer in range(settings["num_layers"])
    ]
    for task in range(settings["num_tasks"]):
        for layer in range(settings["num_layers"]):
            alpha[layer][0, task] = layer + task + 1
        alpha[1][1, task] = 4

    # Call test template.
    meta_backward_template(settings, splits_args, alpha)


def test_backward_multiple() -> None:
    """
    Test that calling `backward()` correctly computes gradients for each parameter of
    the meta splitting network in the case of a multiple split network.
    """

    # Set up case.
    dim = BASE_SETTINGS["obs_dim"] + BASE_SETTINGS["num_tasks"]
    settings = {}
    settings["obs_dim"] = BASE_SETTINGS["obs_dim"]
    settings["num_processes"] = BASE_SETTINGS["num_processes"]
    settings["input_size"] = dim
    settings["output_size"] = dim
    settings["num_tasks"] = BASE_SETTINGS["num_tasks"]
    settings["num_layers"] = BASE_SETTINGS["num_layers"]
    settings["hidden_size"] = dim
    settings["device"] = BASE_SETTINGS["device"]
    settings["seed"] = DEFAULT_SETTINGS["seed"]

    # Split the network at multiple layers.
    num_copies = [2, 3, 2]
    splits_args = [
        {"region": 0, "copy": 0, "group1": [0, 1], "group2": [2, 3]},
        {"region": 1, "copy": 0, "group1": [0, 2], "group2": [1, 3]},
        {"region": 1, "copy": 0, "group1": [0], "group2": [2]},
        {"region": 2, "copy": 0, "group1": [0, 3], "group2": [1, 2]},
    ]

    # Set alpha weights in the meta network.
    alpha = [
        torch.zeros(num_copies[layer], settings["num_tasks"])
        for layer in range(settings["num_layers"])
    ]
    alphas = [0.1, -0.5, 1.0]
    for task in range(settings["num_tasks"]):
        for layer in range(settings["num_layers"]):
            for copy in range(num_copies[layer]):
                idx = (task + layer + copy) % 3
                alpha[layer][copy, task] = alphas[idx]

    # Call test template.
    meta_backward_template(settings, splits_args, alpha)
