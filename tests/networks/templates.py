"""
Templates for tests in tests/networks.
"""

from typing import List, Dict, Any
from gym.spaces import Box

import numpy as np
import torch

from meta.networks.initialize import init_base
from meta.networks.splitting import SplittingMLPNetwork
from tests.helpers import DEFAULT_SETTINGS, get_obs_batch


def gradients_template(
    settings: Dict[str, Any], splits_args: List[Dict[str, Any]]
) -> None:
    """
    Template to test that `get_task_grads()` correctly computes task-specific gradients
    at each region of the network. For simplicity we compute the loss as half of the
    squared norm of the output, and we make the following assumptions: each layer has
    the same size, the activation function is Tanh for each layer, and the final layer
    has no activation.
    """

    # Set up case.
    dim = settings["obs_dim"] + settings["num_tasks"]
    observation_subspace = Box(low=-np.inf, high=np.inf, shape=(settings["obs_dim"],))
    observation_subspace.seed(DEFAULT_SETTINGS["seed"])
    hidden_size = dim

    # Construct network.
    network = SplittingMLPNetwork(
        input_size=dim,
        output_size=dim,
        init_base=init_base,
        init_final=init_base,
        num_tasks=settings["num_tasks"],
        num_layers=settings["num_layers"],
        hidden_size=hidden_size,
        device=settings["device"],
    )

    # Split the network according to `splits_args`.
    for split_args in splits_args:
        network.split(**split_args)

    # Register forward hooks to get activations later from each copy of each region.
    activation = {}

    def get_activation(name):
        def hook(model, ins, outs):
            activation[name] = outs.detach()

        return hook

    for region in range(network.num_regions):
        for copy in range(network.maps[region].num_copies):
            name = "regions.%d.%d" % (region, copy)
            network.regions[region][copy].register_forward_hook(get_activation(name))

    # Construct batch of observations concatenated with one-hot task vectors.
    obs, task_indices = get_obs_batch(
        batch_size=settings["num_processes"],
        obs_space=observation_subspace,
        num_tasks=settings["num_tasks"],
    )

    # Get output of network and compute task gradients.
    output = network(obs, task_indices)
    task_losses = torch.zeros(settings["num_tasks"])
    for task in range(settings["num_tasks"]):
        for current_out, current_task in zip(output, task_indices):
            if current_task == task:
                task_losses[task] += 0.5 * torch.sum(current_out ** 2)

    task_grads = network.get_task_grads(task_losses)

    def get_task_activations(r, t, tasks):
        """ Helper function to get activations from specific regions. """

        c = network.maps[r].module[t]
        copy_indices = network.maps[r].module[tasks]
        sorted_copy_indices, copy_permutation = torch.sort(copy_indices)
        sorted_tasks = tasks[copy_permutation]
        batch_indices = (sorted_copy_indices == c).nonzero().squeeze(-1)
        task_batch_indices = sorted_tasks[batch_indices]
        current_task_indices = (task_batch_indices == t).nonzero().squeeze(-1)
        activations = activation["regions.%d.%d" % (r, c)][current_task_indices]

        return activations

    # Compute expected gradients.
    state_dict = network.state_dict()
    expected_task_grads = torch.zeros(
        (settings["num_tasks"], network.num_regions, network.max_region_size)
    )
    for task in range(settings["num_tasks"]):

        # Get output from current task.
        task_input_indices = (task_indices == task).nonzero().squeeze(-1)
        task_output = output[task_input_indices]

        # Clear local gradients.
        local_grad = {}

        for region in reversed(range(network.num_regions)):

            # Get copy index and layer input.
            copy = network.maps[region].module[task]
            if region > 0:
                layer_input = get_task_activations(region - 1, task, task_indices)
            else:
                layer_input = obs[task_input_indices]

            # Compute local gradient first.
            if region == network.num_regions - 1:
                local_grad[region] = -task_output
            else:
                layer_output = get_task_activations(region, task, task_indices)
                local_grad[region] = torch.zeros(len(layer_output), dim)
                next_copy = network.maps[region + 1].module[task]
                weights = state_dict["regions.%d.%d.0.weight" % (region + 1, next_copy)]
                for i in range(dim):
                    for j in range(dim):
                        local_grad[region][:, i] += (
                            local_grad[region + 1][:, j] * weights[j, i]
                        )
                local_grad[region] = local_grad[region] * (1 - layer_output ** 2)

            # Compute gradient from local gradients.
            grad = torch.zeros(dim, dim + 1)
            for i in range(dim):
                for j in range(dim):
                    grad[i, j] = torch.sum(
                        -local_grad[region][:, i] * layer_input[:, j]
                    )
                grad[i, dim] = torch.sum(-local_grad[region][:, i])

            # Rearrange weights and biases. Should be all weights, then all biases.
            weights = torch.reshape(grad[:, :-1], (-1,))
            biases = torch.reshape(grad[:, -1], (-1,))
            grad = torch.cat([weights, biases])
            expected_task_grads[task, region, : len(grad)] = grad

    # Test gradients.
    assert torch.allclose(task_grads, expected_task_grads, atol=1e-6)


def backward_template(
    settings: Dict[str, Any], splits_args: List[Dict[str, Any]]
) -> None:
    """
    Template to test that the backward() function correctly computes gradients. We don't
    actually compare the gradients against baseline values, instead we just check that
    the gradient of the loss for task i is non-zero for all copies that i is assigned
    to, and zero for all copies i isn't assigned to, for each i. To keep things simple,
    we define each task loss as the squared norm of the output for inputs from the given
    task.
    """

    # Set up case.
    dim = settings["obs_dim"] + settings["num_tasks"]
    observation_subspace = Box(low=-np.inf, high=np.inf, shape=(settings["obs_dim"],))
    observation_subspace.seed(DEFAULT_SETTINGS["seed"])
    hidden_size = dim

    # Construct network.
    network = SplittingMLPNetwork(
        input_size=dim,
        output_size=dim,
        init_base=init_base,
        init_final=init_base,
        num_tasks=settings["num_tasks"],
        num_layers=settings["num_layers"],
        hidden_size=hidden_size,
        device=settings["device"],
    )

    # Split the network according to `splits_args`.
    for split_args in splits_args:
        network.split(**split_args)

    # Construct batch of observations concatenated with one-hot task vectors.
    obs, task_indices = get_obs_batch(
        batch_size=settings["num_processes"],
        obs_space=observation_subspace,
        num_tasks=settings["num_tasks"],
    )

    # Get output of network and compute task losses.
    output = network(obs, task_indices)
    task_losses = {i: None for i in range(settings["num_tasks"])}
    for task in range(settings["num_tasks"]):
        for current_out, current_task in zip(output, task_indices):
            if current_task == task:
                if task_losses[task] is not None:
                    task_losses[task] += torch.sum(current_out ** 2)
                else:
                    task_losses[task] = torch.sum(current_out ** 2)

    # Test gradients.
    for task in range(settings["num_tasks"]):
        network.zero_grad()
        if task_losses[task] is None:
            continue

        task_losses[task].backward(retain_graph=True)
        for region in range(len(network.regions)):
            for copy in range(network.maps[region].num_copies):
                for param in network.regions[region][copy].parameters():
                    zero = torch.zeros(param.grad.shape)
                    if network.maps[region].module[task] == copy:
                        assert not torch.allclose(param.grad, zero)
                    else:
                        assert torch.allclose(param.grad, zero)
