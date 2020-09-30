"""
Unit tests for meta/networks/splitting.py.
"""

from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from gym.spaces import Box

from meta.networks.initialize import init_base
from meta.networks.splitting import SplittingMLPNetwork
from tests.helpers import DEFAULT_SETTINGS, get_obs_batch


SETTINGS = {
    "obs_dim": 8,
    "num_processes": 8,
    "num_tasks": 4,
    "num_layers": 3,
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
    backward_template(splits_args)


def test_backward_single() -> None:
    """
    Test that the backward() function correctly computes gradients in the case of a
    single split.
    """

    splits_args = [
        {"region": 1, "copy": 0, "group_1": [0, 3], "group_2": [1, 2]},
    ]
    backward_template(splits_args)


def test_backward_multiple() -> None:
    """
    Test that the backward() function correctly computes gradients in the case of
    multiple splits.
    """

    splits_args = [
        {"region": 0, "copy": 0, "group_1": [0, 1], "group_2": [2, 3]},
        {"region": 1, "copy": 0, "group_1": [0, 2], "group_2": [1, 3]},
        {"region": 1, "copy": 0, "group_1": [0], "group_2": [2]},
        {"region": 2, "copy": 0, "group_1": [0, 3], "group_2": [1, 2]},
    ]
    backward_template(splits_args)


def test_task_grads_shared() -> None:
    """
    Test that `get_task_grads()` correctly computes task-specific gradients at each
    region of the network in the case of a fully shared network.
    """

    splits_args = []
    gradients_template(splits_args)


def test_task_grads_single() -> None:
    """
    Test that `get_task_grads()` correctly computes task-specific gradients at each
    region of the network in the case of a single split network.
    """

    splits_args = [
        {"region": 1, "copy": 0, "group_1": [0, 3], "group_2": [1, 2]},
    ]
    gradients_template(splits_args)


def test_task_grads_multiple() -> None:
    """
    Test that `get_task_grads()` correctly computes task-specific gradients at each
    region of the network in the case of a multiple split network.
    """

    splits_args = [
        {"region": 0, "copy": 0, "group_1": [0, 1], "group_2": [2, 3]},
        {"region": 1, "copy": 0, "group_1": [0, 2], "group_2": [1, 3]},
        {"region": 1, "copy": 0, "group_1": [0], "group_2": [2]},
        {"region": 2, "copy": 0, "group_1": [0, 3], "group_2": [1, 2]},
    ]
    gradients_template(splits_args)


def gradients_template(splits_args: List[Dict[str, Any]]) -> None:
    """
    Template to test that `get_task_grads()` correctly computes task-specific gradients
    at each region of the network. For simplicity we compute the loss as half of the
    squared norm of the output, and we make the following assumptions: each layer has
    the same size, the activation function is Tanh for each layer, and the final layer
    has no activation.
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
        batch_size=SETTINGS["num_processes"],
        obs_space=observation_subspace,
        num_tasks=SETTINGS["num_tasks"],
    )

    # Get output of network and compute task gradients.
    output = network(obs, task_indices)
    task_losses = torch.zeros(SETTINGS["num_tasks"])
    for task in range(SETTINGS["num_tasks"]):
        for current_out, current_task in zip(output, task_indices):
            if current_task == task:
                task_losses[task] += 0.5 * torch.sum(current_out ** 2)

    task_grads = network.get_task_grads(task_losses)

    # Compute expected gradients.
    state_dict = network.state_dict()
    expected_task_grads = torch.zeros(
        (SETTINGS["num_tasks"], network.num_regions, network.max_region_size)
    )
    for task in range(SETTINGS["num_tasks"]):

        # Get output from current task.
        task_input_indices = (task_indices == task).nonzero().squeeze(-1)
        task_output = output[task_input_indices]

        # Clear local gradients.
        local_grad = {}

        for region in reversed(range(network.num_regions)):

            # Get copy index and layer input.
            copy = network.maps[region].module[task]
            if region > 0:
                prev_copy = network.maps[region - 1].module[task]
                layer_input = activation["regions.%d.%d" % (region - 1, prev_copy)]
                layer_input = layer_input[task_input_indices]
            else:
                layer_input = obs[task_input_indices]

            # Compute local gradient first.
            if region == network.num_regions - 1:
                local_grad[region] = -task_output
            else:
                layer_output = activation["regions.%d.%d" % (region, copy)]
                layer_output = layer_output[task_input_indices]

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


def backward_template(splits_args: List[Dict[str, Any]]) -> None:
    """
    Template to test that the backward() function correctly computes gradients. We don't
    actually compare the gradients against baseline values, instead we just check that
    the gradient of the loss for task i is non-zero for all copies that i is assigned
    to, and zero for all copies i isn't assigned to, for each i. To keep things simple,
    we define each task loss as the squared norm of the output for inputs from the given
    task.
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

    # Split the network according to `splits_args`.
    for split_args in splits_args:
        network.split(**split_args)

    # Construct batch of observations concatenated with one-hot task vectors.
    obs, task_indices = get_obs_batch(
        batch_size=SETTINGS["num_processes"],
        obs_space=observation_subspace,
        num_tasks=SETTINGS["num_tasks"],
    )

    # Get output of network and compute task losses.
    output = network(obs, task_indices)
    task_losses = {i: None for i in range(SETTINGS["num_tasks"])}
    for task in range(SETTINGS["num_tasks"]):
        for current_out, current_task in zip(output, task_indices):
            if current_task == task:
                if task_losses[task] is not None:
                    task_losses[task] += torch.sum(current_out ** 2)
                else:
                    task_losses[task] = torch.sum(current_out ** 2)

    # Test gradients.
    for task in range(SETTINGS["num_tasks"]):
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
