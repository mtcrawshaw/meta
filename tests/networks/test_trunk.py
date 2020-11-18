"""
Unit tests for meta/networks/trunk.py.
"""

from math import log

import numpy as np
import torch
from gym.spaces import Box, Discrete

from meta.networks.utils import init_base, init_final
from meta.networks.trunk import MultiTaskTrunkNetwork
from tests.helpers import DEFAULT_SETTINGS, get_obs_batch


SETTINGS = {
    "obs_dim": 8,
    "num_processes": 8,
    "num_tasks": 3,
    "num_shared_layers": 2,
    "num_task_layers": 1,
    "include_task_index": True,
    "device": torch.device("cpu"),
}


def test_forward() -> None:
    """
    Test forward() when each task-specific output head multiplies the shared trunk
    output by some constant factor, and the task index is included in the input.
    """

    # Set up case.
    dim = SETTINGS["obs_dim"] + SETTINGS["num_tasks"]
    observation_subspace = Box(low=-np.inf, high=np.inf, shape=(SETTINGS["obs_dim"],))
    observation_subspace.seed(DEFAULT_SETTINGS["seed"])
    hidden_size = dim

    # Construct network and set weights of each output head explicitly. We want to make
    # it so that f_i(x) = x * i + i (with broadcasted operations), where the i-th output
    # head is f_i.
    network = MultiTaskTrunkNetwork(
        input_size=dim,
        output_size=dim,
        init_base=init_base,
        init_final=init_final,
        num_tasks=SETTINGS["num_tasks"],
        num_shared_layers=SETTINGS["num_shared_layers"],
        num_task_layers=SETTINGS["num_task_layers"],
        hidden_size=hidden_size,
        device=SETTINGS["device"],
    )
    for i in range(SETTINGS["num_tasks"]):

        # Set weights.
        state_dict = network.output_heads[i].state_dict()
        state_dict["0.0.weight"] = torch.Tensor(i * np.identity(dim))
        state_dict["0.0.bias"] = torch.Tensor(i * np.ones(dim))
        network.output_heads[i].load_state_dict(state_dict)

    # Construct batch of observations concatenated with one-hot task vectors.
    obs, task_indices = get_obs_batch(
        batch_size=SETTINGS["num_processes"],
        obs_space=observation_subspace,
        num_tasks=SETTINGS["num_tasks"],
    )

    # Get output of network.
    output = network(obs, task_indices)

    # Construct expected output of network.
    trunk_output = network.trunk(obs)
    expected_output_list = []
    for i, task_index in enumerate(task_indices):
        expected_output_list.append(trunk_output[i] * task_index + task_index)
    expected_output = torch.stack(expected_output_list)

    # Test output of network.
    assert torch.allclose(output, expected_output)


def test_forward_obs_only() -> None:
    """
    Test forward() when each task-specific output head multiplies the shared trunk
    output by some constant factor, and the task index is not included in the input.
    """

    # Set up case.
    dim = SETTINGS["obs_dim"]
    observation_subspace = Box(low=-np.inf, high=np.inf, shape=(SETTINGS["obs_dim"],))
    observation_subspace.seed(DEFAULT_SETTINGS["seed"])
    hidden_size = dim
    num_shared_layers = 1
    include_task_index = False

    # Construct network and set weights of each output head explicitly. We want to make
    # it so that each layer in the shared trunk computes an identity function (plus the
    # nonlinearity), f_i(x) = x * i + i (with broadcasted operations), where the i-th
    # output head is f_i.
    network = MultiTaskTrunkNetwork(
        input_size=dim,
        output_size=dim,
        init_base=init_base,
        init_final=init_final,
        num_tasks=SETTINGS["num_tasks"],
        num_shared_layers=num_shared_layers,
        num_task_layers=SETTINGS["num_task_layers"],
        hidden_size=hidden_size,
        device=SETTINGS["device"],
    )

    # Set shared trunk weights.
    trunk_state_dict = network.trunk.state_dict()
    trunk_state_dict["0.0.weight"] = torch.Tensor(np.identity(hidden_size))
    trunk_state_dict["0.0.bias"] = torch.zeros(hidden_size)
    network.trunk.load_state_dict(trunk_state_dict)

    # Set task-specific weights.
    for i in range(SETTINGS["num_tasks"]):

        # Set weights.
        state_dict = network.output_heads[i].state_dict()
        state_dict["0.0.weight"] = torch.Tensor(i * np.identity(hidden_size))
        state_dict["0.0.bias"] = i * torch.ones(hidden_size)
        network.output_heads[i].load_state_dict(state_dict)

    # Construct batch of observations concatenated with one-hot task vectors.
    obs, task_indices = get_obs_batch(
        batch_size=SETTINGS["num_processes"],
        obs_space=observation_subspace,
        num_tasks=SETTINGS["num_tasks"],
    )
    obs_only = obs[:, :dim]

    # Get output of network.
    output = network(obs_only, task_indices)

    # Construct expected action distribution of network.
    expected_output_list = []
    for i, task_index in enumerate(task_indices):
        expected_output_list.append(torch.tanh(obs_only[i]) * task_index + task_index)
    expected_output = torch.stack(expected_output_list)

    # Test output of network.
    assert torch.allclose(output, expected_output)


def test_backward() -> None:
    """
    Test backward(). We just want to make sure that the gradient with respect to the
    i-th task loss is zero for all parameters in output head j != i, and is nonzero for
    all parameters in output head i.
    """

    # Set up case.
    dim = SETTINGS["obs_dim"] + SETTINGS["num_tasks"]
    observation_subspace = Box(low=-np.inf, high=np.inf, shape=(SETTINGS["obs_dim"],))
    observation_subspace.seed(DEFAULT_SETTINGS["seed"])
    hidden_size = dim

    # Construct network.
    network = MultiTaskTrunkNetwork(
        input_size=dim,
        output_size=dim,
        init_base=init_base,
        init_final=init_final,
        num_tasks=SETTINGS["num_tasks"],
        num_shared_layers=SETTINGS["num_shared_layers"],
        num_task_layers=SETTINGS["num_task_layers"],
        hidden_size=hidden_size,
        device=SETTINGS["device"],
    )

    # Construct batch of observations concatenated with one-hot task vectors.
    obs, task_indices = get_obs_batch(
        batch_size=SETTINGS["num_processes"],
        obs_space=observation_subspace,
        num_tasks=SETTINGS["num_tasks"],
    )

    # Make sure every task gets at least one process.
    assert set(task_indices.tolist()) == set(range(SETTINGS["num_tasks"]))

    # Get output of network.
    output = network(obs, task_indices)

    # Compute losses (we just compute the squared network output to keep it simple) and
    # test gradients.
    for i in range(SETTINGS["num_tasks"]):

        # Zero out gradients.
        network.zero_grad()

        # Compute loss over outputs from the current task.
        loss = torch.zeros(1)
        for process in range(obs.shape[0]):
            j = task_indices[process].item()
            if i == j:
                loss += torch.sum(output[process] ** 2)

        # Test gradients.
        loss.backward(retain_graph=True)
        check_gradients(network.trunk, nonzero=True)
        for j in range(SETTINGS["num_tasks"]):
            nonzero = j == i
            check_gradients(network.output_heads[j], nonzero=nonzero)


def test_check_conflicting_grads() -> None:
    """ Check whether the frequency of conflicting gradients is measured correctly. """

    # Construct network.
    dim = SETTINGS["obs_dim"] + SETTINGS["num_tasks"]
    total_steps = 5
    network = MultiTaskTrunkNetwork(
        input_size=dim,
        output_size=dim,
        init_base=init_base,
        init_final=init_final,
        num_tasks=SETTINGS["num_tasks"],
        num_shared_layers=SETTINGS["num_shared_layers"],
        num_task_layers=SETTINGS["num_task_layers"],
        hidden_size=dim,
        device=SETTINGS["device"],
        monitor_grads=True,
    )

    # Construct a sequence of task gradients. The shape of `task_grads` is
    # `(total_steps, network.num_tasks, network.num_shared_layers,
    # network.max_shared_layer_size)`
    task_grads = torch.Tensor(
        [
            [[[1.0], [2.0]], [[-1.0], [0.0]], [[0.0], [0.0]]],
            [[[0.5], [1.5]], [[1.0], [1.0]], [[-1.0], [-2.0]]],
            [[[0.0], [0.0]], [[1.0], [1.0]], [[0.0], [0.0]]],
            [[[-0.5], [3.0]], [[1.0], [1.0]], [[1.0], [3.0]]],
            [[[-2.0], [-1.0]], [[1.0], [1.0]], [[2.0], [2.0]]],
        ]
    )
    task_grads = task_grads.expand(-1, -1, -1, network.max_shared_layer_size)
    for layer in range(network.num_shared_layers):
        task_grads[:, :, layer, network.shared_layer_sizes[layer] :] = 0.0
    assert task_grads.shape == (
        total_steps,
        network.num_tasks,
        network.num_shared_layers,
        network.max_shared_layer_size,
    )

    # Construct expected conflict stats.
    expected_conflicts = torch.Tensor(
        [
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ],
            [
                [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
                [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ],
            [
                [[0.0, 0.0], [0.5, 0.0], [1.0, 1.0]],
                [[0.5, 0.0], [0.0, 0.0], [1.0, 1.0]],
                [[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
            ],
            [
                [[0.0, 0.0], [0.5, 0.0], [1.0, 1.0]],
                [[0.5, 0.0], [0.0, 0.0], [1.0, 1.0]],
                [[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
            ],
            [
                [[0.0, 0.0], [2.0 / 3.0, 0.0], [1.0, 0.5]],
                [[2.0 / 3.0, 0.0], [0.0, 0.0], [0.5, 0.5]],
                [[1.0, 0.5], [0.5, 0.5], [0.0, 0.0]],
            ],
            [
                [[0.0, 0.0], [0.75, 0.25], [1.0, 2.0 / 3.0]],
                [[0.75, 0.25], [0.0, 0.0], [1.0 / 3.0, 1.0 / 3.0]],
                [[1.0, 2.0 / 3.0], [1.0 / 3.0, 1.0 / 3.0], [0.0, 0.0]],
            ],
        ]
    )
    expected_layer_conflicts = torch.Tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.75, 0.5],
            [0.75, 0.5],
            [5.0 / 7.0, 2.0 / 7.0],
            [0.7, 0.4],
        ]
    )
    expected_sizes = torch.Tensor(
        [
            [[[0], [0], [0]], [[0], [0], [0]], [[0], [0], [0]]],
            [[[1], [1], [0]], [[1], [1], [0]], [[0], [0], [0]]],
            [[[2], [2], [1]], [[2], [2], [1]], [[1], [1], [1]]],
            [[[2], [2], [1]], [[2], [3], [1]], [[1], [1], [1]]],
            [[[3], [3], [2]], [[3], [4], [2]], [[2], [2], [2]]],
            [[[4], [4], [3]], [[4], [5], [3]], [[3], [3], [3]]],
        ]
    )
    expected_sizes = expected_sizes.expand(-1, -1, -1, network.num_shared_layers)
    assert expected_conflicts.shape == (
        total_steps + 1,
        network.num_tasks,
        network.num_tasks,
        network.num_shared_layers,
    )
    assert expected_layer_conflicts.shape == (
        total_steps + 1,
        network.num_shared_layers,
    )
    assert expected_sizes.shape == (
        total_steps + 1,
        network.num_tasks,
        network.num_tasks,
        network.num_shared_layers,
    )

    # Check computed conflict frequency against expected.
    assert torch.all(network.grad_conflict_stats.mean == expected_conflicts[0])
    assert torch.all(network.grad_conflict_stats.sample_size == expected_sizes[0])
    for step in range(total_steps):
        network.measure_conflicts_from_grads(task_grads[step])
        assert torch.all(
            network.grad_conflict_stats.mean == expected_conflicts[step + 1]
        )
        assert torch.all(
            network.layer_grad_conflicts == expected_layer_conflicts[step + 1]
        )
        assert torch.all(
            network.grad_conflict_stats.sample_size == expected_sizes[step + 1]
        )


def check_gradients(m: torch.nn.Module, nonzero: bool) -> None:
    """ Helper function to test whether gradients are nonzero. """

    for param in m.parameters():
        if nonzero:
            assert (param.grad != 0).any()
        else:
            assert param.grad is None or (param.grad == 0).all()
