"""
Unit tests for meta/networks/trunk.py.
"""

from math import log

import numpy as np
import torch
from gym.spaces import Box, Discrete

from meta.networks.initialize import init_base, init_final
from meta.networks.trunk import MultiTaskTrunkNetwork
from tests.helpers import DEFAULT_SETTINGS, one_hot_tensor


SETTINGS = {
    "obs_dim": 8,
    "num_processes": 6,
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
        state_dict["0.weight"] = torch.Tensor(i * np.identity(dim))
        state_dict["0.bias"] = torch.Tensor(i * np.ones(dim))
        network.output_heads[i].load_state_dict(state_dict)

    # Construct batch of observations concatenated with one-hot task vectors.
    obs_list = []
    for _ in range(SETTINGS["num_processes"]):
        ob = torch.Tensor(observation_subspace.sample())
        task_vector = one_hot_tensor(SETTINGS["num_tasks"])
        obs_list.append(torch.cat([ob, task_vector]))
    obs = torch.stack(obs_list)
    nonzero_pos = obs[:, SETTINGS["obs_dim"] :].nonzero()
    assert nonzero_pos[:, 0].tolist() == list(range(SETTINGS["num_processes"]))
    task_indices = nonzero_pos[:, 1]

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
    trunk_state_dict["0.weight"] = torch.Tensor(np.identity(hidden_size))
    trunk_state_dict["0.bias"] = torch.zeros(hidden_size)
    network.trunk.load_state_dict(trunk_state_dict)

    # Set task-specific weights.
    for i in range(SETTINGS["num_tasks"]):

        # Set weights.
        state_dict = network.output_heads[i].state_dict()
        state_dict["0.weight"] = torch.Tensor(i * np.identity(hidden_size))
        state_dict["0.bias"] = i * torch.ones(hidden_size)
        network.output_heads[i].load_state_dict(state_dict)

    # Construct batch of observations concatenated with one-hot task vectors.
    obs_list = []
    for _ in range(SETTINGS["num_processes"]):
        ob = torch.Tensor(observation_subspace.sample())
        task_vector = one_hot_tensor(SETTINGS["num_tasks"])
        obs_list.append(torch.cat([ob, task_vector]))
    obs = torch.stack(obs_list)
    nonzero_pos = obs[:, SETTINGS["obs_dim"] :].nonzero()
    assert nonzero_pos[:, 0].tolist() == list(range(SETTINGS["num_processes"]))
    task_indices = nonzero_pos[:, 1]
    obs_only = obs[:, : SETTINGS["obs_dim"]]

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
    obs_list = []
    for _ in range(SETTINGS["num_processes"]):
        ob = torch.Tensor(observation_subspace.sample())
        task_vector = one_hot_tensor(SETTINGS["num_tasks"])
        obs_list.append(torch.cat([ob, task_vector]))
    obs = torch.stack(obs_list)
    nonzero_pos = obs[:, SETTINGS["obs_dim"] :].nonzero()
    assert nonzero_pos[:, 0].tolist() == list(range(SETTINGS["num_processes"]))
    task_indices = nonzero_pos[:, 1]

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


def check_gradients(m: torch.nn.Module, nonzero: bool) -> None:
    """ Helper function to test whether gradients are nonzero. """

    for param in m.parameters():
        if nonzero:
            assert (param.grad != 0).any()
        else:
            assert param.grad is None or (param.grad == 0).all()
