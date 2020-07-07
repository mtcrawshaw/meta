"""
Unit tests for meta/networks/mt_trunk.py.
"""

from math import log
import random

import numpy as np
import torch
from gym.spaces import Box, Discrete

from meta.networks.mt_trunk import MultiTaskTrunkNetwork
from tests.helpers import DEFAULT_SETTINGS


SETTINGS = {
    "obs_dim": 8,
    "num_tasks": 3,
    "num_processes": 6,
    "rollout_length": 8,
    "num_shared_layers": 2,
    "num_task_layers": 1,
    "include_task_index": True,
    "recurrent": False,
    "device": torch.device("cpu"),
}


def one_hot_tensor(n: int) -> torch.Tensor:
    """ Sample a one hot vector of length n, return as a torch Tensor. """

    one_hot = torch.zeros(n)
    k = random.randrange(n)
    one_hot[k] = 1.0
    return one_hot


def test_forward_discrete() -> None:
    """
    Test forward() when each task-specific output head multiplies the shared trunk
    output by some constant factor, when the action space is Discrete.
    """

    # Set up case.
    dim = SETTINGS["obs_dim"] + SETTINGS["num_tasks"]
    observation_subspace = Box(low=-np.inf, high=np.inf, shape=(SETTINGS["obs_dim"],))
    observation_subspace.seed(DEFAULT_SETTINGS["seed"])
    observation_space = Box(low=-np.inf, high=np.inf, shape=(dim,))
    action_space = Discrete(dim)
    hidden_size = dim

    # Construct network and set weights of each output head explicitly. We want to make
    # it so that f_i(x) = x * i + i (with broadcasted operations), where the i-th actor
    # head is f_i. Similarly for the critic head, we want to make it so that g_i(x) =
    # sum(x * i) + i, where the i-th critic head is g_i.
    network = MultiTaskTrunkNetwork(
        observation_space=observation_space,
        action_space=action_space,
        num_processes=SETTINGS["num_processes"],
        rollout_length=SETTINGS["rollout_length"],
        num_tasks=SETTINGS["num_tasks"],
        num_shared_layers=SETTINGS["num_shared_layers"],
        num_task_layers=SETTINGS["num_task_layers"],
        hidden_size=hidden_size,
        recurrent=SETTINGS["recurrent"],
        device=SETTINGS["device"],
        include_task_index=SETTINGS["include_task_index"],
    )
    for i in range(SETTINGS["num_tasks"]):

        # Set actor weights.
        actor_state_dict = network.actor_output_heads[i].state_dict()
        actor_state_dict["0.weight"] = torch.Tensor(i * np.identity(dim))
        actor_state_dict["0.bias"] = torch.Tensor(i * np.ones(dim))
        network.actor_output_heads[i].load_state_dict(actor_state_dict)

        # Set critic weights.
        critic_state_dict = network.critic_output_heads[i].state_dict()
        critic_state_dict["0.weight"] = torch.Tensor(i * np.ones(dim)).unsqueeze(0)
        critic_state_dict["0.bias"] = torch.Tensor([i])
        network.critic_output_heads[i].load_state_dict(critic_state_dict)

    # Construct batch of observations concatenated with one-hot task vectors.
    obs_list = []
    for _ in range(SETTINGS["num_processes"]):
        ob = torch.Tensor(observation_subspace.sample())
        task_vector = one_hot_tensor(SETTINGS["num_tasks"])
        obs_list.append(torch.cat([ob, task_vector]))
    obs = torch.stack(obs_list)
    nonzero_pos = obs[:, SETTINGS["obs_dim"] :].nonzero()
    assert nonzero_pos[:, 0].tolist() == list(range(SETTINGS["num_processes"]))
    task_indices = nonzero_pos[:, 1].tolist()

    # Get output of network.
    value_pred, action_dist, _ = network(obs, hidden_state=None, done=None)

    # Construct expected action distribution of network.
    actor_trunk_output = network.actor_trunk(obs)
    expected_logits_list = []
    for i, task_index in enumerate(task_indices):
        expected_logits_list.append(actor_trunk_output[i] * task_index + task_index)
    expected_logits = torch.stack(expected_logits_list)
    expected_probs = torch.softmax(expected_logits, dim=1)

    # Construct expected value prediction of network.
    critic_trunk_output = network.critic_trunk(obs)
    expected_value_list = []
    for i, task_index in enumerate(task_indices):
        expected_value_list.append(
            torch.sum(critic_trunk_output[i] * task_index) + task_index
        )
    expected_value = torch.stack(expected_value_list).unsqueeze(-1)

    # Test output of network.
    assert torch.allclose(action_dist.probs, expected_probs)
    assert torch.allclose(value_pred, expected_value)


def test_forward_box() -> None:
    """
    Test forward() when each task-specific output head multiplies the shared trunk
    output by some constant factor, when the action space is Box.
    """

    # Set up case.
    dim = SETTINGS["obs_dim"] + SETTINGS["num_tasks"]
    observation_subspace = Box(low=-np.inf, high=np.inf, shape=(SETTINGS["obs_dim"],))
    observation_subspace.seed(DEFAULT_SETTINGS["seed"])
    observation_space = Box(low=-np.inf, high=np.inf, shape=(dim,))
    action_space = Box(low=-np.inf, high=np.inf, shape=(dim,))
    hidden_size = dim

    # Construct network and set weights of each output head explicitly. We want to make
    # it so that f_i(x) = x * i + i (with broadcasted operations), where the i-th actor
    # head is f_i. Similarly for the critic head, we want to make it so that g_i(x) =
    # sum(x * i) + i, where the i-th critic head is g_i. We also set the logstd for the
    # action distribution of the i-th actor head to log(i + 1).
    network = MultiTaskTrunkNetwork(
        observation_space=observation_space,
        action_space=action_space,
        num_processes=SETTINGS["num_processes"],
        rollout_length=SETTINGS["rollout_length"],
        num_tasks=SETTINGS["num_tasks"],
        num_shared_layers=SETTINGS["num_shared_layers"],
        num_task_layers=SETTINGS["num_task_layers"],
        hidden_size=hidden_size,
        recurrent=SETTINGS["recurrent"],
        device=SETTINGS["device"],
        include_task_index=SETTINGS["include_task_index"],
    )
    for i in range(SETTINGS["num_tasks"]):

        # Set actor weights.
        actor_state_dict = network.actor_output_heads[i].state_dict()
        actor_state_dict["0.weight"] = torch.Tensor(i * np.identity(dim))
        actor_state_dict["0.bias"] = torch.Tensor(i * np.ones(dim))
        network.actor_output_heads[i].load_state_dict(actor_state_dict)

        # Set critic weights.
        critic_state_dict = network.critic_output_heads[i].state_dict()
        critic_state_dict["0.weight"] = torch.Tensor(i * np.ones(dim)).unsqueeze(0)
        critic_state_dict["0.bias"] = torch.Tensor([i])
        network.critic_output_heads[i].load_state_dict(critic_state_dict)

        # Set logstd for each actor head.
        logstd_state_dict = network.output_logstd[i].state_dict()
        logstd_state_dict["_bias"] = torch.Tensor([log(i + 1)] * dim)
        network.output_logstd[i].load_state_dict(logstd_state_dict)

    # Construct batch of observations concatenated with one-hot task vectors.
    obs_list = []
    for _ in range(SETTINGS["num_processes"]):
        ob = torch.Tensor(observation_subspace.sample())
        task_vector = one_hot_tensor(SETTINGS["num_tasks"])
        obs_list.append(torch.cat([ob, task_vector]))
    obs = torch.stack(obs_list)
    nonzero_pos = obs[:, SETTINGS["obs_dim"] :].nonzero()
    assert nonzero_pos[:, 0].tolist() == list(range(SETTINGS["num_processes"]))
    task_indices = nonzero_pos[:, 1].tolist()

    # Get output of network.
    value_pred, action_dist, _ = network(obs, hidden_state=None, done=None)

    # Construct expected action distribution of network.
    actor_trunk_output = network.actor_trunk(obs)
    expected_mean_list = []
    expected_stddev_list = []
    for i, task_index in enumerate(task_indices):
        expected_mean_list.append(actor_trunk_output[i] * task_index + task_index)
        expected_stddev_list.append(torch.Tensor([task_index + 1] * dim))
    expected_mean = torch.stack(expected_mean_list)
    expected_stddev = torch.stack(expected_stddev_list)

    # Construct expected value prediction of network.
    critic_trunk_output = network.critic_trunk(obs)
    expected_value_list = []
    for i, task_index in enumerate(task_indices):
        expected_value_list.append(
            torch.sum(critic_trunk_output[i] * task_index) + task_index
        )
    expected_value = torch.stack(expected_value_list).unsqueeze(-1)

    # Test output of network.
    assert torch.allclose(action_dist.mean, expected_mean)
    assert torch.allclose(action_dist.stddev, expected_stddev)
    assert torch.allclose(value_pred, expected_value)


def test_forward_obs_only() -> None:
    """
    Test forward() when each task-specific output head multiplies the shared trunk
    output by some constant factor, omitting the one-hot task vector from the network
    input.
    """

    # Set up case.
    dim = SETTINGS["obs_dim"] + SETTINGS["num_tasks"]
    observation_subspace = Box(low=-np.inf, high=np.inf, shape=(SETTINGS["obs_dim"],))
    observation_subspace.seed(DEFAULT_SETTINGS["seed"])
    observation_space = Box(low=-np.inf, high=np.inf, shape=(dim,))
    action_space = Box(low=-np.inf, high=np.inf, shape=(SETTINGS["obs_dim"],))
    hidden_size = SETTINGS["obs_dim"]
    num_shared_layers = 1
    include_task_index = False

    # Construct network and set weights of each output head explicitly. We want to make
    # it so that each layer in the shared trunk computes an identity function (plus the
    # nonlinearity), f_i(x) = x * i + i (with broadcasted operations), where the i-th
    # actor head is f_i. Similarly for the critic head, we want to make it so that
    # g_i(x) = sum(x * i) + i, where the i-th critic head is g_i. We also set the logstd
    # for the action distribution of the i-th actor head to log(i + 1).
    network = MultiTaskTrunkNetwork(
        observation_space=observation_space,
        action_space=action_space,
        num_processes=SETTINGS["num_processes"],
        rollout_length=SETTINGS["rollout_length"],
        num_tasks=SETTINGS["num_tasks"],
        num_shared_layers=num_shared_layers,
        num_task_layers=SETTINGS["num_task_layers"],
        hidden_size=hidden_size,
        recurrent=SETTINGS["recurrent"],
        device=SETTINGS["device"],
        include_task_index=include_task_index,
    )

    # Set shared trunk weights.
    actor_trunk_state_dict = network.actor_trunk.state_dict()
    actor_trunk_state_dict["0.weight"] = torch.Tensor(np.identity(hidden_size))
    actor_trunk_state_dict["0.bias"] = torch.zeros(hidden_size)
    network.actor_trunk.load_state_dict(actor_trunk_state_dict)

    critic_trunk_state_dict = network.critic_trunk.state_dict()
    critic_trunk_state_dict["0.weight"] = torch.Tensor(np.identity(hidden_size))
    critic_trunk_state_dict["0.bias"] = torch.zeros(hidden_size)
    network.critic_trunk.load_state_dict(critic_trunk_state_dict)

    # Set task-specific weights.
    for i in range(SETTINGS["num_tasks"]):

        # Set actor weights.
        actor_state_dict = network.actor_output_heads[i].state_dict()
        actor_state_dict["0.weight"] = torch.Tensor(i * np.identity(hidden_size))
        actor_state_dict["0.bias"] = i * torch.ones(hidden_size)
        network.actor_output_heads[i].load_state_dict(actor_state_dict)

        # Set critic weights.
        critic_state_dict = network.critic_output_heads[i].state_dict()
        critic_state_dict["0.weight"] = (i * torch.ones(hidden_size)).unsqueeze(0)
        critic_state_dict["0.bias"] = torch.Tensor([i])
        network.critic_output_heads[i].load_state_dict(critic_state_dict)

        # Set logstd for each actor head.
        logstd_state_dict = network.output_logstd[i].state_dict()
        logstd_state_dict["_bias"] = torch.Tensor([log(i + 1)] * hidden_size)
        network.output_logstd[i].load_state_dict(logstd_state_dict)

    # Construct batch of observations concatenated with one-hot task vectors.
    obs_list = []
    for _ in range(SETTINGS["num_processes"]):
        ob = torch.Tensor(observation_subspace.sample())
        task_vector = one_hot_tensor(SETTINGS["num_tasks"])
        obs_list.append(torch.cat([ob, task_vector]))
    obs = torch.stack(obs_list)
    nonzero_pos = obs[:, SETTINGS["obs_dim"] :].nonzero()
    assert nonzero_pos[:, 0].tolist() == list(range(SETTINGS["num_processes"]))
    task_indices = nonzero_pos[:, 1].tolist()
    obs_only = obs[:, : SETTINGS["obs_dim"]]

    # Get output of network.
    value_pred, action_dist, _ = network(obs, hidden_state=None, done=None)

    # Construct expected action distribution of network.
    expected_mean_list = []
    expected_stddev_list = []
    for i, task_index in enumerate(task_indices):
        expected_mean_list.append(torch.tanh(obs_only[i]) * task_index + task_index)
        expected_stddev_list.append(torch.Tensor([task_index + 1] * hidden_size))
    expected_mean = torch.stack(expected_mean_list)
    expected_stddev = torch.stack(expected_stddev_list)

    # Construct expected value prediction of network.
    expected_value_list = []
    for i, task_index in enumerate(task_indices):
        expected_value_list.append(
            torch.sum(torch.tanh(obs_only[i]) * task_index) + task_index
        )
    expected_value = torch.stack(expected_value_list).unsqueeze(-1)

    # Test output of network.
    assert torch.allclose(action_dist.mean, expected_mean)
    assert torch.allclose(action_dist.stddev, expected_stddev)
    assert torch.allclose(value_pred, expected_value)


def test_backward_1() -> None:
    """
    Test backward() when the action space is Discrete and network is feedforward. We
    just want to make sure that the gradient with respect to the i-th task loss is zero
    for all parameters in output head j != i, and is nonzero for all parameters in
    output head i.
    """

    # Set up case.
    dim = SETTINGS["obs_dim"] + SETTINGS["num_tasks"]
    observation_subspace = Box(low=-np.inf, high=np.inf, shape=(SETTINGS["obs_dim"],))
    observation_subspace.seed(DEFAULT_SETTINGS["seed"])
    observation_space = Box(low=-np.inf, high=np.inf, shape=(dim,))
    action_space = Discrete(dim)
    hidden_size = dim

    # Construct network.
    network = MultiTaskTrunkNetwork(
        observation_space=observation_space,
        action_space=action_space,
        num_processes=SETTINGS["num_processes"],
        rollout_length=SETTINGS["rollout_length"],
        num_tasks=SETTINGS["num_tasks"],
        num_shared_layers=SETTINGS["num_shared_layers"],
        num_task_layers=SETTINGS["num_task_layers"],
        hidden_size=hidden_size,
        recurrent=SETTINGS["recurrent"],
        device=SETTINGS["device"],
        include_task_index=SETTINGS["include_task_index"],
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
    task_indices = nonzero_pos[:, 1].tolist()

    # Make sure every task gets at least one process.
    assert set(task_indices) == set(range(SETTINGS["num_tasks"]))

    # Get output of network.
    value_pred, action_dist, _ = network(obs, hidden_state=None, done=None)

    # Compute losses (we just compute the squared network output to keep it simple) and
    # test gradients.
    for i in range(SETTINGS["num_tasks"]):

        # Zero out gradients.
        network.zero_grad()

        # Compute loss over outputs from the current task.
        loss = torch.zeros(1)
        for process in range(obs.shape[0]):
            j = task_indices[process]
            if i == j:
                logits = action_dist.logits[process]
                value = value_pred[process]
                loss += torch.sum(logits ** 2)
                loss += value ** 2

        # Test gradients.
        loss.backward(retain_graph=True)
        check_gradients(network.actor_trunk, nonzero=True)
        check_gradients(network.critic_trunk, nonzero=True)
        for j in range(SETTINGS["num_tasks"]):
            nonzero = j == i
            check_gradients(network.actor_output_heads[j], nonzero=nonzero)
            check_gradients(network.critic_output_heads[j], nonzero=nonzero)


def test_backward_2() -> None:
    """
    Test backward() when the action space is Box and network is recurrent. We just want
    to make sure that the gradient with respect to the i-th task loss is zero for all
    parameters in output head j != i, and is nonzero for all parameters in output head
    i.
    """

    # Set up case.
    dim = SETTINGS["obs_dim"] + SETTINGS["num_tasks"]
    observation_subspace = Box(low=-np.inf, high=np.inf, shape=(SETTINGS["obs_dim"],))
    observation_subspace.seed(DEFAULT_SETTINGS["seed"])
    observation_space = Box(low=-np.inf, high=np.inf, shape=(dim,))
    action_space = Box(low=-np.inf, high=np.inf, shape=(dim,))
    hidden_size = dim
    recurrent = True

    # Construct network.
    network = MultiTaskTrunkNetwork(
        observation_space=observation_space,
        action_space=action_space,
        num_processes=SETTINGS["num_processes"],
        rollout_length=SETTINGS["rollout_length"],
        num_tasks=SETTINGS["num_tasks"],
        num_shared_layers=SETTINGS["num_shared_layers"],
        num_task_layers=SETTINGS["num_task_layers"],
        hidden_size=hidden_size,
        recurrent=recurrent,
        device=SETTINGS["device"],
        include_task_index=SETTINGS["include_task_index"],
    )

    # Sample a task for each process.
    task_vector_list = []
    for _ in range(SETTINGS["num_processes"]):
        task_vector_list.append(one_hot_tensor(SETTINGS["num_tasks"]))
    task_vectors = torch.stack(task_vector_list)
    nonzero_pos = task_vectors.nonzero()
    assert nonzero_pos[:, 0].tolist() == list(range(SETTINGS["num_processes"]))
    task_indices = nonzero_pos[:, 1].tolist()

    # Make sure every task gets at least one process.
    assert set(task_indices) == set(range(SETTINGS["num_tasks"]))

    # Run multiple timesteps of forward passes.
    hidden_state = torch.zeros(SETTINGS["num_processes"], hidden_size)
    done = torch.zeros(SETTINGS["num_processes"], 1)
    mean_list = []
    std_list = []
    value_pred_list = []
    for _ in range(SETTINGS["rollout_length"]):

        # Construct batch of observations.
        obs_list = []
        for i in range(SETTINGS["num_processes"]):
            ob = torch.Tensor(observation_subspace.sample())
            obs_list.append(torch.cat([ob, task_vectors[i]]))
        obs = torch.stack(obs_list)

        # Get output of network.
        value_pred, action_dist, hidden_state = network(
            obs, hidden_state=hidden_state, done=done
        )
        mean_list.append(action_dist.mean)
        std_list.append(action_dist.stddev)
        value_pred_list.append(value_pred)

    mean = torch.stack(mean_list)
    std = torch.stack(std_list)
    value_pred = torch.stack(value_pred_list)

    # Compute losses (we just compute the squared network output to keep it simple) and
    # test gradients.
    for i in range(SETTINGS["num_tasks"]):

        # Zero out gradients.
        network.zero_grad()

        # Compute loss over outputs from the current task.
        loss = torch.zeros(1)
        for process in range(obs.shape[0]):
            j = task_indices[process]
            if i == j:
                current_mean = mean[:, process]
                current_std = std[:, process]
                current_value = value_pred[:, process]
                loss += torch.sum(current_mean ** 2)
                loss += torch.sum(current_std ** 2)
                loss += torch.sum(current_value ** 2)

        # Test gradients.
        loss.backward(retain_graph=True)
        check_gradients(network.actor_trunk, nonzero=True)
        check_gradients(network.critic_trunk, nonzero=True)
        for j in range(SETTINGS["num_tasks"]):
            nonzero = j == i
            check_gradients(network.actor_output_heads[j], nonzero=nonzero)
            check_gradients(network.critic_output_heads[j], nonzero=nonzero)


# Helper function to test gradients.
def check_gradients(m: torch.nn.Module, nonzero: bool) -> None:
    correct = True
    for param in m.parameters():
        if nonzero:
            assert (param.grad != 0).any()
        else:
            assert param.grad is None or (param.grad == 0).all()
