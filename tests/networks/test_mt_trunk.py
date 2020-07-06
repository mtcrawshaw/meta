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
    "num_processes": 4,
    "rollout_length": 8,
    "num_shared_layers": 2,
    "num_task_layers": 1,
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
    observation_space = Box(low=-np.inf, high=np.inf, shape=(SETTINGS["obs_dim"],))
    observation_space.seed(DEFAULT_SETTINGS["seed"])
    action_space = Discrete(dim)
    hidden_size = dim
    recurrent = False
    device = torch.device("cpu")

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
        recurrent=recurrent,
        device=device,
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
        ob = torch.Tensor(observation_space.sample())
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
    observation_space = Box(low=-np.inf, high=np.inf, shape=(SETTINGS["obs_dim"],))
    observation_space.seed(DEFAULT_SETTINGS["seed"])
    action_space = Box(low=-np.inf, high=np.inf, shape=(dim,))
    hidden_size = dim
    recurrent = False
    device = torch.device("cpu")

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
        recurrent=recurrent,
        device=device,
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
        ob = torch.Tensor(observation_space.sample())
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
