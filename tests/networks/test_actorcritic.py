"""
Unit tests for meta/networks/actorcritic.py.
"""

from typing import Dict, Any

import numpy as np
import torch

from meta.networks.actorcritic import ActorCriticNetwork
from meta.train.env import get_env
from meta.utils.utils import get_space_size
from tests.helpers import DEFAULT_SETTINGS, one_hot_tensor


TOL = 1e-5


def test_actorcritic_exclude_task():
    """
    Test actorcritic properly excludes task index from trunk network input when
    `include_task_index` is False. To test this, we manually set the network weights and
    check the output.
    """

    # Set up case.
    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "MT10"
    settings["architecture_config"] = {
        "type": "trunk",
        "recurrent": False,
        "recurrent_hidden_size": None,
        "include_task_index": False,
        "num_tasks": 10,
        "actor_config": {
            "num_shared_layers": 1,
            "num_task_layers": 1,
            "hidden_size": 39,
        },
        "critic_config": {
            "num_shared_layers": 1,
            "num_task_layers": 1,
            "hidden_size": 39,
        },
    }
    settings["num_processes"] = 8
    settings["env_kwargs"] = {"save_memory": False, "uniform_tasks": False}

    # Call template.
    actorcritic_exclude_task_template(settings)


def test_actorcritic_exclude_task_recurrent():
    """
    Test actorcritic properly excludes task index from trunk network input when
    `include_task_index` is False, with a recurrent network. To test this, we manually
    set the network weights and check the output.
    """

    # Set up case.
    settings = dict(DEFAULT_SETTINGS)
    settings["env_name"] = "MT10"
    settings["architecture_config"] = {
        "type": "trunk",
        "recurrent": False,
        "recurrent_hidden_size": None,
        "include_task_index": False,
        "num_tasks": 10,
        "actor_config": {
            "num_shared_layers": 1,
            "num_task_layers": 1,
            "hidden_size": 39,
        },
        "critic_config": {
            "num_shared_layers": 1,
            "num_task_layers": 1,
            "hidden_size": 39,
        },
    }
    settings["num_processes"] = 8
    settings["env_kwargs"] = {"save_memory": False, "uniform_tasks": False}

    # Call template.
    actorcritic_exclude_task_template(settings)


def actorcritic_exclude_task_template(settings: Dict[str, any]):
    """ Template for the actorcritic_exclude_task tests. """

    num_tasks = settings["architecture_config"]["num_tasks"]

    # Create environment.
    env = get_env(
        settings["env_name"],
        settings["num_processes"],
        allow_early_resets=True,
        **settings["env_kwargs"]
    )
    obs_size = get_space_size(env.observation_space)
    action_size = get_space_size(env.action_space)
    hidden_size = obs_size - num_tasks

    # Create network.
    network = ActorCriticNetwork(
        observation_space=env.observation_space,
        action_space=env.action_space,
        num_processes=settings["num_processes"],
        rollout_length=settings["rollout_length"],
        architecture_config=dict(settings["architecture_config"]),
        device=settings["device"],
    )

    # Set network weights.
    for m_name in ["actor", "critic"]:
        m = getattr(network, m_name)
        trunk_state_dict = m.trunk.state_dict()
        trunk_state_dict["0.0.weight"] = torch.Tensor(np.identity(hidden_size))
        trunk_state_dict["0.0.bias"] = torch.zeros(hidden_size)
        m.trunk.load_state_dict(trunk_state_dict)

        for i in range(num_tasks):
            state_dict = m.output_heads[i].state_dict()
            if m_name == "actor":
                state_dict["0.0.weight"] = torch.Tensor(
                    i * np.ones((action_size, hidden_size))
                )
                state_dict["0.0.bias"] = i * torch.ones(action_size)
            elif m_name == "critic":
                state_dict["0.0.weight"] = torch.Tensor(
                    i * np.ones(hidden_size)
                ).unsqueeze(0)
                state_dict["0.0.bias"] = i * torch.ones(1)
            else:
                raise NotImplementedError

            m.output_heads[i].load_state_dict(state_dict)

    if settings["architecture_config"]["recurrent"]:
        state_dict = network.recurrent_block.state_dict()
        for key in list(state_dict.keys()):
            state_dict[key] = torch.zeros(*state_dict[key].shape)
        network.recurrent_block.load_state_dict(state_dict)

    # Test size of network layers.
    if settings["architecture_config"]["recurrent"]:
        assert network.recurrent_block.input_size == obs_size - num_tasks
        assert network.recurrent_block.hidden_size == hidden_size
    else:
        assert not hasattr(network, "recurrent_block")
    for m in [network.actor, network.critic]:
        if settings["architecture_config"]["recurrent"]:
            assert m.trunk[0][0].in_features == hidden_size
        else:
            assert m.trunk[0][0].in_features == obs_size - num_tasks
        for i in range(num_tasks):
            assert m.output_heads[i][0][0].in_features == hidden_size
    for i in range(num_tasks):
        assert network.actor.output_heads[i][0][0].out_features == action_size
        assert network.critic.output_heads[i][0][0].out_features == 1

    # Construct batch of observations concatenated with one-hot task vectors.
    obs_list = []
    for _ in range(settings["num_processes"]):
        ob = torch.Tensor(env.observation_space.sample())
        ob = ob[: obs_size - num_tasks]
        task_vector = one_hot_tensor(num_tasks)
        obs_list.append(torch.cat([ob, task_vector]))
    obs = torch.stack(obs_list)
    task_index_pos = obs_size - num_tasks
    nonzero_pos = obs[:, task_index_pos:].nonzero()
    assert nonzero_pos[:, 0].tolist() == list(range(settings["num_processes"]))
    task_indices = nonzero_pos[:, 1]

    # Test output of network layers.
    hidden_state = None
    done = None
    if settings["architecture_config"]["recurrent"]:
        hidden_state = 2 * obs[:, :task_index_pos]
        done = torch.zeros(settings["num_processes"], 1)
    value_pred, action_dist, _ = network(obs, hidden_state=hidden_state, done=done)

    # Compute expected distribution mean and value prediction.
    expected_mean = torch.zeros(settings["num_processes"], action_size)
    expected_value_pred = torch.zeros(settings["num_processes"], 1)
    for i, task in enumerate(task_indices):
        expected_mean[i] = torch.Tensor(
            [torch.sum(task * torch.tanh(obs[i, : obs_size - num_tasks])) + task]
            * action_size
        )
        expected_value_pred[i] = (
            torch.sum(task * torch.tanh(obs[i, : obs_size - num_tasks])) + task
        )

    # Test output.
    print(action_dist.mean - expected_mean)
    assert torch.allclose(action_dist.mean, expected_mean, atol=TOL)
    assert torch.allclose(value_pred, expected_value_pred, atol=TOL)
