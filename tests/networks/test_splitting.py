"""
Unit tests for meta/networks/splitting.py.
"""

import numpy as np
import torch
from gym.spaces import Box

from meta.networks.initialize import init_base
from meta.networks.splitting import SplittingMLPNetwork
from tests.helpers import DEFAULT_SETTINGS, one_hot_tensor


SETTINGS = {
    "obs_dim": 8,
    "num_processes": 6,
    "num_tasks": 3,
    "num_layers": 3,
    "include_task_index": True,
    "device": torch.device("cpu"),
}


def test_forward() -> None:
    """
    Test forward().
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

    # Construct batch of observations concatenated with one-hot task vectors.
    obs_list = []
    for i in range(SETTINGS["num_processes"]):
        # ob = torch.Tensor(observation_subspace.sample())
        ob = torch.Tensor([i] * SETTINGS["obs_dim"])
        task_vector = one_hot_tensor(SETTINGS["num_tasks"])
        obs_list.append(torch.cat([ob, task_vector]))
    obs = torch.stack(obs_list)
    nonzero_pos = obs[:, SETTINGS["obs_dim"] :].nonzero()
    assert nonzero_pos[:, 0].tolist() == list(range(SETTINGS["num_processes"]))
    task_indices = nonzero_pos[:, 1]

    # Get output of network.
    output = network(obs, task_indices)

    assert False
