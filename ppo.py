from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from gym.spaces import Space, Box, Discrete

from storage import RolloutStorage


class PPOPolicy:
    """ A policy class for PPO. """

    def __init__(self, observation_space: Space, action_space: Space):
        """
        init function for PPOPolicy.

        Arguments
        ---------
        observation_space: Space
            Environment's observation space.
        action_space: Space
            Environment's action space.
        """

        self.observation_space = observation_space
        self.action_space = action_space

    def act(self, obs: Union[np.ndarray, int, float]):
        """
        Sample action from policy.

        Arguments
        ---------
        obs: np.ndarray or int or float
            Observation to sample action from.

        Returns
        -------
        value_pred: torch.Tensor
            Value prediction from critic portion of policy.
        action: torch.Tensor
            Action sampled from distribution defined by policy output.
        action_log_prob: torch.Tensor
            Log probability of sampled action.
        """

        # Pass through network to get value prediction and action probabilities.
        # HARDCODE
        # <<<<<<<<<
        # value_pred, action_probs = self.policy_network(obs)
        # =========
        value_pred = torch.Tensor([0.0])
        action_probs = {
            "loc": torch.zeros(self.action_space.shape),
            "scale": torch.ones(self.action_space.shape),
        }
        action_probs["loc"] = torch.Tensor([0.2, 0.2, 0.1, 0.5])
        # >>>>>>>>>

        # Create action distribution object from probabilities.
        if isinstance(self.action_space, Discrete):
            action_dist = Categorical(**action_probs)
        elif isinstance(self.action_space, Box):
            action_dist = Normal(**action_probs)
        else:
            raise ValueError("Action space '%r' unsupported." % type(self.action_space))

        # Sample action and compute log probability of action.
        action = action_dist.sample()
        element_log_probs = action_dist.log_prob(action)
        action_log_prob = element_log_probs.sum(-1)

        return value_pred, action, action_log_prob

    def update(self, rollouts: RolloutStorage):
        """
        Train policy with PPO from ``rollouts``.

        Arguments
        ---------
        rollouts: RolloutStorage
            Storage container holding rollout information to train from.
        """

        pass


class PolicyNetwork(nn.Module):
    """ MLP network parameterizing the policy. """

    def __init__(self, num_inputs: int, hidden_size: int, num_hidden_layers: int):
        """ init function for PolicyNetwork. """
        pass

    def forward(self, inputs):
        """ Forward pass definition for PolicyNetwork. """
        pass
