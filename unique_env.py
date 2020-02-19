from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from gym.spaces import Box, Discrete


class UniqueEnv:
    """ Environment for testing. Each step returns a unique observation and reward. """

    def __init__(self) -> None:
        """ Init function for DummyEnv. """

        self.observation_space = Box(low=0.0, high=np.inf, shape=(1,))
        self.action_space = Discrete(2)
        self.timestep = 1

    def reset(self) -> float:
        """ Reset environment to initial state. """

        self.timestep = 1
        return float(self.timestep)

    def step(self, action: float) -> Tuple[float, float, bool, dict]:
        """
        Step function for environment. Returns an observation, a reward,
        whether or not the environment is done, and an info dictionary, as is
        the standard for OpenAI gym environments.
        """

        reward = float(self.timestep)
        self.timestep += 1
        obs = float(self.timestep)
        done = False
        info = {}

        return obs, reward, done, info


class UniquePolicyNetwork(nn.Module):
    """ Policy for testing. Returns a unique action distribution for each observation. """

    def __init__(self):
        """ Init function for UniquePolicy Network. """

        super().__init__()

    def forward(self, obs: torch.Tensor) -> Tuple[float, Dict[str, torch.Tensor]]:
        """ Forward pass definition for UniquePolicyNetwork. """

        value_pred = torch.zeros(obs.shape)
        value_pred.copy_(obs)
        probs = torch.cat(
            [
                1 / (obs + 1),
                1 - 1 / (obs + 1),
            ]
        )
        action_probs = {"probs": probs}

        return value_pred, action_probs
