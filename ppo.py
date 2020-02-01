import torch
from gym.spaces import Space

from storage import RolloutStorage


class PPOPolicy:
    """ A policy class for PPO. """

    def __init__(self, observation_space: Space, action_space: Space):
        """ init function for PPOPolicy. """

        self.observation_space = observation_space
        self.action_space = action_space

    def act(self, obs):
        """ Sample action from policy. """

        action = self.action_space.sample()
        return torch.Tensor([0.0]), action, torch.Tensor([0.0])

    def update(self, rollouts: RolloutStorage):
        """ Train policy with PPO from ``rollouts``. """

        pass
