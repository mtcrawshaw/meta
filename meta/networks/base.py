"""
Definition of BaseNetwork, which serves as an abstract class for an actor critic network
that is extended by other modules defined in this folder.
"""

from typing import Tuple, List

import torch
import torch.nn as nn
from torch.distributions import Distribution
import numpy as np
from gym.spaces import Space

from meta.utils.utils import get_space_size, get_space_shape


class BaseNetwork(nn.Module):
    """ Module used as an abstract class for an actor critic network. """

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        num_processes: int,
        rollout_length: int,
        hidden_size: int = 64,
        recurrent: bool = False,
        device: torch.device = None,
    ) -> None:

        super(BaseNetwork, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_processes = num_processes
        self.rollout_length = rollout_length
        self.hidden_size = hidden_size
        self.recurrent = recurrent

        # Calculate the input/output size.
        self.input_size = get_space_size(observation_space)
        self.output_size = get_space_size(action_space)

        # Set device.
        self.device = device if device is not None else torch.device("cpu")

        # Generate network layers.
        self.initialize_network()

        # Move model to device.
        self.to(device)

    def initialize_network(self) -> None:
        """ Initialize layers of network. Must be defined in child classes. """

        raise NotImplementedError

    def forward(
        self, obs: torch.Tensor, hidden_state: torch.Tensor, done: torch.Tensor,
    ) -> Tuple[torch.Tensor, Distribution, torch.Tensor]:
        """
        Forward pass definition for BaseNetwork. Must be defined in child classes.

        Arguments
        ---------
        obs : torch.Tensor
            Observation to be used as input to policy network. If the observation space
            is discrete, this function expects ``obs`` to be a one-hot vector.
        hidden_state : torch.Tensor
            Hidden state to use for recurrent layer, if necessary.
        done : torch.Tensor
            Whether or not the last step was a terminal step. We use this to clear the
            hidden state of the network when necessary, if it is recurrent.

        Returns
        -------
        value_pred : torch.Tensor
            Predicted value output from critic.
        action_dist : torch.distributions.Distribution
            Distribution over action space to sample from.
        hidden_state : torch.Tensor
            New hidden state after forward pass.
        """

        raise NotImplementedError
