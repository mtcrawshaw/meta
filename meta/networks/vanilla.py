"""
Definition of VanillaNetwork, a module used to parameterize a vanilla actor/critic
policy.
"""

from typing import Tuple, List

import torch
import torch.nn as nn
from torch.distributions import Distribution, Categorical, Normal
import numpy as np
from gym.spaces import Space, Box, Discrete

from meta.networks.base import BaseNetwork, init_base, init_final, init_recurrent
from meta.utils.utils import AddBias


class VanillaNetwork(BaseNetwork):
    """ Module used to parameterize an actor/critic policy. """

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        num_processes: int,
        rollout_length: int,
        num_layers: int = 3,
        hidden_size: int = 64,
        recurrent: bool = False,
        device: torch.device = None,
    ) -> None:

        self.num_layers = num_layers
        if self.num_layers < 1:
            raise ValueError(
                "Number of layers in network should be at least 1. Given value is: %d"
                % self.num_layers
            )

        super(VanillaNetwork, self).__init__(
            observation_space=observation_space,
            action_space=action_space,
            num_processes=num_processes,
            rollout_length=rollout_length,
            hidden_size=hidden_size,
            recurrent=recurrent,
            device=device,
        )

    def initialize_network(self) -> None:
        """ Initialize layers of network. """

        # Initialize recurrent layer, if necessary.
        if self.recurrent:
            self.gru = init_recurrent(nn.GRU(self.input_size, self.hidden_size))
            self.hidden_state = torch.zeros(self.hidden_size)

        # Initialize feedforward actor/critic layers.
        actor_layers = []
        critic_layers = []
        for i in range(self.num_layers):

            # Calcuate input and output size of layer.
            layer_input_size = (
                self.input_size if i == 0 and not self.recurrent else self.hidden_size
            )
            actor_output_size = (
                self.output_size if i == self.num_layers - 1 else self.hidden_size
            )
            critic_output_size = 1 if i == self.num_layers - 1 else self.hidden_size

            # Determine init function for actor layer. Note that all layers of critic
            # are initialized with init_base, so we only need to do this for actor.
            actor_init = init_base if i < self.num_layers - 1 else init_final

            actor_layers.append(
                actor_init(nn.Linear(layer_input_size, actor_output_size))
            )
            critic_layers.append(
                init_base(nn.Linear(layer_input_size, critic_output_size))
            )

            # Activation functions.
            if i != self.num_layers - 1:
                actor_layers.append(nn.Tanh())
                critic_layers.append(nn.Tanh())

        self.actor = nn.Sequential(*actor_layers)
        self.critic = nn.Sequential(*critic_layers)

        # Extra parameter vector for standard deviations in the case that
        # the policy distribution is Gaussian.
        if isinstance(self.action_space, Box):
            self.logstd = AddBias(torch.zeros(self.output_size))

    def forward(
        self, obs: torch.Tensor, hidden_state: torch.Tensor, done: torch.Tensor,
    ) -> Tuple[torch.Tensor, Distribution, torch.Tensor]:
        """
        Forward pass definition for VanillaNetwork.

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

        x = obs

        # Pass through recurrent layer, if necessary.
        if self.recurrent:
            x, hidden_state = self.recurrent_forward(x, hidden_state, done)

        # Pass through actor and critic networks.
        value_pred = self.critic(x)
        actor_output = self.actor(x)

        # Construct action distribution from actor output.
        action_dist = self.get_action_distribution(actor_output)

        return value_pred, action_dist, hidden_state
