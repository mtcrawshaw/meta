"""
Definition of PolicyNetwork, the module used to parameterize an actor/critic policy.
"""

from typing import Tuple, Dict

import torch
import torch.nn as nn
import numpy as np
from gym.spaces import Box, Discrete

from meta.utils import get_space_size, init, AddBias


class PolicyNetwork(nn.Module):
    """ Module used to parameterize an actor/critic policy. """

    def __init__(self, observation_space, action_space, num_layers=3, hidden_size=64):

        super(PolicyNetwork, self).__init__()
        self.action_space = action_space
        self.observation_space = observation_space
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Calculate the input/output size.
        self.input_size = get_space_size(observation_space)
        self.output_size = get_space_size(action_space)

        # Initialization functions for network, init_final is only used for the last
        # layer of the actor network.
        init_base = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
        )
        init_final = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01
        )

        # Generate layers of network.
        actor_layers = []
        critic_layers = []
        for i in range(num_layers):

            # Calcuate input and output size of layer.
            layer_input_size = self.input_size if i == 0 else hidden_size
            actor_output_size = self.output_size if i == num_layers - 1 else hidden_size
            critic_output_size = 1 if i == num_layers - 1 else hidden_size

            # Determine init function for actor layer. Note that all layers of critic
            # are initialized with init_base, so we only need to do this for actor.
            actor_init = init_base if i < num_layers - 1 else init_final

            actor_layers.append(
                actor_init(nn.Linear(layer_input_size, actor_output_size))
            )
            critic_layers.append(
                init_base(nn.Linear(layer_input_size, critic_output_size))
            )

            # Activation functions.
            if i != num_layers - 1:
                actor_layers.append(nn.Tanh())
                critic_layers.append(nn.Tanh())

        self.actor = nn.Sequential(*actor_layers)
        self.critic = nn.Sequential(*critic_layers)

        # Extra parameter vector for standard deviations in the case that
        # the policy distribution is Gaussian.
        if isinstance(action_space, Box):
            self.logstd = AddBias(torch.zeros(self.output_size))

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass definition for PolicyNetwork.

        Arguments
        ---------
        obs : torch.Tensor
            Observation to be used as input to policy network. If the observation space
            is discrete, this function expects ``obs`` to be a one-hot vector.

        Returns
        -------
        value_pred : torch.Tensor
            Predicted value output from critic.
        action_probs : Dict[str, torch.Tensor]
            Parameterization of policy distribution. Keys and values match
            argument structure for init functions of torch.Distribution.
        """

        value_pred = self.critic(obs)
        actor_output = self.actor(obs)

        if isinstance(self.action_space, Discrete):
            # Matches torch.distribution.Categorical
            action_probs = {"logits": actor_output}
        elif isinstance(self.action_space, Box):
            # Matches torch.distribution.Normal
            action_logstd = self.logstd(torch.zeros(actor_output.size()))
            action_probs = {"loc": actor_output, "scale": action_logstd.exp()}

        return value_pred, action_probs
