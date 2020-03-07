from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn

from meta.utils import init, get_space_size
from gym.spaces import Space, Box, Discrete


class PolicyNetwork(nn.Module):
    """ MLP network parameterizing the policy. """

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        num_layers: int = 3,
        hidden_size: int = 64,
    ):
        """
        init function for PolicyNetwork.

        Arguments
        ---------
        observation_space : Space
            Environment's observation space.
        action_space : Space
            Environment's action space.
        hidden_size : int
        """

        super(PolicyNetwork, self).__init__()
        self.action_space = action_space
        self.observation_space = observation_space

        # Calculate the input/output size.
        self.input_size = get_space_size(observation_space)
        self.output_size = get_space_size(action_space)

        # Instantiate modules.
        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
        )

        # Generate layers of network.
        self.hidden_size = hidden_size
        actor_layers = []
        critic_layers = []
        for i in range(num_layers):

            # Calcuate input and output size of layer.
            layer_input_size = self.input_size if i == 0 else hidden_size
            actor_output_size = self.output_size if i == num_layers - 1 else hidden_size
            critic_output_size = 1 if i == num_layers - 1 else hidden_size

            actor_layers.append(init_(nn.Linear(layer_input_size, actor_output_size)))
            critic_layers.append(init_(nn.Linear(layer_input_size, critic_output_size)))

            # Activation functions.
            if i != num_layers - 1:
                actor_layers.append(nn.Tanh())
                critic_layers.append(nn.Tanh())

        self.actor = nn.Sequential(*actor_layers)
        self.critic = nn.Sequential(*critic_layers)

        # Extra parameter vector for standard deviations in the case that
        # the policy distribution is Gaussian.
        if isinstance(action_space, Box):
            self.logstd = torch.zeros(self.output_size)

    def forward(self, obs: torch.Tensor) -> Tuple[float, Dict[str, torch.Tensor]]:
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
            action_probs = {"loc": actor_output, "scale": self.logstd.exp()}

        return value_pred, action_probs
