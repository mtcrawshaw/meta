import numpy as np
import torch.nn as nn
from gym.spaces import Box, Discrete

from meta.utils import init, get_space_size


class PolicyNetwork(nn.Module):
    def __init__(self, observation_space, action_space, hidden_size=64):

        super(PolicyNetwork, self).__init__()
        self.action_space = action_space
        self.observation_space = observation_space
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
        self.actor = nn.Sequential(
            init_base(nn.Linear(self.input_size, hidden_size)),
            nn.Tanh(),
            init_base(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            init_final(nn.Linear(hidden_size, self.output_size)),
        )

        self.critic = nn.Sequential(
            init_base(nn.Linear(self.input_size, hidden_size)),
            nn.Tanh(),
            init_base(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            init_base(nn.Linear(hidden_size, 1)),
        )

        # Extra parameter vector for standard deviations in the case that
        # the policy distribution is Gaussian.
        if isinstance(action_space, Box):
            self.logstd = torch.zeros(self.output_size)

        self.train()

    def forward(self, obs):

        value_pred = self.critic(obs)
        actor_output = self.actor(obs)

        if isinstance(self.action_space, Discrete):
            # Matches torch.distribution.Categorical
            action_probs = {"logits": actor_output}
        elif isinstance(self.action_space, Box):
            # Matches torch.distribution.Normal
            action_probs = {"loc": actor_output, "scale": self.logstd.exp()}

        return value_pred, action_probs
