"""
Definition of ActorCriticNetwork, a module used to parameterize a vanilla actor/critic
policy.
"""

from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.distributions import Distribution, Categorical, Normal
from gym.spaces import Space, Box, Discrete

from meta.networks.initialize import init_base, init_final
from meta.networks.mlp import MLPNetwork
from meta.networks.recurrent import RecurrentBlock
from meta.utils.utils import AddBias, get_space_size


class ActorCriticNetwork(nn.Module):
    """ Module used to parameterize an actor/critic policy. """

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        num_processes: int,
        rollout_length: int,
        architecture_config: Dict[str, Any],
        device: torch.device = None,
    ) -> None:

        super(ActorCriticNetwork, self).__init__()

        # Set state.
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_processes = num_processes
        self.rollout_length = rollout_length
        self.device = device if device is not None else torch.device("cpu")
        self.architecture_type = architecture_config["type"]
        self.recurrent = architecture_config["recurrent"]
        self.hidden_size = architecture_config["hidden_size"]

        # Compute input and output sizes.
        self.input_size = get_space_size(observation_space)
        self.output_size = get_space_size(action_space)

        # Initialize network.
        self.initialize_network(architecture_config)

        # Move to device.
        self.to(device)

    def initialize_network(self, architecture_config: Dict[str, Any]) -> None:
        """
        Initialize pieces of the network. These are recurrent block (optional), actor,
        and critic networks.
        """

        # Initialize recurrent block, if necessary.
        if architecture_config["recurrent"]:
            self.recurrent_block = RecurrentBlock(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                observation_space=self.observation_space,
                num_processes=self.num_processes,
                rollout_length=self.rollout_length,
                device=self.device,
            )

        # Initialize actor and critic networks.
        architecture_kwargs = dict(architecture_config)
        del architecture_kwargs["type"]
        del architecture_kwargs["recurrent"]
        if architecture_config["type"] == "mlp":
            self.actor = MLPNetwork(
                input_size=self.input_size if not self.recurrent else self.hidden_size,
                output_size=self.output_size,
                init_base=init_base,
                init_final=init_final,
                device=self.device,
                **architecture_kwargs,
            )
            self.critic = MLPNetwork(
                input_size=self.input_size if not self.recurrent else self.hidden_size,
                output_size=1,
                init_base=init_base,
                init_final=init_base,
                device=self.device,
                **architecture_kwargs,
            )

        elif architecture_config["type"] == "trunk":
            raise NotImplementedError

        else:
            raise ValueError(
                "Unsupported architecture type: %s" % str(architecture_config["type"])
            )

        # Extra parameter vector for standard deviations in the case that
        # the policy distribution is Gaussian.
        if isinstance(self.action_space, Box):
            self.logstd = AddBias(torch.zeros(self.output_size))

    def forward(
        self, obs: torch.Tensor, hidden_state: torch.Tensor, done: torch.Tensor,
    ) -> Tuple[torch.Tensor, Distribution, torch.Tensor]:
        """
        Forward pass definition for ActorCriticNetwork.

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
            x, hidden_state = self.recurrent_block(x, hidden_state, done)

        # Pass through actor and critic networks.
        value_pred = self.critic(x)
        actor_output = self.actor(x)

        # Construct action distribution from actor output.
        if isinstance(self.action_space, Discrete):
            action_dist = Categorical(logits=actor_output)
        elif isinstance(self.action_space, Box):
            action_logstd = self.logstd(
                torch.zeros(actor_output.size(), device=self.device)
            )
            action_dist = Normal(loc=actor_output, scale=action_logstd.exp())
        else:
            raise NotImplementedError

        return value_pred, action_dist, hidden_state
