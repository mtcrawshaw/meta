"""
Definition of PolicyNetwork, the module used to parameterize an actor/critic policy.
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Distribution, Categorical, Normal
import numpy as np
from gym.spaces import Space, Box, Discrete

from meta.utils import get_space_size, get_space_shape, init, AddBias


class PolicyNetwork(nn.Module):
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

        super(PolicyNetwork, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_processes = num_processes
        self.rollout_length = rollout_length
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.recurrent = recurrent

        # Calculate the input/output size.
        self.input_size = get_space_size(observation_space)
        self.output_size = get_space_size(action_space)

        # Initialization functions for network, init_final is only used for the last
        # layer of the actor network, init_base is used for all other layers in
        # actor/critic networks, init_recurrent is used for recurrent layer.
        init_base = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
        )
        init_final = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01
        )
        init_recurrent = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.0
        )

        # Generate layers of network.
        if self.recurrent:
            self.gru = init_recurrent(nn.GRU(self.input_size, self.hidden_size))
            self.hidden_state = torch.zeros(self.hidden_size)

        actor_layers = []
        critic_layers = []
        for i in range(num_layers):

            # Calcuate input and output size of layer.
            layer_input_size = (
                self.input_size if i == 0 and not self.recurrent else hidden_size
            )
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

        # Set device.
        self.device = device if device is not None else torch.device("cpu")
        self.to(device)

    def forward(
        self, obs: torch.Tensor, hidden_state: torch.Tensor, done: torch.Tensor,
    ) -> Tuple[torch.Tensor, Distribution, torch.Tensor]:
        """
        Forward pass definition for PolicyNetwork.

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

        # Construct action distribution from actor_output.
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

    def recurrent_forward(
        self, inputs: torch.Tensor, hidden_state: torch.Tensor, done: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through recurrent layer of PolicyNetwork.

        Arguments
        ---------
        inputs : torch.Tensor
            Input to recurrent layer.
        hidden_state : torch.Tensor
            Hidden state to use for recurrent layer, if necessary.
        done : torch.Tensor
            Whether or not the previous environment step was terminal. We use this to
            clear the hidden state when necessary.

        Returns
        -------
        outputs : torch.Tensor
            Output of recurrent layer.
        hidden_state : torch.Tensor
            New hidden state of recurrent layer.
        """

        # Handle cases separately: temporal dimension has length 1, and temporal
        # dimension has length greater than 1. ``inputs`` holds a sequence of
        # observations, though if the sequence has length 1 then there is simply no
        # temporal dimension. To test for this then, we have to test the size of inputs
        # against the size of a single batch of observations. If a sequence is given,
        # the first two dimensions will be combined (this happens in the recurrent
        # minibatch generator).
        observation_shape = get_space_shape(self.observation_space, "obs")
        if inputs.shape == (self.num_processes, *observation_shape):

            # Clear the hidden state for any processes for which the environment just
            # finished.
            hidden_state = hidden_state * (1.0 - done)

            # The squeeze and unsqueeze here is to create a temporal dimension for the
            # recurrent module. The input is technically a sequence of length 1.
            output, hidden_state = self.gru(
                inputs.unsqueeze(0), hidden_state.unsqueeze(0)
            )
            output = output.squeeze(0)
            hidden_state = hidden_state.squeeze(0)

        elif inputs.shape[1:] == observation_shape:

            # The first dimension should be made of concatenated trajectories from
            # mutiple processes, so that the length of this dimension should be a
            # multiple of the rollout length.
            if inputs.shape[0] % self.rollout_length != 0:
                raise ValueError("Invalid tensor shape, can't process input.")
            num_trajectories = inputs.shape[0] // self.rollout_length

            # Flatten inputs and dones, and give hidden_state a temporal dimension.
            inputs = inputs.view(
                self.rollout_length, num_trajectories, *observation_shape
            )
            hidden_state = hidden_state.unsqueeze(0)
            done = done.view(self.rollout_length, num_trajectories)

            # Compute which steps of the sequence were terminal for some environment
            # process. This is to perform an optimization with the calls to self.gru. If
            # two consecutive steps both have all zeros in the corresponding rows of
            # ``done``, then there is no need to have two separate calls to self.gru,
            # one for each step. Instead, we could make one call, passing in a sequence
            # of both of them, since there is no need to reset the hidden state between
            # these steps. Leveraging this information, we can make one call to self.gru
            # for each timestep interval in which no process received a done=True from
            # the environment. We add 0 and self.rollout_length to this list to ensure
            # that the union of all intervals covers the entire input.
            interval_endpoints = (
                (done == 1.0).any(dim=1).nonzero().squeeze().cpu().tolist()
            )
            if not isinstance(interval_endpoints, list):
                # If there is only one nonzero entry, interval_endpoints will be an int.
                interval_endpoints = [interval_endpoints]
            if interval_endpoints == [] or interval_endpoints[0] != 0:
                interval_endpoints = [0] + interval_endpoints
            interval_endpoints += [self.rollout_length]

            # Forward pass for each interval with done=False.
            outputs: List[torch.Tensor] = []
            for endpoint_index in range(len(interval_endpoints) - 1):

                # Get endpoints of current interval.
                start = interval_endpoints[endpoint_index]
                end = interval_endpoints[endpoint_index + 1]

                # Clear the hidden state for any processes for which the environment just
                # finished. We have to create a view of done to make it compatible with
                # hidden_state.
                hidden_state = hidden_state * (1.0 - done[start]).view(
                    1, num_trajectories, 1
                )

                # Forward pass for a single timestep. We use this indexing on step to
                # preserve the first dimension.
                x, hidden_state = self.gru(inputs[start:end], hidden_state)
                outputs.append(x)

            # Combine outputs from each step into a single tensor, and remove temporal
            # dimension from hidden_state.
            output: torch.Tensor = torch.cat(outputs, dim=0)
            output = output.view(num_trajectories * self.rollout_length, -1)
            hidden_state = hidden_state.squeeze(0)

        else:
            raise ValueError(
                "Invalid input tensor shape, can't perform recurrent forward pass."
            )

        return output, hidden_state
