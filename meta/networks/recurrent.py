"""
Definition of RecurrentBlock, which serves as a building block for other networks.
"""

from typing import Tuple, List

import torch
import torch.nn as nn
from gym.spaces import Space

from meta.networks.initialize import init_recurrent


class RecurrentBlock(nn.Module):
    """ Recurrent building block for larger networks. """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        observation_shape: Tuple[int, ...],
        num_processes: int,
        rollout_length: int,
        device: torch.device = None,
    ) -> None:

        super(RecurrentBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.observation_shape = observation_shape
        self.num_processes = num_processes
        self.rollout_length = rollout_length

        # Generate network layers.
        self.initialize_network()

        # Set device.
        self.device = device if device is not None else torch.device("cpu")
        self.to(device)

    def initialize_network(self) -> None:
        """ Initialize layers of network. Must be defined in child classes. """
        self.gru = init_recurrent(nn.GRU(self.input_size, self.hidden_size))
        self.hidden_state = torch.zeros(self.hidden_size)

    def forward(
        self, inputs: torch.Tensor, hidden_state: torch.Tensor, done: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through RecurrentBlock.

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
        if inputs.shape == (self.num_processes, *self.observation_shape):

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

        elif inputs.shape[1:] == self.observation_shape:

            # The first dimension should be made of concatenated trajectories from
            # mutiple processes, so that the length of this dimension should be a
            # multiple of the rollout length.
            if inputs.shape[0] % self.rollout_length != 0:
                raise ValueError("Invalid tensor shape, can't process input.")
            num_trajectories = inputs.shape[0] // self.rollout_length

            # Flatten inputs and dones, and give hidden_state a temporal dimension.
            inputs = inputs.view(
                self.rollout_length, num_trajectories, *self.observation_shape
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
