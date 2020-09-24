"""
Definition of MLPNetwork, a multi-layer perceptron module.
"""

from typing import Callable

import torch
import torch.nn as nn

from meta.networks.recurrent import RecurrentBlock
from meta.utils.utils import AddBias


class MLPNetwork(nn.Module):
    """
    Module used to parameterize an actor/critic policy. `base_init` is the
    initialization function used to initialize all layers except for the last, and
    `final_init` is the initialization function used to initialize the last layer.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        init_base: Callable[[nn.Module], nn.Module],
        init_final: Callable[[nn.Module], nn.Module],
        num_layers: int = 3,
        hidden_size: int = 64,
        device: torch.device = None,
    ) -> None:

        super(MLPNetwork, self).__init__()

        # Check number of layers.
        if num_layers < 1:
            raise ValueError(
                "Number of layers in network should be at least 1. Given value is: %d"
                % num_layers
            )

        # Set state.
        self.input_size = input_size
        self.output_size = output_size
        self.init_base = init_base
        self.init_final = init_final
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Set device.
        self.device = device if device is not None else torch.device("cpu")

        # Generate network layers.
        self.initialize_network()

        # Move model to device.
        self.to(self.device)

    def initialize_network(self) -> None:
        """ Initialize layers of network. """

        # Initialize feedforward actor/critic layers.
        layers = []
        for i in range(self.num_layers):

            # Calcuate input and output size of layer.
            layer_input_size = self.input_size if i == 0 else self.hidden_size
            layer_output_size = (
                self.output_size if i == self.num_layers - 1 else self.hidden_size
            )

            # Determine init function for actor layer. Note that all layers of critic
            # are initialized with init_base, so we only need to do this for actor.
            layer_init = self.init_base if i < self.num_layers - 1 else self.init_final

            layers.append(layer_init(nn.Linear(layer_input_size, layer_output_size)))

            # Activation function.
            if i != self.num_layers - 1:
                layers.append(nn.Tanh())

        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass definition for MLPNetwork.

        Arguments
        ---------
        inputs : torch.Tensor
            Input to MLP network.

        Returns
        -------
        outputs : torch.Tensor
            Output of MLP network when given ``inputs`` as input.
        """

        return self.layers(inputs)
