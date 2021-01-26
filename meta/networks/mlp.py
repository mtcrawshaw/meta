"""
Definition of MLPNetwork, a multi-layer perceptron module.
"""

import torch
import torch.nn as nn

from meta.networks.utils import get_layer, init_base, init_downscale


class MLPNetwork(nn.Module):
    """ Module used to parameterize an MLP. """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: str = "tanh",
        num_layers: int = 3,
        hidden_size: int = 64,
        downscale_last_layer: bool = False,
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
        self.activation = activation
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.downscale_last_layer = downscale_last_layer

        # Set device.
        self.device = device if device is not None else torch.device("cpu")

        # Generate network layers.
        self.initialize_network()

        # Move model to device.
        self.to(self.device)

    def initialize_network(self) -> None:
        """ Initialize layers of network. """

        # Initialize layers.
        layers = []
        for i in range(self.num_layers):

            # Calcuate input and output size of layer.
            layer_input_size = self.input_size if i == 0 else self.hidden_size
            layer_output_size = (
                self.output_size if i == self.num_layers - 1 else self.hidden_size
            )

            # Determine init function for layer.
            if i == self.num_layers - 1 and self.downscale_last_layer:
                layer_init = init_downscale
            else:
                layer_init = init_base

            # Initialize layer.
            layers.append(
                get_layer(
                    in_size=layer_input_size,
                    out_size=layer_output_size,
                    activation=self.activation if i != self.num_layers - 1 else None,
                    layer_init=layer_init,
                )
            )

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
