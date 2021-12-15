"""
Definition of ConvNetwork, a module consisting of a series of convolutional layers
followed by a series of fully connected layers.
"""

from typing import Tuple, Union, List

import torch
import torch.nn as nn

from meta.networks.utils import get_conv_layer, get_fc_layer, init_base


class ConvNetwork(nn.Module):
    """ Module used to parameterize a convolutional network. """

    def __init__(
        self,
        input_size: Tuple[int, int, int],
        num_conv_layers: int,
        initial_channels: int,
        num_fc_layers: int,
        fc_hidden_size: int,
        output_size: Union[int, List[int]],
        activation: str = "relu",
        batch_norm: bool = False,
        pooling: bool = False,
        device: torch.device = None,
    ) -> None:

        super(ConvNetwork, self).__init__()

        # Check number of layers.
        if num_conv_layers < 1 or num_fc_layers < 1:
            raise ValueError(
                "Number of conv/fc layers in network should each be at least 1."
                " Given values are: %d conv, %d fc." % (num_conv_layers, num_fc_layers)
            )

        # Check for multitask status.
        if isinstance(output_size, list):
            raise NotImplementedError

        # Set state.
        self.input_size = input_size
        self.num_conv_layers = num_conv_layers
        self.initial_channels = initial_channels
        self.num_fc_layers = num_fc_layers
        self.fc_hidden_size = fc_hidden_size
        self.output_size = output_size
        self.activation = activation
        self.batch_norm = batch_norm
        self.pooling = pooling

        # Set device.
        self.device = device if device is not None else torch.device("cpu")

        # Generate network layers.
        self.initialize_network()

        # Move model to device.
        self.to(self.device)

    def initialize_network(self) -> None:
        """ Initialize layers of network. """

        # Initialize convolutional layers.
        conv_layers = []
        in_channels = self.input_size[0]
        out_channels = self.initial_channels
        for i in range(self.num_conv_layers):

            # Initialize layer.
            conv_layers.append(
                get_conv_layer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    activation=self.activation,
                    layer_init=init_base,
                    batch_norm=self.batch_norm,
                )
            )

            # Set number of channels for next layer.
            in_channels = out_channels
            out_channels = out_channels

        self.conv = nn.Sequential(*conv_layers)

        # Initialize flattening layers to reshape convolutional features for input to
        # the fully connected layers.
        if self.pooling:
            self.flatten = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(1))
            self.feature_size = self.initial_channels
        else:
            self.flatten = nn.Flatten(1)
            self.feature_size = (
                self.initial_channels * self.input_size[1] * self.input_size[2]
            )

        # Initialize fully connected layers.
        fc_layers = []
        for i in range(self.num_fc_layers):

            # Determine input/output size of layer.
            last_layer = i == self.num_fc_layers - 1
            in_size = self.feature_size if i == 0 else self.fc_hidden_size
            out_size = self.output_size if last_layer else self.fc_hidden_size

            # Initialize_layer.
            fc_layers.append(
                get_fc_layer(
                    in_size=in_size,
                    out_size=out_size,
                    activation=self.activation if not last_layer else None,
                    layer_init=init_base,
                    batch_norm=self.batch_norm if not last_layer else False,
                )
            )

        self.fc = nn.Sequential(*fc_layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass definition for ConvNetwork.

        Arguments
        ---------
        inputs : torch.Tensor
            Input to network.

        Returns
        -------
        outputs : torch.Tensor
            Output of network when given `inputs` as input.
        """
        features = self.conv(inputs)
        features = self.flatten(features)
        out = self.fc(features)
        return out
