"""
Definition of ResNetwork, a network mimicking ResNets. This is a temporary object that
will soon be removed.
"""

from math import floor
from typing import Tuple, List, Iterator

import torch
import torch.nn as nn

from meta.networks.utils import get_fc_layer, get_resnet_layer, init_base, Parallel


class ResNetwork(nn.Module):
    """ Module used to parameterize a multi-task residual network. """

    def __init__(
        self,
        input_size: Tuple[int, int, int],
        output_size: List[int],
        num_layers: int = 3,
        width: int = 64,
        num_downscales: int = 0,
        activation: str = "tanh",
        batch_norm: bool = False,
        device: torch.device = None,
    ) -> None:

        super(ResNetwork, self).__init__()

        # Check validity of arguments. This currently only supports sets of tasks that
        # have equal output size.
        for i in range(len(output_size)):
            assert output_size[0] == output_size[i]

        # Set state.
        self.input_size = input_size
        self.output_size = output_size
        self.num_tasks = len(output_size)
        self.num_layers = num_layers
        self.width = width
        self.num_downscales = num_downscales
        self.activation = activation
        self.batch_norm = batch_norm
        self.scales_per_layer = (self.num_downscales + 1) / (self.num_layers - 1)

        # Set device.
        self.device = device if device is not None else torch.device("cpu")

        # Generate network layers.
        self.initialize_network()

        # Move model to device.
        self.to(self.device)

    def initialize_network(self) -> None:
        """ Initialize layers of network. """

        # Initialize shared backbone.
        backbone_layers = []
        for i in range(self.num_layers - 1):

            # Determine whether to downscale activations at current layer.
            current_seg = floor(self.scales_per_layer * i)
            next_seg = floor(self.scales_per_layer * (i + 1))
            next_seg = min(next_seg, self.num_downscales)
            downscale = current_seg != next_seg

            # Calculate input channels, output channels, and stride of layer.
            if i == 0:
                layer_in_channels = self.input_size[0]
            else:
                layer_in_channels = self.width * 2 ** current_seg
            layer_out_channels = self.width * 2 ** next_seg
            stride = 2 if downscale else 1

            # Initialize layer.
            backbone_layers.append(
                get_resnet_layer(
                    in_channels=layer_in_channels,
                    out_channels=layer_out_channels,
                    activation=self.activation,
                    layer_init=init_base,
                    batch_norm=self.batch_norm,
                    stride=stride,
                )
            )

        self.backbone = nn.Sequential(*backbone_layers)

        # Initialize task-specific output heads.
        heads = [
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(1),
                get_fc_layer(
                    in_size=self.width * 2 ** self.num_downscales,
                    out_size=self.output_size[0],
                    activation=None,
                    layer_init=init_base,
                    batch_norm=False,
                ),
            )
            for _ in range(self.num_tasks)
        ]
        self.head = Parallel(heads, combine_dim=1, new_dim=True)

        # Save number of shared and task-specific parameters.
        self.num_shared_params = sum([p.numel() for p in self.backbone.parameters()])
        self.num_specific_params = {
            task: sum([p.numel() for p in self.head[task].parameters()])
            for task in range(self.num_tasks)
        }

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

        # Pass input through backbone and output head.
        features = self.backbone(inputs)
        out = self.head(features)
        return out

    def last_shared_params(self) -> Iterator[nn.Parameter]:
        """
        Return a list of the parameters of the last layer in `self` whose parameters are
        shared between multiple tasks.
        """
        return self.backbone[-1].parameters()

    def shared_params(self) -> Iterator[nn.parameter.Parameter]:
        """ Iterator over parameters which are shared between all tasks. """
        return self.backbone.parameters()

    def specific_params(self, task: int) -> Iterator[nn.parameter.Parameter]:
        """ Iterator over task-specific parameters. """
        assert self.num_tasks > 1
        return self.head[task].parameters()
