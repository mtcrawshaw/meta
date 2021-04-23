"""
Definition of BackboneNetwork, a network with a pre-trained backbone and a newly
initialized output head. This network is meant to be used for dense computer vision
tasks, i.e. the output shape has the same spatial resolution as the input.
"""

from typing import Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from meta.networks.utils import (
    get_conv_layer,
    init_base,
    IntermediateLayerGetter,
    Parallel,
)


PRETRAINED_MODELS = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
OUT_CHANNELS = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "resnet152": 2048,
}


class BackboneNetwork(nn.Module):
    """ Module used to parameterize a backbone network. """

    def __init__(
        self,
        pretrained: bool,
        arch_type: str,
        head_channels: int,
        input_size: Tuple[int, int, int],
        output_size: Union[Tuple[int, int, int], List[Tuple[int, int, int]]],
        device: torch.device = None,
    ) -> None:

        super(BackboneNetwork, self).__init__()

        # Check architecture type.
        if arch_type not in PRETRAINED_MODELS:
            raise ValueError(
                "Architecture '%s' not supported for BackboneNetwork." % arch_type
            )

        # Set state.
        self.pretrained = pretrained
        self.arch_type = arch_type
        self.head_channels = head_channels
        self.input_size = input_size
        self.output_size = output_size
        if isinstance(self.output_size, tuple):
            assert len(self.output_size) == 3
            self.num_tasks = 1
        elif isinstance(self.output_size, list):
            self.num_tasks = len(self.output_size)
            for size in self.output_size:
                assert isinstance(size, tuple)
                assert len(size) == 3

        # Set device.
        self.device = device if device is not None else torch.device("cpu")

        # Generate network layers.
        self.initialize_network()

        # Move model to device.
        self.to(self.device)

    def initialize_network(self) -> None:
        """ Initialize backbone and output layers of network. """

        # Check that backbone type is ResNet. We do this because our method to get the
        # features computed by the pre-trained models is specific to the ResNet
        # implementation in torchvision.
        assert "resnet" in self.arch_type

        # Initialize backbone and get features (before spatial dimensions are
        # collapsed).
        backbone_cls = eval("torchvision.models.%s" % self.arch_type)
        self.backbone = backbone_cls(
            pretrained=self.pretrained, replace_stride_with_dilation=[True, True, True],
        )
        return_layers = {"layer4": "features"}
        self.backbone = IntermediateLayerGetter(
            self.backbone, return_layers=return_layers
        )
        backbone_out_channels = OUT_CHANNELS[self.arch_type]

        # Initialize output head(s).
        if self.num_tasks == 1:
            head_layers = [
                get_conv_layer(
                    in_channels=backbone_out_channels,
                    out_channels=self.head_channels,
                    activation="relu",
                    layer_init=init_base,
                    batch_norm=True,
                ),
                get_conv_layer(
                    in_channels=self.head_channels,
                    out_channels=self.output_size[0],
                    activation=None,
                    layer_init=init_base,
                    kernel_size=1,
                ),
            ]
            self.head = nn.Sequential(*head_layers)
        else:
            heads = []
            for task in range(self.num_tasks):
                head_layers = [
                    get_conv_layer(
                        in_channels=backbone_out_channels,
                        out_channels=self.head_channels,
                        activation="relu",
                        layer_init=init_base,
                        batch_norm=True,
                    ),
                    get_conv_layer(
                        in_channels=self.head_channels,
                        out_channels=self.output_size[task][0],
                        activation=None,
                        layer_init=init_base,
                        kernel_size=1,
                    ),
                ]
                heads.append(nn.Sequential(*head_layers))
            self.head = Parallel(heads, combine_dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass definition for BackboneNetwork.

        Arguments
        ---------
        inputs : torch.Tensor
            Input to network.

        Returns
        -------
        outputs : torch.Tensor
            Output of network when given `inputs` as input.
        """

        input_size = inputs.shape[-2:]

        # Pass input through backbone and output head.
        features = self.backbone(inputs)["features"]
        out = self.head(features)

        # Upsample output to match input resolution.
        out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)

        return out
