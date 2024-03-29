"""
Definition of BackboneNetwork, a network with a pre-trained backbone and a newly
initialized output head. This network is meant to be used for dense computer vision
tasks, i.e. the output shape has the same spatial resolution as the input.
"""

from typing import Tuple, Union, List, Iterator

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


PRETRAINED_MODELS = ["resnet"]
SIMPLE_MODELS = ["conv"]
ARCH_TYPES = SIMPLE_MODELS + PRETRAINED_MODELS
PRETRAINED_LAYERS = {"resnet": [18, 34, 50, 101, 152]}
PRETRAINED_INCHANNELS = {"resnet": 64}
PRETRAINED_OUTCHANNELS = {"resnet": {18: 512, 34: 512, 50: 2048, 101: 2048, 152: 2048}}


class BackboneNetwork(nn.Module):
    """ Module used to parameterize a backbone network. """

    def __init__(
        self,
        input_size: Tuple[int, int, int],
        output_size: Union[Tuple[int, int, int], List[Tuple[int, int, int]]],
        arch_type: str,
        num_backbone_layers: int,
        num_head_layers: int,
        head_channels: int,
        initial_channels: int = None,
        pretrained: bool = False,
        device: torch.device = None,
    ) -> None:
        """
        Init function for BackboneNetwork.

        Parameters
        ----------
        input_size : Tuple[int, int, int]
            Input size for network, describing dimension of input images. Order of
            dimensions is [channels, width, height].
        output_size : Union[Tuple[int, int, int], List[Tuple[int, int, int]]]
            Output size(s) for network. If this is a list of tuples, then the network
            will have one shared trunk that feeds into task-specific output heads. Order
            of channels is same as `input_size`.
        arch_type : str
            Trunk architectuer type. The options are listed in `ARCH_TYPES`.
        num_backbone_layers : int
            Number of layers in backbone of network.
        num_head_layers: int
            Number of layers in each task's output head.
        head_channels : int
            Number of channels in each layer of each task-specific output head.
        initial_channels : int
            Number of channels in first layer of network. This value is ignored when
            using a pretrained architecture that has a fixed number of initial channels,
            such as ResNet.
        pretrained : bool
            Whether or not to use pre-trained weights. This is only supported when
            `arch_type` is in `PRETRAINED_MODELS`.
        device : torch.device
            Network device. Defaults to CPU.
        """

        super(BackboneNetwork, self).__init__()

        # Check architecture type.
        if arch_type not in ARCH_TYPES:
            raise ValueError(
                "Architecture '%s' not supported for BackboneNetwork." % arch_type
            )

        # Set state and check for validity.
        self.input_size = input_size
        self.output_size = output_size
        self.num_backbone_layers = num_backbone_layers
        self.num_head_layers = num_head_layers
        self.head_channels = head_channels
        self.arch_type = arch_type
        self.initial_channels = initial_channels
        if self.arch_type in PRETRAINED_MODELS:
            assert self.num_backbone_layers in PRETRAINED_LAYERS[self.arch_type]
            self.initial_channels = PRETRAINED_INCHANNELS[self.arch_type]
        else:
            assert self.initial_channels is not None
        self.pretrained = pretrained
        if self.pretrained:
            assert self.arch_type in PRETRAINED_MODELS
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

        if self.arch_type == "conv":

            backbone_layers = [
                get_conv_layer(
                    in_channels=self.input_size[0],
                    out_channels=self.initial_channels,
                    activation="relu",
                    layer_init=init_base,
                    batch_norm=True,
                )
            ]
            backbone_layers += [
                get_conv_layer(
                    in_channels=(self.initial_channels * (2 ** i)),
                    out_channels=(self.initial_channels * (2 ** (i + 1))),
                    activation="relu",
                    layer_init=init_base,
                    batch_norm=True,
                )
                for i in range(self.num_backbone_layers - 1)
            ]
            self.backbone = nn.Sequential(*backbone_layers)
            backbone_out_channels = self.initial_channels * (
                2 ** (self.num_backbone_layers - 1)
            )

        elif self.arch_type == "resnet":

            # Initialize backbone and get features (before spatial dimensions are
            # collapsed).
            backbone_cls = eval(
                "torchvision.models.resnet%d" % self.num_backbone_layers
            )
            self.backbone = backbone_cls(
                pretrained=self.pretrained,
                replace_stride_with_dilation=[True, True, True],
            )
            self.backbone = IntermediateLayerGetter(self.backbone, "layer4")
            backbone_out_channels = PRETRAINED_OUTCHANNELS[self.arch_type][
                self.num_backbone_layers
            ]

        else:
            raise NotImplementedError

        def get_head(num_layers, head_channels, output_size):
            """ Helper function to get output head. """
            head_layers = []
            for i in range(num_layers):
                in_channels = backbone_out_channels if i == 0 else head_channels
                out_channels = output_size if i == num_layers - 1 else head_channels
                activation = None if i == num_layers - 1 else "relu"
                batch_norm = i < num_layers - 1
                kernel_size = 1 if i == num_layers - 1 else 3
                head_layers.append(
                    get_conv_layer(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        activation=activation,
                        layer_init=init_base,
                        batch_norm=batch_norm,
                        kernel_size=kernel_size,
                    )
                )
            return nn.Sequential(*head_layers)

        # Initialize output head(s).
        if self.num_tasks == 1:
            self.head = get_head(
                self.num_head_layers, self.head_channels, self.output_size[0]
            )
        else:
            heads = [
                get_head(
                    self.num_head_layers, self.head_channels, self.output_size[task][0]
                )
                for task in range(self.num_tasks)
            ]
            self.head = Parallel(heads, combine_dim=1)

            # Save number of shared and task-specific parameters.
            self.num_shared_params = sum(
                [p.numel() for p in self.backbone.parameters()]
            )
            self.num_specific_params = {
                task: sum([p.numel() for p in self.head[task].parameters()])
                for task in range(self.num_tasks)
            }

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
        features = self.backbone(inputs)
        out = self.head(features)

        # Upsample output to match input resolution.
        out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)

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
