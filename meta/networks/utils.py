""" Misc functionality for meta/networks. """

from collections import OrderedDict
from typing import Any, Dict, Union, Callable

import torch
import numpy as np
import torch.nn as nn


def init(
    module: nn.Module, weight_init: Any, bias_init: Any, gain: Union[float, int] = 1
) -> nn.Module:
    """ Helper function to initialize network weights. """

    # This is a somewhat gross way to handle both Linear/Conv modules and GRU modules.
    # It can probably be cleaned up.
    if hasattr(module, "weight") and hasattr(module, "bias"):
        weight_init(module.weight.data, gain=gain)
        bias_init(module.bias.data)
    else:
        for name, param in module.named_parameters():
            if "weight" in name:
                weight_init(param)
            elif "bias" in name:
                bias_init(param)

    return module


def get_fc_layer(
    in_size: int,
    out_size: int,
    activation: str,
    layer_init: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """
    Construct a fully-connected layer with the given input size, output size, activation
    function, and initialization function.
    """

    layer = []
    layer.append(layer_init(nn.Linear(in_size, out_size)))
    if activation is not None:
        layer.append(get_activation(activation))
    return nn.Sequential(*layer)


def get_conv_layer(
    in_channels: int,
    out_channels: int,
    activation: str,
    layer_init: Callable[[nn.Module], nn.Module],
    batch_norm: bool = False,
    kernel_size: int = 3,
) -> nn.Module:
    """
    Construct a convolutional layer with the given number of input channels and output
    channels, using the initialization function `layer_init`. The input is padded to
    preserve the spatial resolution through the layer.
    """

    layer = []
    assert kernel_size % 2 == 1
    padding = (kernel_size - 1) // 2
    layer.append(
        layer_init(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, padding=padding
            )
        )
    )
    if batch_norm:
        bn = nn.BatchNorm2d(out_channels)
        nn.init.constant_(bn.weight, 1)
        nn.init.constant_(bn.bias, 0)
        layer.append(bn)
    if activation is not None:
        layer.append(get_activation(activation))
    return nn.Sequential(*layer)


def get_activation(activation: str) -> nn.Module:
    """ Get single activation layer by name. """

    layer = None
    if activation == "tanh":
        layer = nn.Tanh()
    elif activation == "relu":
        layer = nn.ReLU()
    else:
        raise ValueError("Unsupported activation function: %s" % activation)

    return layer


# Initialization functions for network weights. `init_downscale` is usually only used for
# the last layer of the actor network, `init_recurrent` is used for the recurrent block,
# and `init_base` is used for all other layers in actor/critic networks. We initialize
# the final layer of the actor network with much smaller weights than all other network
# layers, as recommended by https://arxiv.org/abs/2006.05990. For `init_base` is also
# used for convolutional layers.
init_recurrent = lambda m: init(
    m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.0
)
init_base = lambda m: init(
    m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
)
init_downscale = lambda m: init(
    m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01
)


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Note: This class was copied out of torchvision/models/_utils.py from the torchvision
    repo.

    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered into the model in
    the same order as they are used.  This means that one should **not** reuse the same
    nn.Module twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly assigned to the
    model. So if `model` is passed, `model.feature1` can be returned, but not
    `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names of the modules
            for which the activations will be returned as the key of the dict, and the
            value of the dict is the name of the returned activation (which the user can
            specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """

    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset(
            [name for name, _ in model.named_children()]
        ):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class CosineSimilarityLoss(nn.Module):
    """
    Returns negative mean of cosine similarity between two tensors computed along `dim`.
    """

    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        super(CosineSimilarityLoss, self).__init__()
        self.single_loss = nn.CosineSimilarity(dim=dim, eps=eps)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return (1 - torch.mean(self.single_loss(x1, x2))) / 2.0
