""" Misc functionality for meta/networks. """

from typing import Any, Union, Callable

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


def get_layer(
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

    if activation == "tanh":
        layer.append(nn.Tanh())
    elif activation == "relu":
        layer.append(nn.ReLU())
    elif activation is not None:
        raise ValueError("Unsupported activation function: %s" % activation)

    return nn.Sequential(*layer)


# Initialization functions for network weights. `init_downscale` is usually only used for
# the last layer of the actor network, `init_recurrent` is used for the recurrent block,
# and `init_base` is used for all other layers in actor/critic networks. We initialize
# the final layer of the actor network with much smaller weights than all other network
# layers, as recommended by https://arxiv.org/abs/2006.05990.
init_recurrent = lambda m: init(
    m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.0
)
init_base = lambda m: init(
    m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
)
init_downscale = lambda m: init(
    m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01
)
