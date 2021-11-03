""" Misc functionality for meta/networks. """

from typing import Any, Union, Callable, List

import numpy as np
import torch
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
    batch_norm: bool = False,
) -> nn.Module:
    """
    Construct a fully-connected layer with the given input size, output size, activation
    function, and initialization function.
    """

    layer = []
    layer.append(layer_init(nn.Linear(in_size, out_size)))
    if batch_norm:
        bn = nn.BatchNorm2d(out_channels)
        nn.init.constant_(bn.weight, 1)
        nn.init.constant_(bn.bias, 0)
        layer.append(bn)
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


class IntermediateLayerGetter(nn.Sequential):
    """
    Note: This class was based on IntermediateLayerGetter from
    torchvision/models/_utils.py from the torchvision repo.

    Module wrapper that returns a single intermediate layer from a model.

    It has a strong assumption that the modules have been registered into the model in
    the same order as they are used.  This means that one should **not** reuse the same
    nn.Module twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly assigned to the
    model. So if `model` is passed, `model.feature1` can be returned, but not
    `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features
        m_name (str): name of module for which activation will be returned
    """

    def __init__(self, model: nn.Module, m_name: str) -> None:
        if not m_name in [name for name, _ in model.named_children()]:
            raise ValueError(f"module with name '{m_name}' not present in model")
        layers = []
        for name, module in model.named_children():
            layers.append(module)
            if name == m_name:
                break

        super(IntermediateLayerGetter, self).__init__(*layers)


class Parallel(nn.Module):
    """
    Module container that executes a set of modules in parallel. This is analagous to
    nn.Sequential. Holds a list of modules, and the input to Parallel will be fed to
    each module in the list. The outputs of each module are combined along an existing
    or new dimension and returned as a single Tensor. Note that the output of each
    module is not actually computed in parallel, i.e. this happens in a for loop.
    """

    def __init__(self, modules: List[nn.Module], combine_dim=0, new_dim=False) -> None:
        """ Init function for Parallel. """
        super(Parallel, self).__init__()
        self.p_modules = nn.ModuleList(modules=modules)
        self.combine_dim = combine_dim
        self.new_dim = new_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Computes the output of each module in `self.p_modules` when given input `inputs`.
        The results are stacked or concatenated and returned as a single Tensor.
        """

        # Compute output of each module.
        outs = [module(inputs) for module in self.p_modules]

        # Combine outputs of each module.
        if self.new_dim:
            out = torch.stack(outs, dim=self.combine_dim)
        else:
            out = torch.cat(outs, dim=self.combine_dim)

        return out

    def __getitem__(self, i: int) -> nn.Module:
        """ Access `i`th module. """
        return self.p_modules[i]
