from typing import Iterator

import torch.nn as nn

from meta.networks.actorcritic import ActorCriticNetwork
from meta.networks.backbone import BackboneNetwork, PRETRAINED_MODELS
from meta.networks.conv import ConvNetwork
from meta.networks.mlp import MLPNetwork
from meta.networks.recurrent import RecurrentBlock
from meta.networks.trunk import MultiTaskTrunkNetwork
from meta.networks.splitting import (
    MultiTaskSplittingNetworkV1,
    MultiTaskSplittingNetworkV2,
    MetaSplittingNetwork,
)


def last_shared_params(net: nn.Module) -> Iterator[nn.Parameter]:
    """
    Return an Iterator over the parameters of the last layer in `net` whose parameters
    are shared between multiple tasks. This is used in the GradNorm implementation,
    since this method only considers the gradients of the last layer of shared
    parameters. Note that this only applies in the multi-task setting, so `net` must be
    an instance of `BackboneNetwork` or `MultiTaskTrunkNetwork` and `net` must be a
    multi-task network.
    """

    # Check for valid network.
    is_backbone = isinstance(net, BackboneNetwork) and net.num_tasks > 1
    is_trunk = isinstance(net, MultiTaskTrunkNetwork)
    if not is_backbone and not is_trunk:
        raise ValueError(
            "To use GradNorm, network must be either be TrunkNetwork"
            " or BackboneNetwork with multiple tasks."
        )

    # Get last shared layer of parameters.
    if is_backbone:
        params = net.backbone[-1].parameters()
    elif is_trunk:
        params = net.trunk[-1].parameters()
    else:
        # This should never execute.
        assert False

    return params
