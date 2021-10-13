from typing import List

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
