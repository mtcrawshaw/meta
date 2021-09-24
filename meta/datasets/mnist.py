"""
Dataset object for MNIST dataset. This is a wrapper around the PyTorch MNIST Dataset
object, which additionally contains information about metrics to compute, data
augmentation, loss function, etc.
"""

from typing import Dict, List, Any

import torch
import torch.nn as nn
from torchvision.datasets import MNIST as torch_MNIST

from meta.train.loss import get_accuracy
from meta.datasets import BaseDataset
from meta.datasets.utils import GRAY_TRANSFORM, get_split


class MNIST(torch_MNIST, BaseDataset):
    """ MNIST dataset wrapper. """

    input_size = (1, 28, 28)
    output_size = 10
    loss_cls = nn.CrossEntropyLoss
    extra_metrics = {
        "accuracy": {"maximize": True, "train": True, "eval": True, "show": True},
    }

    def __init__(self, root: str, train: bool = True) -> None:
        """ Init function for MNIST. """
        super(MNIST, self).__init__(
            root=root, train=train, download=True, transform=GRAY_TRANSFORM
        )

    @staticmethod
    def compute_metrics(
        outputs: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module = None,
        train: bool = True,
    ) -> Dict[str, float]:
        """ Compute classification accuracy from `outputs` and `labels`. """
        split = get_split(train)
        return {f"{split}_accuracy": get_accuracy(outputs, labels)}
