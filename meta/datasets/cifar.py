"""
Dataset object for CIFAR datasets. These are wrappers around the PyTorch CIFAR10 and
CIFAR100 Datasets, which additionally contain information about metrics to compute, data
augmentation, loss function, etc.
"""

from typing import Dict, List, Any

import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10 as torch_CIFAR10, CIFAR100 as torch_CIFAR100

from meta.train.loss import get_accuracy
from meta.datasets import BaseDataset
from meta.datasets.utils import RGB_TRANSFORM, get_split


class CIFAR10(torch_CIFAR10, BaseDataset):
    """ CIFAR10 dataset wrapper. """

    input_size = (3, 32, 32)
    output_size = 10
    loss_cls = nn.CrossEntropyLoss
    extra_metrics = {
        "accuracy": {"maximize": True, "train": True, "eval": True, "show": True},
    }

    def __init__(self, root: str, train: bool = True) -> None:
        """ Init function for CIFAR10. """
        super(CIFAR10, self).__init__(
            root=root, train=train, download=True, transform=RGB_TRANSFORM
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


class CIFAR100(torch_CIFAR100, BaseDataset):
    """ CIFAR100 dataset wrapper. """

    input_size = (3, 32, 32)
    output_size = 100
    loss_cls = nn.CrossEntropyLoss
    extra_metrics = {
        "accuracy": {"maximize": True, "train": True, "eval": True, "show": True},
    }

    def __init__(self, root: str, train: bool = True) -> None:
        """ Init function for CIFAR100. """
        super(CIFAR100, self).__init__(
            root=root, train=train, download=True, transform=RGB_TRANSFORM
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
