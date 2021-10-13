"""
Dataset object for MNIST dataset. This is a wrapper around the PyTorch MNIST Dataset
object, which additionally contains information about metrics, loss function, input and
output size, etc.
"""

from typing import Dict

import torch
import torch.nn as nn
from torchvision.datasets import MNIST as torch_MNIST

from meta.train.loss import get_accuracy
from meta.datasets import BaseDataset
from meta.datasets.utils import GRAY_TRANSFORM, get_split


class MNIST(torch_MNIST, BaseDataset):
    """ MNIST dataset wrapper. """

    def __init__(self, root: str, train: bool = True) -> None:
        """ Init function for MNIST. """

        torch_MNIST.__init__(
            self, root=root, train=train, download=True, transform=GRAY_TRANSFORM
        )
        BaseDataset.__init__(self)

        self.input_size = (1, 28, 28)
        self.output_size = 10
        self.loss_cls = nn.CrossEntropyLoss
        self.extra_metrics = {
            "accuracy": {"maximize": True, "train": True, "eval": True, "show": True},
        }

    def compute_metrics(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module = None,
        train: bool = True,
    ) -> Dict[str, float]:
        """ Compute classification accuracy from `outputs` and `labels`. """
        split = get_split(train)
        return {f"{split}_accuracy": get_accuracy(outputs, labels)}
