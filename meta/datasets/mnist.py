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
from meta.datasets.utils import GRAY_TRANSFORM


class MNIST(torch_MNIST):
    """ MNIST dataset wrapper. """

    input_size = (1, 28, 28)
    output_size = 10
    loss_cls = nn.CrossEntropyLoss
    loss_kwargs = {}
    criterion_kwargs = {"train": {}, "eval": {}}
    extra_metrics = {
        "accuracy": {
            "maximize": True,
            "train": True,
            "eval": True,
            "show": True,
        },
    }
    dataset_kwargs = {
        "train": {"download": True, "transform": GRAY_TRANSFORM},
        "eval": {"download": True, "transform": GRAY_TRANSFORM},
    }

    def __init__(self, root: str, train: bool = True) -> None:
        """ Init function for MNIST. """

        split = "train" if train else "eval"
        kwargs = MNIST.dataset_kwargs[split]
        super(MNIST, self).__init__(root=root, train=train, **kwargs)

    @staticmethod
    def compute_metrics(outputs: torch.Tensor, labels: torch.Tensor, criterion: nn.Module = None, train: bool = True) -> Dict[str, float]:
        """
        Compute metrics from `outputs` and `labels`, returning a dictionary whose keys
        are metric names and whose values are floats. Note that the returned metrics
        should match those listed in `extra_metrics`.
        """
        split = "train" if train else "eval"
        return {f"{split}_accuracy": get_accuracy(outputs, labels)}
