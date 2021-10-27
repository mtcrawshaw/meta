"""
Dataset object for CelebA dataset. This is a wrapper around the PyTorch CelebA Dataset
object, which additionally contains information about metrics, loss function, input and
output size, etc.
"""

from typing import Dict

import torch
import torch.nn as nn
from torchvision.datasets import CelebA as torch_CelebA

from meta.datasets import BaseDataset
from meta.datasets.utils import get_split, RGB_TRANSFORM


class CelebA(torch_CelebA, BaseDataset):
    """ CelebA dataset wrapper. """

    def __init__(self, root: str, train: bool = True) -> None:
        """ Init function for CelebA. """

        split = "train" if train else "test"
        torch_CelebA.__init__(
            self,
            root=root,
            target_type="attr",
            split=split,
            transform=RGB_TRANSFORM,
            download=True,
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
