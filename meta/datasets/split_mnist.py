""" Dataset object for continual learning on SplitMNIST. """

from typing import Dict

import torch
import torch.nn as nn
from torchvision.datasets import MNIST

from meta.train.loss import get_accuracy
from meta.datasets import ContinualDataset
from meta.datasets.utils import GRAY_TRANSFORM, get_split


class SplitMNIST(ContinualDataset):
    """
    SplitMNIST dataset. This is a continual learning dataset made from MNIST data. Each
    of the 5 tasks is a binary classification task in which the model must distinguish
    between two consecutive digits. The first task uses digis 0-1, the second uses 2-3,
    etc.
    """

    def __init__(self, root: str, train: bool = True) -> None:
        """ Init function for MNIST. """

        super().__init__(self)

        self.input_size = (1, 28, 28)
        self.output_size = 10
        self.loss_cls = nn.CrossEntropyLoss
        self.extra_metrics = {
            "accuracy": {"maximize": True, "train": True, "eval": True, "show": True},
        }

        self.num_tasks = 5
        self.current_task = 0

        # Construct a dataset for each task.
        self.task_datasets = []
        for task in range(self.num_tasks):
            pass

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
