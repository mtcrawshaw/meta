""" Dataset object for continual learning on split datasets. """

import os
from typing import Dict

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.datasets import MNIST, CIFAR10, CIFAR100


from meta.train.loss import get_accuracy, PartialCrossEntropyLoss
from meta.datasets import ContinualDataset
from meta.datasets.utils import get_split, GRAY_TRANSFORM, RGB_TRANSFORM


class Split(ContinualDataset):
    """
    Split MNIST/CIFAR dataset. This is a continual learning dataset made from
    MNIST/CIFAR data.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = False,
        dataset: str = "MNIST",
    ) -> None:
        """ Init function for Rotated. """

        super().__init__()

        # Check for valid arguments.
        assert dataset in ["MNIST", "CIFAR10", "CIFAR100"]

        if dataset == "MNIST":
            self.input_size = (1, 28, 28)
            self.output_size = 10
            self.dataset_cls = MNIST
            self.num_tasks = 5
            self.transform = GRAY_TRANSFORM
        elif dataset == "CIFAR10":
            self.input_size = (3, 32, 32)
            self.output_size = 10
            self.dataset_cls = CIFAR10
            self.num_tasks = 5
            self.transform = RGB_TRANSFORM
        elif dataset == "CIFAR100":
            self.input_size = (3, 32, 32)
            self.output_size = 100
            self.dataset_cls = CIFAR100
            self.num_tasks = 10
            self.transform = RGB_TRANSFORM
        else:
            assert False
        self.loss_cls = PartialCrossEntropyLoss
        self.loss_kwargs = {
            "num_classes": self.output_size,
            "num_tasks": self.num_tasks,
            "reduction": "mean",
        }
        self.extra_metrics = {
            "accuracy": {"maximize": True, "train": True, "eval": True, "show": True},
        }

        self.root = os.path.join(os.path.dirname(root), dataset)
        self.train = train
        self.download = download
        self.total_dataset = self.dataset_cls(
            root=self.root,
            train=self.train,
            transform=self.transform,
            download=self.download,
        )
        assert self.output_size % self.num_tasks == 0
        self.classes_per_task = self.output_size // self.num_tasks
        self.task_subindices = None
        self.task_len = None

        # Set current task.
        self.set_current_task(0)

    def __len__(self):
        """ Length of dataset. Overridden from parent class. """
        return len(self.task_subindices)

    def __getitem__(self, idx: int):
        """ Get item `idx` from dataset. Overridden from parent class. """
        return self.total_dataset[self.task_subindices[idx]]

    def set_current_task(self, new_task: int) -> None:
        """ Set the current training task to `new_task`. """

        super().set_current_task(new_task)

        # Get subindices for current task.
        self.task_subindices = []
        current_labels = list(
            range(
                self._current_task * self.classes_per_task,
                (self._current_task + 1) * self.classes_per_task,
            )
        )
        for i, (_, label) in enumerate(self.total_dataset):
            if label in current_labels:
                self.task_subindices.append(i)

    def compute_metrics(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module = None,
        train: bool = True,
    ) -> Dict[str, float]:
        """
        Compute classification accuracy from `outputs` and `labels`. Notice that we only
        consider predictions from the classes relevant to the current task.
        """
        split = get_split(train)
        start_class = self._current_task * self.classes_per_task
        end_class = (self._current_task + 1) * self.classes_per_task
        partial_outputs = outputs[:, start_class:end_class]
        partial_labels = labels - start_class
        return {f"{split}_accuracy": get_accuracy(partial_outputs, partial_labels)}
