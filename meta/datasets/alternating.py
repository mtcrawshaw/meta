""" Dataset object for continual learning on alternating datasets. """

import os
from typing import Dict

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.datasets import MNIST, CIFAR10, CIFAR100


from meta.train.loss import get_accuracy
from meta.datasets import ContinualDataset
from meta.datasets.utils import GRAY_TRANSFORM, RGB_TRANSFORM, get_split


class Alternating(ContinualDataset):
    """
    Alternating MNIST/CIFAR dataset. This is a continual learning dataset made from
    MNIST/CIFAR data. Each of the tasks is a classification task over all classes of the
    original dataset, but the pixels from task i are inverted for when i is odd.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = False,
        dataset: str = "MNIST",
        num_tasks: int = 10,
    ) -> None:
        """ Init function for Rotated. """

        super().__init__()

        # Check for valid arguments.
        assert dataset in ["MNIST", "CIFAR10", "CIFAR100"]

        if dataset == "MNIST":
            self.input_size = (1, 28, 28)
            self.output_size = 10
            self.dataset_cls = MNIST
            self.transform = GRAY_TRANSFORM
        elif dataset == "CIFAR10":
            self.input_size = (3, 32, 32)
            self.output_size = 10
            self.dataset_cls = CIFAR10
            self.transform = RGB_TRANSFORM
        elif dataset == "CIFAR100":
            self.input_size = (3, 32, 32)
            self.output_size = 100
            self.dataset_cls = CIFAR100
            self.transform = RGB_TRANSFORM
        else:
            assert False
        self.loss_cls = nn.CrossEntropyLoss
        self.extra_metrics = {
            "accuracy": {"maximize": True, "train": True, "eval": True, "show": True},
        }

        self.root = os.path.join(os.path.dirname(root), dataset)
        self.train = train
        self.download = download
        self.num_tasks = num_tasks

        # Set current task.
        self.set_current_task(0)

    def __len__(self):
        """ Length of dataset. Overridden from parent class. """
        return len(self.current_dataset)

    def __getitem__(self, idx: int):
        """ Get item `idx` from dataset. Overridden from parent class. """
        return self.current_dataset[idx]

    def set_current_task(self, new_task: int) -> None:
        """ Set the current training task to `new_task`. """

        super().set_current_task(new_task)

        # Set inversion for current task and create corresponding dataset object.
        if self._current_task % 2 == 1:
            task_transform = transforms.Compose([InversionTransform(), self.transform])
        else:
            task_transform = self.transform
        self.current_dataset = self.dataset_cls(
            root=self.root,
            train=self.train,
            transform=task_transform,
            download=self.download,
        )

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


class InversionTransform:
    """ Invert the pixels of an image. This is just a wrapper around TF.invert(). """
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """ Compute inversion on input `x`. """
        return TF.invert(x)
