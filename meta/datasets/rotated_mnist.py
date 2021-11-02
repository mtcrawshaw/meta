""" Dataset object for continual learning on Rotated MNIST. """

from typing import Dict

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.datasets import MNIST


from meta.train.loss import get_accuracy
from meta.datasets import ContinualDataset
from meta.datasets.utils import GRAY_TRANSFORM, get_split


class RotatedMNIST(ContinualDataset):
    """
    Rotated MNIST dataset. This is a continual learning dataset made from MNIST data.
    Each of the tasks a classification task over the 10 digits, but the data from task i
    is rotated by an angle of i * alpha degrees, where alpha is a parameter of the
    dataset.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = False,
        num_tasks: int = 10,
        alpha: float = 18.0,
    ) -> None:
        """ Init function for MNIST. """

        super().__init__()

        # Check for valid arguments.
        assert alpha >= 0
        assert alpha * (num_tasks - 1) <= 180

        self.input_size = (1, 28, 28)
        self.output_size = 10
        self.loss_cls = nn.CrossEntropyLoss
        self.extra_metrics = {
            "accuracy": {"maximize": True, "train": True, "eval": True, "show": True},
        }

        self.root = root
        self.train = train
        self.download = download
        self.num_tasks = num_tasks
        self.alpha = alpha

        # Set current task.
        self.set_current_task(0)

    def __len__(self):
        """ Length of dataset. Overridden from parent class. """
        return len(self.current_mnist)

    def __getitem__(self, idx: int):
        """ Get item `idx` from dataset. Overridden from parent class. """
        return self.current_mnist[idx]

    def set_current_task(self, new_task: int) -> None:
        """ Set the current training task to `new_task`. """

        super().set_current_task(new_task)
        
        # Set angle of rotation for current task and create corresponding dataset
        # object.
        angle = self.alpha * self._current_task
        task_transform = transforms.Compose([
            RotationTransform(angle),
            GRAY_TRANSFORM,
        ])
        self.current_mnist = MNIST(
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


class RotationTransform:
    """
    Rotate an image tensor by a fixed angle. This is just a wrapper around TF.rotate().
    """

    def __init__(self, angle) -> None:
        """ Init function for RotationTransform. Given angle should be in degrees. """
        self.angle = angle

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """ Compute rotation on input `x`. """
        return TF.rotate(x, self.angle, fill=0)
