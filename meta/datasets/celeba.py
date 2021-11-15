"""
Dataset object for CelebA dataset. This is a wrapper around the PyTorch CelebA Dataset
object, which additionally contains information about metrics, loss function, input and
output size, etc.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torchvision.datasets import CelebA as torch_CelebA

from meta.datasets import BaseDataset
from meta.datasets.utils import get_split, slice_second_dim, RGB_TRANSFORM
from meta.train.loss import MultiTaskLoss


TOTAL_TASKS = 40


class CelebA(torch_CelebA, BaseDataset):
    """ CelebA dataset wrapper. """

    def __init__(self, root: str, train: bool = True, num_tasks: int = 9) -> None:
        """ Init function for CelebA. """

        # Check for valid arguments.
        assert 1 <= num_tasks <= TOTAL_TASKS

        # Call parent constructors.
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

        # Store settings.
        self.num_tasks = num_tasks

        # Store static dataset properties.
        self.input_size = (3, 178, 218)
        self.output_size = [2] * self.num_tasks
        self.loss_cls = MultiTaskLoss
        self.loss_kwargs = {
            "task_losses": [
                {
                    "loss": nn.CrossEntropyLoss(reduction="mean"),
                    "output_slice": slice_second_dim(t),
                    "label_slice": slice_second_dim(t),
                }
                for t in range(self.num_tasks)
            ]
        }
        self.extra_metrics = {
            "avg_accuracy": {"maximize": True, "train": True, "eval": True, "show": True},
            **{
                f"task_{t}_accuracy": {
                    "maximize": True,
                    "train": True,
                    "eval": True,
                    "show": False,
                }
                for t in range(self.num_tasks)
            }
        }

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Get image and labels for element with index `idx`. """
        image, label = torch_CelebA.__getitem__(self, idx)
        task_labels = label[:self.num_tasks]
        return image, task_labels

    def compute_metrics(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module = None,
        train: bool = True,
    ) -> Dict[str, float]:
        """
        Compute classification accuracy for each task from `outputs` and `labels`. Note
        that `outputs` has shape `(batch_size, self.num_tasks, 2)` and `labels` has
        shape `(batch_size, self.num_tasks)`.
        """

        # Compute accuracy for each task.
        task_accs = (outputs.argmax(dim=-1) == labels).sum(dim=-1) / labels.shape[0]
        avg_acc = torch.mean(task_accs)

        # Store in expected format.
        split = get_split(train)
        metrics = {
            f"{split}_task_{t}_accuracy": float(task_accs[t])
            for t in range(self.num_tasks)
        }
        metrics[f"{split}_avg_accuracy"] = float(avg_acc)

        return metrics
