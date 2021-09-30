"""
Multi-task regression dataset object (toy problem). This task is defined and motivated
in the GradNorm paper here: https://arxiv.org/abs/1711.02257.
"""

import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from meta.datasets import BaseDataset
from meta.datasets.utils import get_split, slice_second_dim
from meta.train.loss import MultiTaskLoss


SCALES = {
    2: [1, 10],
    **{num_tasks: list(range(1, num_tasks + 1)) for num_tasks in [10, 20, 30, 40, 50]},
}
DATASET_SIZE = 10000
TRAIN_SPLIT = 0.9
INPUT_DIM = 250
OUTPUT_DIM = 100
INPUT_STD = 0.01
BASE_STD = 10.0
TASK_STD = 3.5


class MTRegression(Dataset, BaseDataset):
    """ PyTorch wrapper for the multi-task regression toy dataset. """

    def __init__(self, root: str, num_tasks: int, train: bool = True):
        """
        Init function for MTRegression.

        Parameters
        ----------
        root : str
            Path to folder containing dataset files.
        num_tasks : int
            Number of regression tasks in dataset. Value should be a key in `SCALES`.
        train : bool
            Whether to load training set. Otherwise, test set is loaded.
        """

        # Check that `num_tasks` is valid.
        assert num_tasks in SCALES

        Dataset.__init__(self)
        BaseDataset.__init__(self)

        # Store data settings.
        self.num_tasks = num_tasks
        self.root = os.path.join(root, f"MTRegression{self.num_tasks}")
        self.train = train
        self.scales = SCALES[self.num_tasks]

        # Set static dataset properties.
        self.input_size = INPUT_DIM
        self.output_size = OUTPUT_DIM
        self.loss_cls = MultiTaskLoss
        self.loss_kwargs = {
            "task_losses": [
                {
                    "loss": nn.MSELoss(),
                    "output_slice": slice_second_dim(t),
                    "label_slice": slice_second_dim(t),
                }
                for t in range(self.num_tasks)
            ]
        }
        self.criterion_kwargs = {"train": {"train": True}, "eval": {"train": False}}
        self.extra_metrics = {
            "normal_loss": {"maximize": False, "train": True, "eval": True, "show": True},
            "var_normal_loss": {"maximize": None, "train": True, "eval": True, "show": False},
            "loss_weight_error": {"maximize": False, "train": True, "eval": False, "show": False},
            **{
                f"loss_weight_{t}": {"maximize": None, "train": True, "eval": False, "show": False}
                for t in range(self.num_tasks)
            },
        }

        # Load dataset files, or create them if they don't yet exist.
        self.load_or_create()

        # Check for consistent sizes and valid number of tasks.
        assert self.inputs.shape[0] == self.labels.shape[0]
        assert self.num_tasks <= self.labels.shape[1]
        self.dataset_size = self.inputs.shape[0]

    def __getitem__(self, index: int):

        inp = self.inputs[index]
        labels = self.labels[index, : self.num_tasks]
        return inp, labels

    def __len__(self):
        return self.dataset_size

    def __repr__(self):
        rep = "Dataset MTRegression\n"
        rep += "    Number of data points: %d\n" % self.dataset_size
        rep += f"    Split: %s\n" % "train" if self.train else "test"
        rep += f"    Root Location: %s\n"
        return fmt_str

    def load_or_create(self) -> None:
        """ Load dataset from files if they exist, otherwise generate dataset. """

        # Check if input/label files already exist.
        train_input_path = os.path.join(self.root, f"train_input.npy")
        train_label_path = os.path.join(self.root, f"train_label.npy")
        test_input_path = os.path.join(self.root, f"test_input.npy")
        test_label_path = os.path.join(self.root, f"test_label.npy")
        train_input_exists = os.path.isfile(train_input_path)
        train_label_exists = os.path.isfile(train_label_path)
        test_input_exists = os.path.isfile(test_input_path)
        test_label_exists = os.path.isfile(test_label_path)
        exists_list = [
            train_input_exists,
            train_label_exists,
            test_input_exists,
            test_label_exists,
        ]
        all_exist = all(exists_list)
        any_exist = any(exists_list)
        if all_exist != any_exist:
            raise ValueError(
                "Some (but not all) of the MTRegression dataset files exist:\n"
                f"  {train_input_path}\n"
                f"  {train_label_path}\n"
                f"  {test_input_path}\n"
                f"  {test_label_path}\n"
                "Delete the existing files so that all files can be re-generated together."
            )
        exists = all_exist

        # Generate dataset files if they don't exist.
        if not exists:

            print(
                f"Files for dataset MTRegression{self.num_tasks} do not exist."
                " Generating now."
            )
            if not os.path.isdir(self.root):
                os.makedirs(self.root)

            # Generate B and e_i matrices which define the input-output mapping.
            base_transform = np.random.normal(
                loc=0.0, scale=BASE_STD, size=(OUTPUT_DIM, INPUT_DIM)
            )
            task_transforms = [
                np.random.normal(loc=0.0, scale=TASK_STD, size=(OUTPUT_DIM, INPUT_DIM))
                for _ in range(self.num_tasks)
            ]

            # Generate input-output pairs for training and testing.
            sizes = {}
            sizes["train"] = round(TRAIN_SPLIT * DATASET_SIZE)
            sizes["test"] = DATASET_SIZE - sizes["train"]
            for split, split_size in sizes.items():

                # Generate inputs.
                split_inputs = np.random.normal(
                    loc=0.0, scale=INPUT_STD, size=(split_size, INPUT_DIM)
                )

                # Generate outputs.
                split_outputs = np.zeros((split_size, self.num_tasks, OUTPUT_DIM))
                for task in range(self.num_tasks):
                    task_outputs = np.matmul(
                        (base_transform + task_transforms[task]),
                        np.transpose(split_inputs),
                    )
                    task_outputs = self.scales[task] * np.tanh(task_outputs)
                    split_outputs[:, task] = np.copy(np.transpose(task_outputs))

                # Save inputs and outputs for split.
                input_path = train_input_path if split == "train" else test_input_path
                label_path = train_label_path if split == "train" else test_label_path
                np.save(input_path, split_inputs.astype(np.float32))
                np.save(label_path, split_outputs.astype(np.float32))

        # Load dataset from files.
        input_path = train_input_path if self.train else test_input_path
        label_path = train_label_path if self.train else test_label_path
        self.inputs = np.load(input_path)
        self.labels = np.load(label_path)

    def compute_metrics(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module = None,
        train: bool = True,
    ) -> Dict[str, float]:
        """ Compute training/testing metrics from `outputs` and `labels`. """

        split = get_split(train)

        # Compute average and variance of normalized task losses.
        normal_losses = self.get_normal_loss(outputs, labels)
        metrics = {
            f"{split}_normal_loss": np.mean(normal_losses),
            f"{split}_var_normal_loss": np.var(normal_losses),
        }

        # Compute loss weight error and add loss weights, if this is a training step.
        if train:
            loss_weights = criterion.loss_weighter.loss_weights
            metrics[f"{split}_loss_weight_error"] = self.get_loss_weight_error(loss_weights)
            metrics.update(
                {f"{split}_loss_weight_{t}": float(loss_weights[t]) for t in range(self.num_tasks)}
            )

        return metrics

    def get_normal_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
        """
        Computes normalized multi-task losses for MTRegression task. Both `outputs` and
        `labels` should have shape `(batch_size, num_tasks, output_dim)`. Returns a tensor
        holding the normalized loss for each task.
        """
        scales = np.array(SCALES[self.num_tasks])
        diffs = torch.sum((outputs - labels) ** 2, dim=2).detach()
        diffs = torch.mean(diffs, dim=0)
        diffs = diffs if diffs.device == torch.device("cpu") else diffs.cpu()
        diffs = diffs.numpy()
        weighted_diffs = diffs / (scales ** 2)
        return weighted_diffs

    def get_loss_weight_error(self, loss_weights: torch.Tensor) -> float:
        """
        Computes the mean square error between `loss_weights` and the ideal loss weights for
        the MTRegression task.
        """
        scales = np.array(SCALES[self.num_tasks])
        ideal_weights = 1.0 / (scales ** 2)
        ideal_weights *= self.num_tasks / np.sum(ideal_weights)
        current_weights = loss_weights.detach().cpu().numpy()
        normalized_weights = self.num_tasks * current_weights / np.sum(current_weights)
        error = np.mean((normalized_weights - ideal_weights) ** 2)
        return error
