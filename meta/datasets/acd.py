"""
Dataset object for continual learning on a simple adverarially constructed dataset.
"""

import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from meta.datasets import ContinualDataset


DATASET_SIZE = 10000
TRAIN_SPLIT = 0.9
INPUT_DIM = 250
OUTPUT_DIM = 100
MU = 2
SIGMA = 0.01
BASE_STD = 10.0
TOTAL_TASKS = 10


class ACD(ContinualDataset):
    """
    Alternating Continual Dataset. This is a continual learning dataset made from data
    in which the input distribution for each task alternates over consecutive tasks.
    Each input for task i is a real-valued vector, drawn from N(\mu (-1)^i, \sigma),
    where \mu and \sigma are hand-coded constants. This dataset is meant to demonstrate
    the weaknesses of batch normalization for continual learning. When the data
    distribution alternates with each consecutive tasks, the batchnorm parameters should
    oscillate back and forth, without learning anything that is applicable to all tasks.
    """

    def __init__(
        self, root: str, num_tasks: int, train: bool = True, flip_target: bool = False,
    ) -> None:
        """ Init function for Rotated. """

        # Check for valid arguments.
        assert num_tasks <= TOTAL_TASKS

        super().__init__()

        # Store data settings.
        self.root = root
        self.num_tasks = num_tasks
        self.train = train
        self.flip_target = flip_target

        # Set static dataset properties.
        self.input_size = INPUT_DIM
        self.output_size = OUTPUT_DIM
        self.loss_cls = nn.MSELoss
        self.extra_metrics = {}

        # Load dataset files, or create them if they don't yet exist.
        self.load_or_create()

        # Set current task.
        self.set_current_task(0)

        # Check for consistent data sizes.
        assert self.inputs.shape[0] == self.labels.shape[0]

    def __len__(self):
        """ Length of dataset. Overridden from parent class. """
        return self.inputs.shape[0]

    def __getitem__(self, idx: int):
        """ Get item `idx` from dataset. Overridden from parent class. """
        inp = self.inputs[idx, self._current_task]
        labels = self.labels[idx, self._current_task]
        return inp, labels

    def load_or_create(self) -> None:
        """ Load dataset from files if they exist, otherwise generate dataset. """

        # Check in input/label files already exist.
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

            print(f"Files for dataset ACD do not exist. Generating now.")
            if not os.path.isdir(self.root):
                os.makedirs(self.root)

            # Generate input-output pairs for training and testing.
            base_transform = np.random.normal(
                loc=0.0, scale=BASE_STD, size=(OUTPUT_DIM, INPUT_DIM)
            )
            sizes = {
                "train": round(TRAIN_SPLIT * DATASET_SIZE),
                "test": DATASET_SIZE - round(TRAIN_SPLIT * DATASET_SIZE),
            }
            for split, split_size in sizes.items():

                assert split_size > 0
                split_inputs = np.zeros((split_size, TOTAL_TASKS, INPUT_DIM))
                split_outputs = np.zeros((split_size, TOTAL_TASKS, OUTPUT_DIM))
                for task in range(TOTAL_TASKS):

                    # Generate inputs for task.
                    task_mean = MU * (-1 if task % 2 == 1 else 1)
                    split_task_inputs = np.random.normal(
                        loc=task_mean, scale=SIGMA, size=(split_size, INPUT_DIM)
                    )
                    split_inputs[:, task] = np.copy(split_task_inputs)

                    # Generate outputs for task.
                    split_task_outputs = np.matmul(
                        base_transform, np.transpose(split_task_inputs)
                    )
                    split_task_outputs = np.tanh(split_task_outputs)
                    split_outputs[:, task] = np.copy(np.transpose(split_task_outputs))

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

        # Flip targets for odd tasks, if necessary.
        if self.flip_target:
            for task in range(1, TOTAL_TASKS, 2):
                self.labels[:, task] *= -1

    def compute_metrics(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module = None,
        train: bool = True,
    ) -> Dict[str, float]:
        """ Compute classification accuracy from `outputs` and `labels`. """
        return {}
