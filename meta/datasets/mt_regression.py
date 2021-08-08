"""
Multi-task regression dataset object (toy problem). This task is defined and motivated
in the GradNorm paper here: https://arxiv.org/abs/1711.02257.
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset


SCALES = {
    2: [1, 10],
    5: [1, 3, 5, 7, 9],
    **{num_tasks: list(range(1, num_tasks + 1)) for num_tasks in [10, 20, 30, 40, 50]},
}
DATASET_SIZE = 10000
TRAIN_SPLIT = 0.9
INPUT_DIM = 250
OUTPUT_DIM = 100
INPUT_STD = 0.01
BASE_STD = 10.0
TASK_STD = 3.5


class MTRegression(Dataset):
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

        # Save state.
        super().__init__()
        self.num_tasks = num_tasks
        self.root = os.path.join(root, f"MTRegression{self.num_tasks}")
        self.train = train
        self.scales = SCALES[self.num_tasks]

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
