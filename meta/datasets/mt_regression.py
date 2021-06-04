""" Multi-task regression dataset object (toy problem). """

import os

import numpy as np
import torch
from torch.utils.data import Dataset


class MTRegression(Dataset):
    """ PyTorch wrapper for the multi-task regression toy dataset. """

    def __init__(self, root: str, num_tasks: int, train: bool = True):
        """
        Init function for MTRegression.

        Parameters
        ----------
        root : str
            Path to folder containing dataset files.
        train : bool
            Whether to load training set. Otherwise, test set is loaded.
        """

        super().__init__()
        self.root = root
        self.num_tasks = num_tasks
        self.train = train

        # Load dataset files.
        if self.train:
            self.inputs = np.load(os.path.join(self.root, "train_input.npy"))
            self.labels = np.load(os.path.join(self.root, "train_output.npy"))
        else:
            self.inputs = np.load(os.path.join(self.root, "test_input.npy"))
            self.labels = np.load(os.path.join(self.root, "test_output.npy"))

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
