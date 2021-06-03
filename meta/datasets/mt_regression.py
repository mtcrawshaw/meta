""" Multi-task regression dataset object (toy problem). """

import torch
import numpy as np

from torch.utils.data import Dataset


class MTRegression(Dataset):
    """ PyTorch wrapper for the multi-task regression toy dataset. """

    def __init__(self, root: str, train: bool = True):
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
        self.train = train

        # Load dataset files.
        if self.train:
            self.inputs = np.load(os.path.join("root", "train_input.npy"))
            self.labels = np.load(os.path.join("root", "train_output.npy"))
        else:
            self.inputs = np.load(os.path.join("root", "test_input.npy"))
            self.labels = np.load(os.path.join("root", "test_output.npy"))

        # Check for consistent sizes.
        assert self.inputs.shape[0] == self.labels.shape[0]
        self.dataset_size = self.inputs.shape[0]

    def __getitem__(self, index: int):

        inp = self.input_arr[index]
        labels = self.label_arr[index]
        return inp, labels

    def __len__(self):
        return self.dataset_size

    def __repr__(self):
        rep = "Dataset MTRegression\n"
        rep += "    Number of data points: %d\n" % self.dataset_size
        rep += f"    Split: %s\n" % "train" if self.train else "test"
        rep += f"    Root Location: %s\n"
        return fmt_str
