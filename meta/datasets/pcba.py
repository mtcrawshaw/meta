"""
Dataset object for PubChem BioAssay dataset. The data is given in the DeepChem package
(https://github.com/deepchem/deepchem), and the details can be found here:
https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html?highlight=PCBA#pcba-datasets.
"""

import os
from typing import Tuple

import numpy as np
import pandas
import torch
from torch.utils.data import Dataset


TOTAL_TASKS = 128
DOWNLOAD_URL = "https://github.com/deepchem/deepchem/raw/master/datasets/pcba.csv.gz"
DATA_DIRNAME = "PCBA"
RAW_DATA_FNAME = "pcba.csv.gz"
TRAIN_SPLIT = None


class PCBA(Dataset):
    """ PyTorch wrapper for the PCBA dataset. """

    def __init__(self, root: str, num_tasks: int, train: bool = True):
        """
        Init function for PCBA.

        Parameters
        ----------
        root : str
            Path to folder containing dataset files.
        num_tasks : int
            Number of tasks for instance of PCBA. Should be between 1 and TOTAL_TASKS.
            For each input molecule, only the labels from the first `num_tasks` tasks
            are loaded.
        train : bool
            Whether to load training set. Otherwise, test set is loaded.
        """

        # Check that `num_tasks` is valid.
        assert 1 <= num_tasks <= TOTAL_TASKS

        # Save state.
        super().__init__()
        self.num_tasks = num_tasks
        self.root = os.path.join(root, DATA_DIRNAME)
        self.raw_data_path = os.path.join(root, RAW_DATA_FNAME)
        self.train = train
        self.split = "train" if self.train else "test"

        # Preprocess data if necessary.
        if not os.path.isdir(self.root):
            if not os.path.isdir(self.raw_data_path):
                raise ValueError(
                    f"Neither raw data nor processed dataset exist in folder"
                    f" '{self.root}'. Download the raw data from:\n  {DOWNLOAD_URL}\n"
                    " and place it in `{self.root}'."
                )
            self.preprocess()

        # Load data.
        input_path = os.path.join(self.root, self.data_fname(self.train, inp=True)
        output_path = os.path.join(self.root, self.data_fname(self.train, inp=False)
        self.inputs = np.load(input_path)
        self.outputs = np.load(output_path)
        assert len(self.inputs) == len(self.outputs)
        self.dataset_size = len(self.inputs)

    def __len__(self):
        """
        Number of data points in dataset. For PCBA, this is the total number of input
        molecules. However, most input molecules don't contain labels for many of the
        targets (tasks).
        """
        return self.dataset_size

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """ Return dataset item with input `index`. """
        return self.inputs[index], self.labels[index]

    def preprocess(self) -> None:
        """
        Preprocesses the raw PCBA data from the DeepChem repository:
        - Uncompresses the compressed CSV file
        - Featurizes input molecules with ?
        - Fills in missing label values with -1
        - Randomly splits into a training and testing set
        - Saves results as numpy arrays
        """

        # Uncompress the raw PCBA data.
        with gzip.open(self.raw_data_path, "rb") as csv_file:
            dataframe = pandas.read_csv(csv_file)

        print(dataframe)
        exit()

    def data_fname(self, train: bool, inp: bool) -> str:
        """ Names for dataset files. """
        split = "train" if train else "test"
        name = "input" if inp else "label"
        return f"{split}_{name}.npy"
