"""
Dataset object for PubChem BioAssay dataset. The data is given in the DeepChem package
(https://github.com/deepchem/deepchem), and the details can be found here:
https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html?highlight=PCBA#pcba-datasets.
"""

import os
import random
import json
import gzip
from typing import Tuple, Optional

import numpy as np
import pandas
import torch
import deepchem as dc
from torch.utils.data import Dataset


TOTAL_TASKS = 128
DOWNLOAD_URL = "https://github.com/deepchem/deepchem/raw/master/datasets/pcba.csv.gz"
RAW_DATA_FNAME = "pcba.csv.gz"
DATASET_CONFIG = {
    "feature_size": 2048,
    "train_split": 0.9,
    "ecfp_radius": 4,
}
CLASS_SAMPLES = np.array([30269634, 427685])
CLASS_WEIGHTS = 1.0 - CLASS_SAMPLES / np.sum(CLASS_SAMPLES)


class PCBA(Dataset):
    """ PyTorch wrapper for the PCBA dataset. """

    def __init__(
        self,
        root: str,
        num_tasks: int,
        data_tasks: Optional[int] = None,
        train: bool = True,
    ):
        """
        Init function for PCBA.

        Parameters
        ----------
        root : str
            Path to folder containing dataset files.
        num_tasks : int
            Number of tasks for instance of PCBA. Should be between 1 and `TOTAL_TASKS`.
            For each input molecule, only the labels from the first `num_tasks` tasks
            are loaded.
        data_tasks : Optional[int]
            Only use data points that are labeled for at least one of the first
            `data_tasks` tasks. This can be used to ensure that a comparison of training
            on e.g. 128 tasks vs. 32 tasks is using the same data, by setting
            `data_tasks = 32`. If None, this will be set to `num_tasks`.
        train : bool
            Whether to load training set. Otherwise, test set is loaded.
        """

        # Check that parameter values are valid.
        assert 1 <= num_tasks <= TOTAL_TASKS
        assert data_tasks is None or 1 <= data_tasks <= TOTAL_TASKS

        # Save state.
        super().__init__()
        self.num_tasks = num_tasks
        self.data_tasks = data_tasks if data_tasks is not None else self.num_tasks
        self.root = root
        self.raw_data_path = os.path.join(os.path.dirname(root), RAW_DATA_FNAME)
        self.train = train
        self.split = "train" if self.train else "test"
        self.feature_size = DATASET_CONFIG["feature_size"]

        # Preprocess data if necessary.
        if not os.path.isdir(self.root):
            if not os.path.isfile(self.raw_data_path):
                raise ValueError(
                    f"Neither raw data nor processed dataset exist in folder"
                    f" '{root}'. Download the raw data from:\n  {DOWNLOAD_URL}\n"
                    f"and place it in '{root}'."
                )
            self.preprocess()

        # Load data.
        self.inputs = np.load(self.data_path(train=self.train, inp=True))
        self.labels = np.load(self.data_path(train=self.train, inp=False))
        assert len(self.inputs) == len(self.labels)
        original_dataset_size = len(self.inputs)

        # Remove datapoints that aren't labeled for any of the chosen subset of tasks.
        good_idxs = np.any(self.labels[:, : self.data_tasks] != -1, axis=1).nonzero()[0]
        self.inputs = self.inputs[good_idxs]
        self.labels = self.labels[good_idxs]
        self.dataset_size = len(self.inputs)
        if self.dataset_size != original_dataset_size:
            removed = original_dataset_size - self.dataset_size
            print(
                f"Removing {removed} datapoints from PCBA {self.split} that aren't"
                f" labeled for first {self.data_tasks} tasks. {self.dataset_size}"
                f" {self.split} datapoints remaining."
            )

        # Remove labels from tasks not being trained on.
        self.labels = self.labels[:, : self.num_tasks]

        # Load dataset config to ensure that the config of loaded data matches the
        # current config.
        config_path = self.config_path()
        with open(config_path, "r") as config_file:
            loaded_config = json.load(config_file)
        if DATASET_CONFIG != loaded_config:
            raise ValueError(
                f"Config of loaded PCBA dataset ({config_path}) doesn't match current"
                f" PCBA config (hard-coded in {os.path.relpath(__file__)})."
                " To run training, you must either:\n"
                " (1) Change the current PCBA config to match the loaded config.\n"
                " (2) Delete the processed PCBA data to regenerate it with new config."
            )

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
        - Uncompress the raw CSV file
        - Fill in missing label values with -1
        - Featurize input molecules with Extended Connectivity Circular Fingerprints
        - Randomly split into a training and testing set
        - Save inputs and labels to disk as numpy arrays
        """

        print(
            "Processing raw PCBA data."
            " This only needs to be done once, but will take 5-10 minutes."
        )

        # Uncompress the raw PCBA data.
        with gzip.open(self.raw_data_path, "rb") as csv_file:
            dataframe = pandas.read_csv(csv_file)

        # Fill in missing label values.
        dataframe.fillna(value=-1, inplace=True)

        # Featurize input molecules and check that they were computed without errors.
        featurizer = dc.feat.CircularFingerprint(
            size=self.feature_size, radius=DATASET_CONFIG["ecfp_radius"]
        )
        features = featurizer.featurize(dataframe["smiles"])
        assert np.logical_or(features == 0, features == 1).all()

        # Drop unneeded columns.
        dataframe.drop(labels=["mol_id", "smiles"], axis=1, inplace=True)

        # Randomly split into training and testing set.
        dataset_size = len(dataframe)
        train_size = round(dataset_size * DATASET_CONFIG["train_split"])
        assert 0 < train_size < dataset_size
        idxs = list(range(dataset_size))
        random.shuffle(idxs)
        train_idxs = idxs[:train_size]
        test_idxs = idxs[train_size:]
        train_input = features[train_idxs].astype(dtype=np.float32)
        test_input = features[test_idxs].astype(dtype=np.float32)
        train_label = dataframe.iloc[train_idxs].to_numpy(dtype=np.float32)
        test_label = dataframe.iloc[test_idxs].to_numpy(dtype=np.float32)

        # Save results as numpy arrays.
        os.makedirs(self.root)
        np.save(self.data_path(train=True, inp=True), train_input)
        np.save(self.data_path(train=True, inp=False), train_label)
        np.save(self.data_path(train=False, inp=True), test_input)
        np.save(self.data_path(train=False, inp=False), test_label)
        print(f"train input shape: {train_input.shape}")
        print(f"train label shape: {train_label.shape}")
        print(f"test input shape: {test_input.shape}")
        print(f"test label shape: {test_label.shape}")

        # Save out dataset config.
        with open(self.config_path(), "w") as config_file:
            json.dump(DATASET_CONFIG, config_file, indent=4)

    def data_path(self, train: bool, inp: bool) -> str:
        """ Names for dataset files. """
        split = "train" if train else "test"
        name = "input" if inp else "label"
        return os.path.join(self.root, f"{split}_{name}.npy")

    def config_path(self) -> str:
        return os.path.join(self.root, "dataset_config.json")