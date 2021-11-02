"""
Abstract class for datasets for continual learning. Defines attributes and functions
that any dataset used for continual learning should have.
"""

import torch

from meta.datasets.base import BaseDataset


class ContinualDataset(BaseDataset):
    """ Abstract class for continual learning datasets. """

    def __init__(self) -> None:
        """
        Init function for ContinualDatset. Initializes all required members, but to null
        values. Any subclass of ContinualDataset should initialize these values AFTER
        calling ContinualDataset.__init__(). This function mostly just exists to
        document the required members that must be populated by any subclass.
        """

        super().__init__()

        self._current_task = None
        self.num_tasks = None

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> torch.Tensor:
        """ Return the item with index `idx` from the dataset for the current task. """
        raise NotImplementedError

    def set_current_task(self, new_task: int) -> None:
        """ Set the current training task to `new_task`. """
        self._current_task = new_task

    def advance_task(self) -> None:
        """ Wrapper for convenience. """
        self.set_current_task(self._current_task + 1)
