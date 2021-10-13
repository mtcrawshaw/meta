"""
Abstract class for datasets. Defines attributes and functions that any dataset used for
supervised learning should have.
"""

from typing import Dict

import torch
import torch.nn as nn


class BaseDataset:
    """ Abstract class for datasets. """

    def __init__(self) -> None:
        """
        Init function for BaseDatset. Initializes all required members, but to null
        values. Any subclass of BaseDataset should initialize these values AFTER calling
        BaseDataset.__init__(). This function mostly just exists to document the
        required members that must be populated by any subclass.
        """
        self.input_size = None
        self.output_size = None
        self.loss_cls = None
        self.loss_kwargs = {}
        self.criterion_kwargs = {"train": {}, "eval": {}}
        self.extra_metrics = {}

    def compute_metrics(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module = None,
        train: bool = True,
    ) -> Dict[str, float]:
        """
        Compute metrics from `outputs` and `labels`, returning a dictionary whose keys
        are metric names and whose values are floats. Note that the returned metrics
        from the subclass definition should match the metrics listed in `extra_metrics`. 
        """
        raise NotImplementedError
