"""
Abstract class for datasets. Defines attributes and functions that any dataset used for
supervised learning should have. Note: When implementing a subclass of this class that
also inherits from another object, you'll have a much easier time by including this
object second in the method resolution order. For example, a subclass defined like so:

    class Child(Parent, BaseDataset)

will be a lot easier to work with than:

    class Child(BaseDataset, Parent)

Since this class doesn't have an __init__ function, you can simple call

    super(Child, self).__init__()

in the __init__ function of Child, and it will run the __init__ function from Parent.
"""

from typing import Dict

import torch
import torch.nn as nn


class BaseDataset:
    """ Abstract class for datasets. """

    input_size = None
    output_size = None
    loss_cls = None
    loss_kwargs = {}
    criterion_kwargs = {"train": {}, "eval": {}}
    extra_metrics = {}

    @staticmethod
    def compute_metrics(
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
