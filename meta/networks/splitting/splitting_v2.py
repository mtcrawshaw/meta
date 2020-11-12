"""
Definition of MultiTaskSplittingNetworkV2, a splitting network where splitting decisions
are made simply by splitting the region with the largest task gradient distance every K
steps.
"""

import math
from itertools import product
from copy import deepcopy
from typing import Callable, List, Tuple

import numpy as np
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.nn.functional as F

from meta.networks.splitting import BaseMultiTaskSplittingNetwork
from meta.utils.estimate import RunningStats
from meta.utils.logger import logger


class MultiTaskSplittingNetworkV2(BaseMultiTaskSplittingNetwork):
    """
    A splitting network where splitting decisions are made simply by splitting the
    region with the largest task gradient distance every K steps.
    """

    def __init__(self, split_freq: int, splits_per_step: int, **kwargs) -> None:
        """
        Init function for MultiTaskSplittingNetworkV2. `kwargs` should hold the
        arguments for the init function of BaseMultiTaskSplittingNetwork.

        Arguments
        ---------
        split_freq : int
            Number of steps between batches of splits. This means at every `split_freq`
            training steps, a batch of splits will be executed.
        splits_per_step : int
            Number of splits to perform at each batch of splits.
        """

        super(MultiTaskSplittingNetworkV1, self).__init__(**kwargs)

        # Set state.
        self.split_freq = split_freq
        self.splits_per_step = splits_per_step

    def determine_splits(self) -> torch.Tensor:
        """
        Determine which splits (if any) should occur checking whether the number of
        steps is a multiple of `split_freq`. If so, we split the `splits_per_step`
        regions with the largest task gradient distances.

        Returns
        -------
        should_split : torch.Tensor
            Tensor of size (self.num_tasks, self.num_tasks, self.num_regions), where
            `should_split[i, j, k]` holds 1/True if tasks `i, j` should be split at
            region `k`, and 0/False otherwise..
        """

        should_split = torch.zeros(
            self.num_tasks, self.num_tasks, self.num_regions, device=self.device
        )

        # Don't perform splits if the number of steps is less than the minimum or if the
        # current step doesn't fall on a multiple of the splitting frequency.
        if self.num_steps <= self.split_step_threshold:
            return should_split
        if self.num_steps % self.split_freq != 0:
            return should_split

        # Only split the `splits_per_step` regions with largest task gradient distances.
        # Note that we only want to consider (task1, task2, region) when task1 and task2
        # are shared at region.
        is_shared = self.splitting_map.shared_regions()
        distance_scores = self.grad_diff_stats.mean * is_shared
        top_values, _ = torch.topk(distance_scores.view(-1), self.splits_per_step)
        score_threshold = top_values[0]
        should_split = distance_score >= score_threshold

        return should_split
