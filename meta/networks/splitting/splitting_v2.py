"""
Definition of MultiTaskSplittingNetworkV2, a splitting network where splitting decisions
are made simply by splitting the N regions with the largest task gradient distance every
K steps.
"""

from typing import Any

import torch

from meta.networks.splitting import BaseMultiTaskSplittingNetwork


class MultiTaskSplittingNetworkV2(BaseMultiTaskSplittingNetwork):
    """
    A splitting network where splitting decisions are made simply by splitting the N
    regions with the largest task gradient distance every K steps.
    """

    def __init__(self, split_freq: int, splits_per_step: int, **kwargs: Any) -> None:
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

        super(MultiTaskSplittingNetworkV2, self).__init__(**kwargs)

        # Set state.
        self.split_freq = split_freq
        self.splits_per_step = splits_per_step

    def determine_splits(self) -> torch.Tensor:
        """
        Determine which splits (if any) should occur checking whether the number of
        steps is a multiple of `split_freq`. If so, we split the `splits_per_step`
        regions with the largest task gradient distances. If there is a tie for the
        regions with the largest distance, we split all tying regions.

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
        if torch.all(self.grad_diff_stats.num_steps < self.split_step_threshold):
            return should_split
        if self.num_steps % self.split_freq != 0:
            return should_split

        # Get normalized distance scores. For each (task1, task2, region), we divide the
        # corresponding squared gradient distance by the region size, to normalize the
        # effect that squared gradient distance is linearly scaled by the region size.
        distance_scores = self.grad_diff_stats.mean / self.region_sizes

        # Set distance scores to zero for task/region pairs that aren't shared, have too
        # small sample size, or have task1 < task 2 (this way we avoid duplicate values
        # from (task1, task2) and (task2, task1).
        is_shared = self.splitting_map.shared_regions()
        distance_scores *= is_shared

        sufficient_sample = self.grad_diff_stats.num_steps >= self.split_step_threshold
        distance_scores *= sufficient_sample

        upper_triangle = torch.triu(
            torch.ones(self.num_tasks, self.num_tasks), diagonal=1
        )
        upper_triangle = upper_triangle.unsqueeze(-1).expand(-1, -1, self.num_regions)
        distance_scores *= upper_triangle

        # Filter out zero distance pairs and find regions with largest distance.
        flat_scores = distance_scores.view(-1)
        flat_scores = flat_scores[(flat_scores > 0).nonzero()].squeeze(-1)
        num_valid_scores = flat_scores.shape[0]
        if num_valid_scores > 0:
            num_splits = min(self.splits_per_step, num_valid_scores)
            top_values, _ = torch.topk(flat_scores, num_splits)
            score_threshold = top_values[-1]
            should_split = distance_scores >= score_threshold

        return should_split
