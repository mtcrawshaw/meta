"""
Definition of MultiTaskSplittingNetworkV1, a splitting network where splitting decisions
are made by running tests of statistical significance on the differences between task
gradients.
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


class MultiTaskSplittingNetworkV1(BaseMultiTaskSplittingNetwork):
    """
    A splitting network where splitting decisions are made by running tests of
    statistical significance on the differences between task gradients.
    """

    def __init__(
        self,
        split_alpha: float = 0.05,
        grad_var: float = None,
        log_z: bool = True,
        **kwargs
    ) -> None:
        """
        Init function for MultiTaskSplittingNetworkV1. `kwargs` should hold the
        arguments for the init function of BaseMultiTaskSplittingNetwork.

        Arguments
        ---------
        split_alpha : float
            Alpha value for statistical test when determining whether or not to split.
            If the null hypothesis is true, then we will perform a split
            `100*split_alpha` percent of the time.
        grad_var : float
            Estimate of variance of each component of task-specific gradients (where
            task-specific gradients are modeled as multi-variate Gaussians with diagonal
            covariance matrices). If set to None, this value is estimated
            online.
        log_z : bool
            Whether or not to log out values of the z-scores throughout training. We
            leave this as an option because it requires tracking z-score stats at every
            update step, which will slow down training. This is essentially a diagnostic
            tool to see why the network is/isn't splitting.
        """

        super(MultiTaskSplittingNetworkV1, self).__init__(**kwargs)

        # Set state.
        self.split_alpha = split_alpha
        self.grad_var = grad_var
        self.log_z = log_z

        # Initialize running estimate of gradient statistics, in order to measure the
        # variance of the gradient of each network weight. Note that we only measure
        # stats of individual gradients if `self.grad_var = None`, and otherwise we use
        # `self.grad_var` as an estimate of the standard deviation of gradient
        # components.
        if self.grad_var is None:
            self.grad_stats = RunningStats(
                shape=(self.num_tasks, self.total_region_size),
                compute_stdev=True,
                condense_dims=(1,),
                cap_sample_size=self.cap_sample_size,
                ema_alpha=self.ema_alpha,
                device=self.device,
            )

        # Compute critical value of z-statistic based on given value of `split_alpha`.
        self.critical_z = norm.ppf(1 - self.split_alpha)

    def update_grad_stats(self, task_grads: torch.Tensor) -> None:
        """
        Overridden from superclass. Update our running estimates of gradient statistics.
        We keep running estimates of the mean of the squared difference between task
        gradients at each region, and an estimate of the standard deviation of the
        gradient of each weight.
        """

        # Get indices of tasks with non-zero gradients. A task will have zero gradients
        # when the current batch doesn't contain any data from that task, and in that
        # case we do not want to update the gradient stats for this task.
        task_flags = (task_grads.view(self.num_tasks, -1) != 0.0).any(dim=1)
        task_pair_flags = task_flags.unsqueeze(0) * task_flags.unsqueeze(1)
        task_pair_flags = task_pair_flags.unsqueeze(-1)
        task_pair_flags = task_pair_flags.expand(-1, -1, self.num_regions)

        # Compute pairwise differences between task-specific gradients.
        task_grad_diffs = self.get_task_grad_diffs(task_grads)

        # Update our estimates of the mean pairwise distance between tasks and the
        # standard deviation of the gradient of each individual weight. Since
        # `task_grads` is a single tensor padded with zeros, we extract the non-pad
        # values before updating `self.grad_stats`. The non-pad values are extracted
        # into a tensor of shape `(self.num_tasks, self.total_region_size)`. Note that
        # we only need to estimate the standard deviation of the gradient of each weight
        # when `self.grad_var` is None.
        self.grad_diff_stats.update(task_grad_diffs, task_pair_flags)
        if self.grad_var is None:
            flattened_grad = torch.cat(
                [
                    task_grads[:, r, : self.region_sizes[r]]
                    for r in range(self.num_regions)
                ],
                dim=1,
            )
            self.grad_stats.update(flattened_grad, task_flags)

    def determine_splits(self) -> torch.Tensor:
        """
        Determine which splits (if any) should occur by computing the z-statistic for
        each pair of tasks at each region. If a z-score is large enough, then we will perform
        a split for the corresponding tasks/region.

        Returns
        -------
        should_split : torch.Tensor
            Tensor of size (self.num_tasks, self.num_tasks, self.num_regions), where
            `should_split[i, j, k]` holds 1/True if tasks `i, j` should be split at
            region `k`, and 0/False otherwise..
        """

        # Compute z-scores.
        z = self.get_split_statistics()

        # Determine splits. The regions that are split are those with z-scores above the
        # critical value and a sufficiently large sample size.
        should_split = z > self.critical_z
        should_split *= self.grad_diff_stats.num_steps >= self.split_step_threshold

        return should_split

    def get_split_statistics(self) -> torch.Tensor:
        """
        Compute the z-score of each region/task pair to determine whether or not splits
        should be performed. Intuitively, the magnitude of this value represents the
        difference in the distributions of task gradients between each pair of tasks at
        each region. 
        """

        # Get estimated variance of each weight's gradient.
        if self.grad_var is None:
            est_grad_var = torch.sum(
                self.grad_stats.var * self.grad_stats.sample_size
            ) / torch.sum(self.grad_stats.sample_size)
        else:
            est_grad_var = float(self.grad_var)

        # Compute population statistics from estimated gradient variance.
        mu = 2 * self.region_sizes * est_grad_var
        mu = mu.expand(self.num_tasks, self.num_tasks, -1)
        sigma = 2 * torch.sqrt(2 * self.region_sizes.to(dtype=torch.float32))
        sigma *= est_grad_var
        sigma = sigma.expand(self.num_tasks, self.num_tasks, -1)

        # Compute z-scores and log them out, if necessary.
        z = (
            torch.sqrt(self.grad_diff_stats.sample_size.to(dtype=torch.float32))
            * (self.grad_diff_stats.mean - mu)
            / sigma
        )
        if self.log_z:
            self.log_current_z(z)

        return z

    def log_current_z(self, z: torch.Tensor) -> None:
        """ Prints summary of z-scores at each region to log. """

        msg = "z-scores:\n"
        for region in range(self.num_regions):
            scores = []
            for task1 in range(self.num_tasks - 1):
                for task2 in range(task1 + 1, self.num_tasks):
                    copy1 = self.splitting_map.copy[region, task1]
                    copy2 = self.splitting_map.copy[region, task2]
                    if copy1 == copy2:
                        scores.append(float(z[task1, task2, region]))
            score_mean = None
            score_min = None
            score_max = None
            if len(scores) > 0:
                score_mean = np.mean(scores)
                score_min = min(scores)
                score_max = max(scores)
            msg += "Region %d mean, min, max: %r, %r, %r\n" % (
                region,
                score_mean,
                score_min,
                score_max,
            )
        msg += "\n"
        logger.log(msg)
