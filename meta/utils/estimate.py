"""
Utility classes to compute running mean and standard deviation of torch.Tensors.
"""

from typing import Tuple

import torch


alpha_to_threshold = (
    lambda alpha: round(1.0 / (1.0 - alpha)) if alpha != 1.0 else float("inf")
)


class RunningStats:
    """
    Utility class to compute running estimates of mean/stdev of torch.Tensor.
    """

    def __init__(
        self,
        compute_stdev: bool = False,
        shape: Tuple[int, ...] = None,
        condense_dims: Tuple[int, ...] = (),
        cap_sample_size: bool = False,
        ema_alpha: float = 0.999,
        device: torch.device = None,
    ) -> None:
        """
        Init function for RunningStats. The running mean will always be computed, and a
        running standard deviation is also computed if `compute_stdev = True`.

        Arguments
        ---------
        compute_stdev : bool
            Whether or not to compute a standard deviation along with a mean.
        shape : Tuple[int, ...]
            The shape of the tensors that we will be computing stats over.
        condense_dims : Tuple[int, ...]
            The indices of dimensions to condense. For example, if `shape=(2,3)` and
            `condense=(1,)`, then a tensor `val` with shape `(2, 3)` will be treated as
            3 samples of a random variable with shape `(2,)`.
        cap_sample_size : bool
            Whether or not to stop increasing the sample size when we switch to EMA.
            This may be helpful because an EMA weights recent samples more than older
            samples, which can increase variance. To offset this, we can leave the
            sample size at a fixed value so that the sample size reflects the level of
            variance.
        ema_alpha : float
            Coefficient used to compute exponential moving average. We compute an
            arithmetic mean for the first `ema_threshold` steps (as computed below),
            then switch to EMA. If `ema_alpha == 1.0`, then we will never switch to EMA.
        """

        self.compute_stdev = compute_stdev
        self.condense_dims = condense_dims
        self.cap_sample_size = cap_sample_size
        self.ema_alpha = ema_alpha
        self.ema_threshold = alpha_to_threshold(ema_alpha)
        self.device = device if device is not None else torch.device("cpu")

        self.shape = shape
        self.condensed_shape = tuple(
            [shape[i] for i in range(len(shape)) if i not in condense_dims]
        )
        self.mean = torch.zeros(self.condensed_shape, device=self.device)
        if self.compute_stdev:
            self.square_mean = torch.zeros(self.condensed_shape, device=self.device)
            self.var = torch.zeros(self.condensed_shape, device=self.device)
            self.stdev = torch.zeros(self.condensed_shape, device=self.device)

        # Used to keep track of number of updates and effective sample size, which may
        # stop decreasing when we switch to using exponential moving averages.
        self.num_steps = torch.zeros(self.condensed_shape, device=self.device)
        self.sample_size = torch.zeros(self.condensed_shape, device=self.device)

    def update(self, val: torch.Tensor, flags: torch.Tensor = None) -> None:
        """
        Update running stats with new value.

        Arguments
        ---------
        val : torch.Tensor
            Tensor with shape `self.shape` representing a new sample to update running
            statistics.
        flags : torch.Tensor
            Tensor with shape `self.condensed_shape` representing whether or not to
            update the stats at each element of the stats tensors (0/False for don't
            update and 1/True for update). This allows us to only update a subset of the
            means/stdevs in the case that we receive a sample for some of the elements,
            but not all of them.
        """

        if flags is None:
            flags = torch.ones(self.condensed_shape, device=self.device)

        # Update `self.num_steps` and `self.sample_size`.
        self.num_steps += flags
        if self.cap_sample_size:
            below = self.sample_size + flags < self.ema_threshold
            above = torch.logical_not(below)
            self.sample_size = (
                self.sample_size + flags
            ) * below + self.ema_threshold * above
        else:
            self.sample_size += flags

        # Condense dimensions of sample if necessary.
        if len(self.condense_dims) > 0:
            new_val = torch.mean(val, dim=self.condense_dims)
            if self.compute_stdev:
                new_square_val = torch.mean(val ** 2, dim=self.condense_dims)
        else:
            new_val = val
            if self.compute_stdev:
                new_square_val = val ** 2

        # Update stats.
        self.mean = self.single_update(self.mean, new_val, flags)
        if self.compute_stdev:
            self.square_mean = self.single_update(
                self.square_mean, new_square_val, flags
            )
            self.var = self.square_mean - self.mean ** 2
            self.stdev = torch.sqrt(self.var)

    def single_update(
        self, m: torch.Tensor, v: torch.Tensor, flags: torch.Tensor
    ) -> torch.Tensor:
        """
        Update a mean, either through computing the arithmetic mean or an exponential
        moving average.
        """

        below = self.num_steps <= self.ema_threshold
        above = torch.logical_not(below)
        if torch.all(below):
            new_m = (m * (self.num_steps - 1) + v) / self.num_steps
        elif torch.all(above):
            new_m = m * self.ema_alpha + v * (1.0 - self.ema_alpha)
        else:
            arithmetic = (m * (self.num_steps - 1) + v) / self.num_steps
            ema = m * self.ema_alpha + v * (1.0 - self.ema_alpha)
            new_m = arithmetic * below + ema * above

        # Trick to set nans to zero (in case self.num_steps = 0 for any elements), since
        # these values can only be overwritten in the return statement if set to zero.
        nan_indices = self.num_steps == 0
        if torch.any(nan_indices):
            new_m[nan_indices] = 0

        return new_m * flags + m * torch.logical_not(flags)
