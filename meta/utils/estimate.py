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
        ema_alpha : float
            Coefficient used to compute exponential moving average. We compute an
            arithmetic mean for the first `ema_threshold` steps (as computed below),
            then switch to EMA. If `ema_alpha == 1.0`, then we will never switch to EMA.
        """

        self.compute_stdev = compute_stdev
        self.condense_dims = condense_dims
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

        self.num_steps = 0
        self.sample_size = 0

    def update(self, val: torch.Tensor) -> None:
        """ Update running stats with new value. """

        self.num_steps += 1
        self.sample_size = min(self.sample_size + 1, self.ema_threshold)
        if len(self.condense_dims) > 0:
            self.mean = self.single_update(
                self.mean, torch.mean(val, dim=self.condense_dims)
            )
            if self.compute_stdev:
                self.square_mean = self.single_update(
                    self.square_mean, torch.mean(val ** 2, dim=self.condense_dims),
                )
        else:
            self.mean = self.single_update(self.mean, val)
            if self.compute_stdev:
                self.square_mean = self.single_update(self.square_mean, val ** 2)
        if self.compute_stdev:
            self.var = self.square_mean - self.mean ** 2
            self.stdev = torch.sqrt(self.var)

    def single_update(self, m: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.num_steps <= self.ema_threshold:
            return (m * (self.num_steps - 1) + v) / self.num_steps
        else:
            return m * self.ema_alpha + v * (1.0 - self.ema_alpha)
