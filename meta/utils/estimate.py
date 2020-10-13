"""
Utility classes to compute running mean and standard deviation of torch.Tensors.
"""

from typing import Tuple

import torch


alpha_to_threshold = lambda alpha: round(1.0 / (1.0 - alpha)) if alpha != 1.0 else float('inf')


def single_update(
    m: torch.Tensor, v: torch.Tensor, n: int, threshold: int, alpha: float
) -> torch.Tensor:
    if n <= threshold:
        return (m * (n - 1) + v) / n
    else:
        return m * alpha + v * (1.0 - alpha)


class RunningMean:
    """ Utility class to compute running mean of torch.Tensor. """

    def __init__(
        self, shape: Tuple[int, ...] = None, condense: bool = False, ema_alpha=0.999
    ) -> None:
        """
        Init function for RunningMean. `shape` is shape of input tensors, which is only
        needed when `condense=False`. If `condense=True`, then we treat all elements of
        the input tensors as samples of the same variable, and compute a single-valued
        mean. `ema_alpha` is the coefficient used to compute an exponential moving
        average. Note that we compute an arithmetic mean for the first `ema_threshold`
        steps (as computed below), then switch to EMA. If `ema_alpha == 1.0`, then we
        will never switch to EMA. `self.sample_size` is akin to `self.num_steps`, but we
        stop increasing `self.sample_size` once we switch to EMA.
        """

        self.num_steps = 0
        self.sample_size = 0
        self.condense = condense
        self.ema_alpha = ema_alpha
        self.ema_threshold = alpha_to_threshold(ema_alpha)
        if self.condense:
            self.mean = torch.zeros(())
        else:
            assert shape is not None
            self.shape = shape
            self.mean = torch.zeros(*shape)

    def update(self, val: torch.Tensor) -> None:
        """ Update running mean with new value. """

        self.num_steps += 1
        self.sample_size = min(self.sample_size + 1, self.ema_threshold)
        if self.condense:
            self.mean = single_update(
                self.mean,
                torch.mean(val),
                self.num_steps,
                self.ema_threshold,
                self.ema_alpha,
            )
        else:
            self.mean = single_update(
                self.mean, val, self.num_steps, self.ema_threshold, self.ema_alpha
            )


class RunningMeanStdev:
    """
    Utility class to compute running mean and standard deviation of torch.Tensor. We do
    this by keeping a running estimate of the mean and a running estimate of the mean of
    the square, and using the formula `Var[X] = E[X^2] - E[X]^2.
    """

    def __init__(
        self, shape: Tuple[int, ...] = None, condense: bool = False, ema_alpha=0.999
    ) -> None:
        """
        Init function for RunningMeanStd. `shape` is shape of input tensors, which is
        only needed when `condense=False`. If `condense=True`, then we treat all
        elements of the input tensors as samples of the same variable, and compute a
        single-valued mean and standard deviation. `ema_alpha` is the coefficient used
        to compute an exponential moving average. Note that we compute an arithmetic
        mean for the first `ema_threshold` steps (as computed below), then switch to
        EMA. If `ema_alpha == 1.0`, then we will never switch to EMA.
        """

        self.num_steps = 0
        self.sample_size = 0
        self.condense = condense
        self.ema_alpha = ema_alpha
        self.ema_threshold = alpha_to_threshold(ema_alpha)
        if self.condense:
            self.mean = torch.zeros(())
            self.square_mean = torch.zeros(())
            self.var = torch.zeros(())
            self.stdev = torch.zeros(())
        else:
            assert shape is not None
            self.shape = shape
            self.mean = torch.zeros(*shape)
            self.square_mean = torch.zeros(*shape)
            self.var = torch.zeros(*shape)
            self.stdev = torch.zeros(*shape)

    def update(self, val: torch.Tensor) -> None:
        """ Update running mean with new value. """

        self.num_steps += 1
        self.sample_size = min(self.sample_size + 1, self.ema_threshold)
        if self.condense:
            self.mean = single_update(
                self.mean,
                torch.mean(val),
                self.num_steps,
                self.ema_threshold,
                self.ema_alpha,
            )
            self.square_mean = single_update(
                self.square_mean,
                torch.mean(val ** 2),
                self.num_steps,
                self.ema_threshold,
                self.ema_alpha,
            )
        else:
            self.mean = single_update(
                self.mean, val, self.num_steps, self.ema_threshold, self.ema_alpha
            )
            self.square_mean = single_update(
                self.square_mean,
                val ** 2,
                self.num_steps,
                self.ema_threshold,
                self.ema_alpha,
            )
        self.var = self.square_mean - self.mean ** 2
        self.stdev = torch.sqrt(self.var)
