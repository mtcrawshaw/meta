"""
Object definition for Metrics class, which stores and updates training performance metrics.
"""

from math import sqrt
from collections import deque
from typing import Dict, List, Any

import numpy as np


class Metrics:
    """
    Metrics object, which stores and updates training performance metrics.
    """

    def __init__(self) -> None:
        """ Init function for Metrics object. """

        self.reward = Metric()
        self.success = Metric()

        self.state_vars = ["reward", "success"]

    def __repr__(self) -> str:
        """ String representation of ``self``. """

        message = ""
        for i, state_var in enumerate(self.state_vars):
            if i != 0:
                message += ", "
            message += "%s %s" % (state_var, getattr(self, state_var))

        return message

    def update(self, episode_rewards: List[float], episode_successes: List[float]) -> None:
        """
        Update performance metrics with a sequence of the most recent episode rewards.
        """

        self.reward.update(episode_rewards)

        # Success rates are None for environments that don't provide a 0-1 success
        # signal, so we only compute success rate if this is not the case.
        if not any(success is None for success in episode_successes):
            self.success.update(episode_successes)

    def current_values(self) -> Dict[str, float]:
        """
        Return a dictionary of the most recent values for each performance metric.
        """

        return {
            state_var: getattr(self, state_var).mean[-1]
            for state_var in self.state_vars
        }

    def history(self) -> Dict[str, List[float]]:
        """
        Return a dictionary of the history of values for each performance metric.
        """

        return {
            state_var: getattr(self, state_var).history
            for state_var in self.state_vars
        }

    def state(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a dictionary with the value of all state variables.
        """

        return {state_var: getattr(self, state_var).state() for state_var in self.state_vars}


class Metric:
    """ Class to store values for a single metric. """

    def __init__(self) -> None:
        """ Init function for Metric. """

        # Exponential moving average settings. ema_alpha is the coefficient used to
        # compute EMA. ema_threshold is the number of data points at which we switch
        # from a regular average to an EMA. Using a regular average reduces bias when
        # there are a small number data points, so we use it at the beginning.
        self.ema_alpha = 0.999
        self.ema_threshold = 30

        # Metric values.
        self.history = []
        self.mean = []
        self.stdev = []

        self.state_vars = ["history", "mean", "stdev"]

    def __repr__(self) -> None:
        """ String representation of ``self``. """

        if len(self.history) > 0:
            mean = self.mean[-1]
            stdev = self.stdev[-1]
            message = "mean, stdev: %.5f, %.5f" % (mean, stdev)
        else:
            message = "mean, stdev: %r, %r" % (None, None)

        return message

    def update(self, values: List[float]) -> None:
        """ Update history, mean, and stdev with new values. """

        # Add each episode's reward to history and update running estimates.
        for value in values:
            self.history.append(value)

            # Compute a regular average and standard deviation when the number of data
            # points is small, otherwise compute an EMA.
            if len(self.history) <= self.ema_threshold:
                self.mean.append(np.mean(self.history))
                self.stdev.append(np.std(self.history))
            else:
                old_second_moment = self.mean[-1] ** 2 + self.stdev[-1] ** 2
                self.mean.append(self.ema_update(self.mean[-1], value))
                new_second_moment = self.ema_update(old_second_moment, value ** 2)
                self.stdev.append(sqrt(new_second_moment - self.mean[-1] ** 2))

    def ema_update(self, average: float, new_value: float) -> float:
        """ Compute one exponential moving average update. """

        return average * self.ema_alpha + new_value * (1.0 - self.ema_alpha)

    def state(self) -> Dict[str, Any]:
        """ Return a dictionary with the value of all state variables. """

        return {state_var: getattr(self, state_var) for state_var in self.state_vars}
