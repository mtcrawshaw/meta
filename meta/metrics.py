"""
Object definition for Metrics class, which stores and updates training performance metrics.
"""

from math import sqrt
from collections import deque
from typing import Dict, List, Any

import numpy as np


# Exponential moving average settings. ema_alpha is the coefficient used to
# compute EMA. ema_threshold is the number of data points at which we switch
# from a regular average to an EMA. Using a regular average reduces bias when
# there are a small number data points, so we use it at the beginning.
EMA_ALPHA = 0.999
EMA_THRESHOLD = 1000


class Metrics:
    """
    Metrics object, which stores and updates training performance metrics.
    """

    def __init__(self) -> None:
        """ Init function for Metrics object. """

        self.train_reward = Metric()
        self.train_success = Metric()
        self.eval_reward = Metric(point_avg=True, ema_alpha=0.75, ema_threshold=10)
        self.eval_success = Metric(point_avg=True, ema_alpha=0.75, ema_threshold=10)

        self.state_vars = [
            "train_reward",
            "train_success",
            "eval_reward",
            "eval_success",
        ]

    def __repr__(self) -> str:
        """ String representation of ``self``. """

        message = ""
        for i, state_var in enumerate(self.state_vars):
            if i != 0:
                message += " | "
            message += "%s %s" % (state_var, getattr(self, state_var))

        return message

    def update(self, update_values: Dict[str, List[float]]) -> None:
        """
        Update performance metrics with a sequence of the most recent episode rewards.
        """

        for metric_name, metric_values in update_values.items():
            getattr(self, metric_name).update(metric_values)

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
            state_var: getattr(self, state_var).history for state_var in self.state_vars
        }

    def state(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a dictionary with the value of all state variables.
        """

        return {
            state_var: getattr(self, state_var).state() for state_var in self.state_vars
        }


class Metric:
    """ Class to store values for a single metric. """

    def __init__(
        self, point_avg=False, ema_alpha=EMA_ALPHA, ema_threshold=EMA_THRESHOLD
    ) -> None:
        """ Init function for Metric. """

        self.point_avg = point_avg
        self.ema_alpha = ema_alpha
        self.ema_threshold = ema_threshold

        # Metric values.
        self.history = []
        self.mean = []
        self.stdev = []
        self.maximum = None

        self.state_vars = ["history", "mean", "stdev", "maximum"]

    def __repr__(self) -> str:
        """ String representation of ``self``. """

        if len(self.history) > 0:
            mean = self.mean[-1]
            stdev = self.stdev[-1]
            maximum = self.maximum
            message = "mean, maximum: %.5f, %.5f" % (mean, maximum)
        else:
            message = "mean, maximum: %r, %r" % (None, None)

        return message

    def update(self, values: List[float]) -> None:
        """ Update history, mean, and stdev with new values. """

        # Replace ``values`` with average of ``values`` if self.point_avg, using recent
        # history to fill in if ``values`` is empty.
        if self.point_avg:

            if len(values) > 0:
                new_val = np.mean(values)
            elif len(self.mean) > 0:
                new_val = self.mean[-1]
            else:
                # If ``values`` and history are both empty, do nothing.
                return

            values = [new_val]

        # Add each new value to history and update running estimates.
        for value in values:
            self.history.append(value)

            # Compute a regular average and standard deviation when the number of
            # data points is small, otherwise compute an EMA.
            if len(self.history) <= self.ema_threshold:
                self.mean.append(np.mean(self.history))
                self.stdev.append(np.std(self.history))
            else:
                old_second_moment = self.mean[-1] ** 2 + self.stdev[-1] ** 2
                self.mean.append(self.ema_update(self.mean[-1], value))
                new_second_moment = self.ema_update(old_second_moment, value ** 2)
                self.stdev.append(sqrt(new_second_moment - self.mean[-1] ** 2))

            # Compute new maximum.
            if self.maximum is None or self.mean[-1] > self.maximum:
                self.maximum = self.mean[-1]

    def ema_update(self, average: float, new_value: float) -> float:
        """ Compute one exponential moving average update. """

        return average * self.ema_alpha + new_value * (1.0 - self.ema_alpha)

    def state(self) -> Dict[str, Any]:
        """ Return a dictionary with the value of all state variables. """

        return {state_var: getattr(self, state_var) for state_var in self.state_vars}
