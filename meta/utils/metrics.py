"""
Object definition for Metrics class, which stores and updates training performance metrics.
"""

from typing import Dict, List, Tuple, Any

import numpy as np


class Metrics:
    """
    Metrics object, which stores and updates training performance metrics.
    """

    def __init__(self, metric_set: List[Tuple[Any]]) -> None:
        """ Init function for Metrics object. """

        # Set metrics.
        for metric_name, metric_window, point_avg, maximize in metric_set:
            setattr(
                self,
                metric_name,
                Metric(
                    window_len=metric_window, point_avg=point_avg, maximize=maximize
                ),
            )

        self.state_vars = [single_metric[0] for single_metric in metric_set]

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
        self, window_len: int = 50, point_avg: bool = False, maximize: bool = True
    ) -> None:
        """
        Init function for Metric. We keep track of the total history of metric values, a
        moving average of the past `window_len` values, a moving standard deviation of
        the past `window_len` values, and a maximum average so far. If `point_avg` is
        True, then `update` will condense the list of given values into their average
        and treat it as a single update for the metric.
        """

        self.window_len = window_len
        self.point_avg = point_avg
        self.maximize = maximize

        # Metric values.
        self.history: List[float] = []
        self.mean: List[float] = []
        self.stdev: List[float] = []
        self.best: float = None

        self.state_vars = ["history", "mean", "stdev", "best"]

    def __repr__(self) -> str:
        """ String representation of ``self``. """

        if len(self.history) > 0:
            mean = self.mean[-1]
            best = self.best
            message = "mean, best: %.5f, %.5f" % (mean, best)
        else:
            message = "mean, max: %r, %r" % (None, None)

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

            # Update moving average and standard deviation.
            self.mean.append(np.mean(self.history[-self.window_len :]))
            self.stdev.append(np.std(self.history[-self.window_len :]))

            # Compute new best.
            if self.best is None:
                self.best = self.mean[-1]
            else:
                if self.maximize:
                    new_best = self.mean[-1] > self.best
                else:
                    new_best = self.mean[-1] < self.best
                if new_best:
                    self.best = self.mean[-1]

    def state(self) -> Dict[str, Any]:
        """ Return a dictionary with the value of all state variables. """

        return {state_var: getattr(self, state_var) for state_var in self.state_vars}
