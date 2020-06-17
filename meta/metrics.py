"""
Object definition for Metrics class, which stores and updates training performance metrics.
"""

from math import sqrt
from collections import deque
from typing import Dict, List, Any

import numpy as np


class Metrics:
    """
    Metric object, which stores and updates training performance metrics. Each metric is
    represented with a dictionary, holding the history, running mean, and running stdev.
    """

    def __init__(self) -> None:
        """ Init function for Metrics object. """

        self.reward = {
            "history": [],
            "mean": [],
            "stdev": [],
        }

        # Exponential moving average settings. ema_alpha is the coefficient used to
        # compute EMA. ema_threshold is the number of data points at which we switch
        # from a regular average to an EMA. Using a regular average reduces bias when
        # there are a small number data points, so we use it at the beginning.
        self.ema_alpha = 0.9
        self.ema_threshold = 30

        self.state_vars = ["reward"]

    def __repr__(self) -> str:
        """ String representation of ``self``. """

        message = ""
        for i, state_var in enumerate(self.state_vars):
            if i != 0:
                message += ", "

            if len(getattr(self, state_var)["mean"]) > 0:
                mean = getattr(self, state_var)["mean"][-1]
                stdev = getattr(self, state_var)["stdev"][-1]
                message += "%s mean, stdev: %.5f, %.5f" % (state_var, mean, stdev)
            else:
                message += "%s mean, stdev: %r, %r" % (state_var, None, None)

        return message

    def update(self, episode_rewards: List[float]) -> None:
        """
        Update performance metrics with a sequence of the most recent episode rewards.
        """

        # Add each episode's reward to history and update running estimates.
        for episode_reward in episode_rewards:
            self.reward["history"].append(episode_reward)

            # Compute a regular average and standard deviation when the number of data
            # points is small, otherwise compute an EMA.
            if len(self.reward["history"]) <= self.ema_threshold:
                self.reward["mean"].append(np.mean(self.reward["history"]))
                self.reward["stdev"].append(np.std(self.reward["history"]))
            else:
                old_second_moment = (
                    self.reward["mean"][-1] ** 2 + self.reward["stdev"][-1] ** 2
                )
                self.reward["mean"].append(
                    self.ema_update(self.reward["mean"][-1], episode_reward)
                )
                new_second_moment = self.ema_update(
                    old_second_moment, episode_reward ** 2
                )
                self.reward["stdev"].append(
                    sqrt(new_second_moment - self.reward["mean"][-1] ** 2)
                )

    def ema_update(self, average: float, new_value: float) -> float:
        """ Compute one exponential moving average update. """

        return average * self.ema_alpha + new_value * (1.0 - self.ema_alpha)

    def current_values(self) -> Dict[str, float]:
        """
        Return a dictionary of the most recent values for each performance metric.
        """

        return {
            state_var: getattr(self, state_var)["mean"][-1]
            for state_var in self.state_vars
        }

    def history(self) -> Dict[str, List[float]]:
        """
        Return a dictionary of the history of values for each performance metric.
        """

        return {
            state_var: getattr(self, state_var)["history"]
            for state_var in self.state_vars
        }

    def state(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a dictionary with the value of all state variables.
        """

        return {state_var: getattr(self, state_var) for state_var in self.state_vars}
