"""
Object definition for Metrics class, which stores and updates training performance metrics.
"""

from collections import deque
from typing import Dict, List

import numpy as np


class Metrics:
    """ Metric object, which stores and updates training performance metrics. """

    def __init__(self) -> None:
        """ Init function for Metrics object. """

        self.mean_reward: List[float] = []
        self.median_reward: List[float] = []
        self.min_reward: List[float] = []
        self.max_reward: List[float] = []

        self.state_vars = [
            "mean_reward",
            "median_reward",
            "min_reward",
            "max_reward",
        ]

    def __repr__(self) -> str:
        """ String representation of ``self``. """

        message = ""
        for i, (metric_name, value) in enumerate(self.current_values().items()):
            if i != 0:
                message += ", "
            message += "%s: %.5f" % (metric_name, value)
        return message

    def update(self, episode_rewards: deque) -> None:
        """
        Update performance metrics with a sequence of the most recent episode rewards.
        """

        self.mean_reward.append(np.mean(episode_rewards))
        self.median_reward.append(np.median(episode_rewards))
        self.min_reward.append(np.min(episode_rewards))
        self.max_reward.append(np.max(episode_rewards))

    def current_values(self) -> Dict[str, float]:
        """
        Return a dictionary of the most recent values for each performance metric.
        """

        return {
            state_var: getattr(self, state_var)[-1] for state_var in self.state_vars
        }

    def history(self) -> Dict[str, List[float]]:
        """
        Return a dictionary of the history of values for each performance metric.
        """

        return {state_var: getattr(self, state_var) for state_var in self.state_vars}
