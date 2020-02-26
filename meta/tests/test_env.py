from typing import Tuple

import numpy as np
from gym.spaces import Discrete


class TestEnv:
    """ Environment for testing. Only has two states, and two actions.  """

    def __init__(self) -> None:
        """ Init function for TestEnv. """

        self.states = [np.array([1, 0]), np.array([0, 1])]
        self.observation_space = Discrete(len(self.states))
        self.action_space = Discrete(len(self.states))
        self.initial_state_index = 0
        self.state_index = self.initial_state_index
        self.state = self.states[self.state_index]

    def reset(self) -> int:
        """ Reset environment to initial state. """

        self.state_index = self.initial_state_index
        self.state = self.states[self.state_index]
        return self.state

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """
        Step function for environment. Returns an observation, a reward,
        whether or not the environment is done, and an info dictionary, as is
        the standard for OpenAI gym environments.
        """

        reward = 1 if action == self.state_index else -1
        self.state_index = (self.state_index + 1) % len(self.states)
        self.state = self.states[self.state_index]
        done = False
        info = {}

        return self.state, reward, done, info
