import copy
from typing import Union

import numpy as np
import torch
from gym.spaces import Space, Discrete, Box

from utils import convert_to_tensor


class RolloutStorage:
    """ An object to store rollout information. """

    def __init__(
        self, rollout_length: int, observation_space: Space, action_space: Space
    ):
        """
        init function for RolloutStorage class.

        Arguments
        ---------
        rollout_length : int
            Length of the rollout between each update.
        observation_shape : Space
            Environment's observation space.
        action_space : Space
            Environment's action space.
        """

        # Get observation and action shape.
        self.observation_space = observation_space
        self.action_space = action_space
        self.space_shapes = {}
        spaces = {"obs": observation_space, "action": action_space}
        for space_name, space in spaces.items():
            if isinstance(space, Discrete):
                self.space_shapes[space_name] = (1,)
            elif isinstance(space, Box):
                self.space_shapes[space_name] = space.shape
            else:
                raise ValueError(
                    "'%r' not a supported %s space." % (type(space), space_name)
                )

        # Initialize rollout information.
        # The +1 is here because we want to keep the obs before the first step
        # and after the last step of the rollout.
        self.obs = torch.zeros(rollout_length + 1, *self.space_shapes["obs"])
        self.actions = torch.zeros(rollout_length, *self.space_shapes["action"])
        self.action_log_probs = torch.zeros(rollout_length, 1)
        self.value_preds = torch.zeros(rollout_length, 1)
        self.rewards = torch.zeros(rollout_length, 1)

        # Misc state.
        self.rollout_length = rollout_length
        self.rollout_step = 0

    def add_step(
        self,
        obs: Union[np.ndarray, int, float],
        action: Union[np.ndarray, int, float],
        action_log_prob: torch.Tensor,
        value_pred: torch.Tensor,
        reward: float,
    ):
        """
        Add an environment step to storage.

        obs : Union[np.ndarray, int],
            Observation returned from environment after step was taken.
        action : torch.Tensor,
            Action taken in environment step.
        action_log_prob : torch.Tensor,
            Log probs of action distribution output by policy network.
        value_pred : torch.Tensor,
            Value prediction from policy at step.
        reward : float,
            Reward earned from environment step.
        """

        if self.rollout_step == self.rollout_length:
            raise ValueError(
                "Rollout storage is already full, call RolloutStorage.clear() "
                "to clear storage before calling add_step()."
            )

        self.obs[self.rollout_step + 1].copy_(convert_to_tensor(obs))
        self.actions[self.rollout_step].copy_(convert_to_tensor(action))
        self.action_log_probs[self.rollout_step].copy_(action_log_prob)
        self.value_preds[self.rollout_step].copy_(value_pred)
        self.rewards[self.rollout_step].copy_(convert_to_tensor(reward))

        self.rollout_step += 1

    def clear(self):
        """ Clear the stored rollout. """

        # Store last observation to bring it into next rollout.
        last_obs = self.obs[-1]

        # The +1 is here because we want to keep the obs before the first step
        # and after the last step of the rollout.
        self.obs = torch.zeros(self.rollout_length + 1, *self.space_shapes["obs"])
        self.actions = torch.zeros(self.rollout_length, *self.space_shapes["action"])
        self.action_log_probs = torch.zeros(self.rollout_length, 1)
        self.value_preds = torch.zeros(self.rollout_length, 1)
        self.rewards = torch.zeros(self.rollout_length, 1)

        # Bring last observation into next rollout.
        self.obs[0].copy_(last_obs)

        # Reset rollout step.
        self.rollout_step = 0

    def set_initial_obs(self, obs: Union[np.ndarray, int]):
        """
        Set the first observation in storage.

        Arguments
        ---------
        obs: np.ndarray or int
            Observation returned from the environment.
        """

        tensor_obs = convert_to_tensor(obs)
        self.obs[0].copy_(tensor_obs)
