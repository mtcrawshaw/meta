"""
Definition of RolloutStorage, an object to hold rollout information for one or more episodes.
"""

from typing import Dict, Tuple, Generator

import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from gym.spaces import Space, Discrete, Box


class RolloutStorage:
    """ An object to store rollout information. """

    def __init__(
        self,
        rollout_length: int,
        observation_space: Space,
        action_space: Space,
        num_processes: int,
    ) -> None:
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
        num_processes : int
            Number of processes to run in the environment.
        """

        # Get observation and action shape.
        self.observation_space = observation_space
        self.action_space = action_space
        self.space_shapes: Dict[str, Tuple[int]] = {}
        spaces = {"obs": observation_space, "action": action_space}
        for space_name, space in spaces.items():
            if isinstance(space, Discrete):
                if space_name == "obs":
                    self.space_shapes[space_name] = (space.n,)
                elif space_name == "action":
                    self.space_shapes[space_name] = (1,)
                else:
                    raise ValueError("Unrecognized space '%s'." % space_name)
            elif isinstance(space, Box):
                self.space_shapes[space_name] = space.shape
            else:
                raise ValueError(
                    "'%r' not a supported %s space." % (type(space), space_name)
                )

        # Misc state.
        self.rollout_length = rollout_length
        self.num_processes = num_processes
        self.rollout_step = 0

        # Initialize rollout information.
        self.init_rollout_info()

    def __repr__(self) -> str:
        """ String representation of RolloutStorage. """

        attrs = [
            "obs",
            "value_preds",
            "actions",
            "action_log_probs",
            "rewards",
            "dones",
        ]
        state = {attr: getattr(self, attr) for attr in attrs}
        return str(state)

    def init_rollout_info(self) -> None:
        """ Initialize rollout information. """

        # The +1 is here because we want to store the obs/value prediction
        # from before the first step and after the last step of the rollout.
        self.obs = torch.zeros(self.rollout_length + 1, self.num_processes, *self.space_shapes["obs"])
        self.value_preds = torch.zeros(self.rollout_length + 1, self.num_processes)
        self.actions = torch.zeros(self.rollout_length, self.num_processes, *self.space_shapes["action"])
        self.dones = torch.zeros(self.rollout_length, self.num_processes)

        self.action_log_probs = torch.zeros(self.rollout_length, self.num_processes)
        self.rewards = torch.zeros(self.rollout_length, self.num_processes)

    def add_step(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        done: bool,
        action_log_prob: torch.Tensor,
        value_pred: torch.Tensor,
        reward: torch.Tensor,
    ) -> None:
        """
        Add an environment step to storage.

        obs : torch.Tensor
            Observation returned from environment after step was taken.
        action : torch.Tensor,
            Action taken in environment step.
        done : bool,
            Whether or not step was terminal (done value returned from env.step()).
        action_log_prob : torch.Tensor,
            Log probs of action distribution output by policy network.
        value_pred : torch.Tensor,
            Value prediction from policy at step.
        reward : torch.Tensor,
            Reward earned from environment step.
        """

        if self.rollout_step >= self.rollout_length:
            raise ValueError("RolloutStorage object is full.")

        self.obs[self.rollout_step + 1] = obs
        self.actions[self.rollout_step] = action
        self.dones[self.rollout_step] = 1 if done else 0
        self.action_log_probs[self.rollout_step] = action_log_prob
        self.value_preds[self.rollout_step] = value_pred
        self.rewards[self.rollout_step] = reward

        self.rollout_step += 1

    def set_initial_obs(self, obs: torch.Tensor) -> None:
        """
        Set the first observation in storage.

        Arguments
        ---------
        obs : torch.Tensor
            Observation returned from the environment.
        """

        self.obs[0].copy_(obs)

    def minibatch_generator(self, minibatch_size: int) -> Generator:
        """
        Generates minibatches from rollout to train on. Note that this samples from the
        entire RolloutStorage object, even if only a small portion of it has been
        filled. The remaining values default to zero.

        Arguments
        ---------
        minibatch_size : int
            Size of minibatches to return.

        Yields
        ------
        minibatch: Tuple[List[int], torch.Tensor, ...]
            Tuple of batch indices with tensors containing rollout minibatch info.
        """

        if minibatch_size > self.rollout_length:
            raise ValueError(
                "Minibatch size (%d) is required to be no larger than"
                " rollout_length (%d)" % (minibatch_size, self.rollout_length)
            )

        total_steps = self.rollout_length * self.num_processes
        sampler = BatchSampler(
            sampler=SubsetRandomSampler(range(total_steps)),
            batch_size=minibatch_size,
            drop_last=True,
        )

        # Here we aggregate the obs, value_preds, etc. from each process into one
        # dimension.
        agg_obs = self.obs[:-1].view(total_steps, *self.space_shapes["obs"])
        agg_value_preds = self.value_preds[:-1].view(total_steps)
        agg_actions = self.actions.view(total_steps, *self.space_shapes["action"])
        agg_action_log_probs = self.action_log_probs.view(total_steps)

        for batch_indices in sampler:

            # Yield a minibatch corresponding to indices from sampler.
            # The -1 here is to exclude the obs/value_pred from after the last step.
            obs_batch = agg_obs[batch_indices]
            value_preds_batch = agg_value_preds[batch_indices]
            actions_batch = agg_actions[batch_indices]
            action_log_probs_batch = agg_action_log_probs[batch_indices]

            yield batch_indices, obs_batch, value_preds_batch, actions_batch, action_log_probs_batch

    def insert_rollout(self, new_rollout: "RolloutStorage", pos: int) -> None:
        """
        Insert the values from one RolloutStorage object into ``self`` at position
        ``pos``, ignoring the values from after the last step.

        new_rollout : RolloutStorage
            New rollout values to insert into ``self``.
        pos : int
            Position at which to insert values of ``new_rollout``.
        """

        end = pos + new_rollout.rollout_step
        self.obs[pos:end] = new_rollout.obs[: new_rollout.rollout_step]
        self.value_preds[pos:end] = new_rollout.value_preds[: new_rollout.rollout_step]
        self.actions[pos:end] = new_rollout.actions[: new_rollout.rollout_step]
        self.action_log_probs[pos:end] = new_rollout.action_log_probs[
            : new_rollout.rollout_step
        ]
        self.rewards[pos:end] = new_rollout.rewards[: new_rollout.rollout_step]
