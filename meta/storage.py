"""
Definition of RolloutStorage, an object to hold rollout information for one or more episodes.
"""

from typing import Dict, Tuple, Generator, List

import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from gym.spaces import Space, Discrete, Box

from meta.utils import get_space_shape, combine_first_two_dims


class RolloutStorage:
    """ An object to store rollout information. """

    def __init__(
        self,
        rollout_length: int,
        observation_space: Space,
        action_space: Space,
        num_processes: int,
        hidden_state_size: int,
        device: torch.device = None,
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
        hidden_state_size: int,
            Size of hidden state of recurrent layer of policy.
        device : torch.device
            Which device to store rollout on.
        """

        # Get observation and action shape.
        self.observation_space = observation_space
        self.action_space = action_space
        self.space_shapes: Dict[str, Tuple[int]] = {}
        spaces = {"obs": observation_space, "action": action_space}
        for space_name, space in spaces.items():
            self.space_shapes[space_name] = get_space_shape(space, space_name)

        # Misc state.
        self.rollout_length = rollout_length
        self.num_processes = num_processes
        self.hidden_state_size = hidden_state_size
        self.device = device if device is not None else torch.device("cpu")
        self.rollout_step = 0
        self.members = [
            "obs",
            "value_preds",
            "actions",
            "action_log_probs",
            "rewards",
            "hidden_states",
            "dones",
        ]

        # Initialize rollout information.
        self.init_rollout_info()

        # Set device.
        self.to(device)

    def __repr__(self) -> str:
        """ String representation of RolloutStorage. """

        state = {member: getattr(self, member) for member in self.members}
        return str(state)

    def init_rollout_info(self) -> None:
        """ Initialize rollout information. """

        # The +1 is here because we want to store the obs/value prediction from before
        # the first step and after the last step of the rollout. The dimensions of
        # length 1 on the end of certain members is for convenience; this is how tensors
        # are shaped when they come out of the network. The choice is either to have 1's
        # here, or use squeezes in many places through the training pipeline.
        self.obs = torch.zeros(
            self.rollout_length + 1, self.num_processes, *self.space_shapes["obs"]
        )
        self.value_preds = torch.zeros(self.rollout_length + 1, self.num_processes, 1)
        self.actions = torch.zeros(
            self.rollout_length, self.num_processes, *self.space_shapes["action"]
        )
        self.dones = torch.zeros(self.rollout_length + 1, self.num_processes, 1)

        self.action_log_probs = torch.zeros(self.rollout_length, self.num_processes, 1)
        self.rewards = torch.zeros(self.rollout_length, self.num_processes, 1)
        self.hidden_states = torch.zeros(
            self.rollout_length + 1, self.num_processes, self.hidden_state_size
        )

    def add_step(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        dones: List[bool],
        action_log_prob: torch.Tensor,
        value_pred: torch.Tensor,
        reward: torch.Tensor,
        hidden_state: torch.Tensor,
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
        hidden_state : torch.Tensor,
            Hidden state of recurrent layer of policy at step.
        """

        if self.rollout_step >= self.rollout_length:
            raise ValueError("RolloutStorage object is full.")

        # This is to ensure that, in the case of a discrete action space, ``action`` has
        # shape [num_processes, 1] and not [num_processes].
        if action.shape == torch.Size([self.num_processes]):
            action = action.unsqueeze(-1)

        self.obs[self.rollout_step + 1] = obs
        self.actions[self.rollout_step] = action
        self.dones[self.rollout_step + 1] = torch.Tensor(
            [[1.0] if done else [0.0] for done in dones]
        )
        self.action_log_probs[self.rollout_step] = action_log_prob
        self.value_preds[self.rollout_step] = value_pred
        self.rewards[self.rollout_step] = reward
        self.hidden_states[self.rollout_step + 1] = hidden_state

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

    def reset(self) -> None:
        """
        Bring obs, hidden state, and done from last step into first step for next
        rollout.
        """

        self.obs[0].copy_(self.obs[self.rollout_step])
        self.hidden_states[0].copy_(self.hidden_states[self.rollout_step])
        self.dones[0].copy_(self.dones[self.rollout_step])
        self.rollout_step = 0

    def feedforward_minibatch_generator(self, num_minibatch: int) -> Generator:
        """
        Generates minibatches from rollout to train a feedforward policy network. Note
        that this samples from the entire RolloutStorage object, even if only a small
        portion of it has been filled. The remaining values default to zero.

        Arguments
        ---------
        num_minibatch : int
            Number of minibatches to return.

        Yields
        ------
        minibatch: Tuple[List[int], torch.Tensor, ...]
            Tuple of batch indices with tensors containing rollout minibatch info.
        """

        # Compute minibatch size.
        total_steps = self.rollout_length * self.num_processes
        minibatch_size = total_steps // num_minibatch
        if minibatch_size == 0:
            raise ValueError(
                "The number of minibatches (%d) is required to be no larger than"
                " rollout_length (%d) * num_processes (%d)"
                % (num_minibatch, self.rollout_length, self.num_processes)
            )

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
        agg_dones = self.dones[:-1].view(total_steps)

        # We need to return some hidden state, so just return zeros every time.
        hidden_states_batch = torch.zeros(total_steps, self.hidden_state_size)

        for batch_indices in sampler:

            # Yield a minibatch corresponding to indices from sampler.
            # The -1 here is to exclude the obs/value_pred from after the last step.
            obs_batch = agg_obs[batch_indices]
            value_preds_batch = agg_value_preds[batch_indices]
            actions_batch = agg_actions[batch_indices]
            action_log_probs_batch = agg_action_log_probs[batch_indices]
            dones_batch = agg_dones[batch_indices]

            yield batch_indices, obs_batch, value_preds_batch, actions_batch, action_log_probs_batch, dones_batch, hidden_states_batch

    def recurrent_minibatch_generator(self, num_minibatch: int) -> Generator:
        """
        Generates minibatches from rollout to train a recurrent policy network. Note
        that this samples from the entire RolloutStorage object, even if only a small
        portion of it has been filled. The remaining values default to zero.

        Arguments
        ---------
        num_minibatch : int
            Number of minibatches to return.

        Yields
        ------
        minibatch: Tuple[List[int], torch.Tensor, ...]
            Tuple of batch indices with tensors containing rollout minibatch info.
        """

        # Compute number trajectories (single process rollout) per minibatch.
        trajectory_per_minibatch = self.num_processes // num_minibatch
        if trajectory_per_minibatch == 0:
            raise ValueError(
                "The number of minibatches (%d) is required to be no larger than"
                " num_processes (%d)"
                % (num_minibatch, self.rollout_length, self.num_processes)
            )

        # Randomly permute trajectories.
        trajectory_permutation = torch.randperm(self.num_processes)

        # Each loop iteration constructs one minibatch.
        for minibatch in range(num_minibatch):
            batch_indices = []
            obs_batch_list = []
            value_preds_batch_list = []
            actions_batch_list = []
            action_log_probs_batch_list = []
            dones_batch_list = []
            hidden_states_batch_list = []

            # Add trajectory info to list. We only take the first hidden_state from each
            # trajectory, since hidden states for later steps will be computed during
            # the training step.
            for trajectory_index in range(trajectory_per_minibatch):
                process = trajectory_permutation[
                    minibatch * trajectory_per_minibatch + trajectory_index
                ]
                batch_indices.append(process)
                obs_batch_list.append(self.obs[: self.rollout_step, process])
                value_preds_batch_list.append(
                    self.value_preds[: self.rollout_step, process]
                )
                actions_batch_list.append(self.actions[:, process])
                action_log_probs_batch_list.append(self.action_log_probs[:, process])
                dones_batch_list.append(self.dones[: self.rollout_step, process])
                hidden_states_batch_list.append(self.hidden_states[0:1, process])

            # Stack each list into a single tensor of size (self.rollout_step,
            # trajectory_per_minibatch, ...).
            obs_batch = torch.stack(obs_batch_list, 1)
            value_preds_batch = torch.stack(value_preds_batch_list, 1)
            actions_batch = torch.stack(actions_batch_list, 1)
            action_log_probs_batch = torch.stack(action_log_probs_batch_list, 1)
            dones_batch = torch.stack(dones_batch_list, 1)
            hidden_states_batch = torch.stack(hidden_states_batch_list, 1)

            # Combine first two dimensions of each tensor, so that they are each of size
            # (self.rollout_step * trajectory_per_minibatch, ...).
            obs_batch = combine_first_two_dims(obs_batch)
            value_preds_batch = combine_first_two_dims(value_preds_batch)
            actions_batch = combine_first_two_dims(actions_batch)
            action_log_probs_batch = combine_first_two_dims(action_log_probs_batch)
            dones_batch = combine_first_two_dims(dones_batch)
            hidden_states_batch = combine_first_two_dims(hidden_states_batch)

            yield batch_indices, obs_batch, value_preds_batch, actions_batch, action_log_probs_batch, dones_batch, hidden_states_batch

    def to(self, device: torch.device) -> None:
        """ Move tensor members to ``device``. """

        for member in self.members:
            tensor = getattr(self, member)
            setattr(self, member, tensor.to(device))

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
        self.hidden_states[pos:end] = new_rollout.rewards[: new_rollout.rollout_step]

    def print_devices(self) -> None:
        """ Print devices of tensor members. """
        for member in self.members:
            print("%s device: %s" % (member, getattr(self, member).device))
