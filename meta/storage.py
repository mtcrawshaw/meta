from typing import List

import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutStorage:
    def __init__(
        self, num_steps, observation_space, action_space,
    ):

        self.observation_space = observation_space
        self.action_space = action_space

        self.obs = torch.zeros(num_steps + 1, *self.observation_space.shape)
        self.rewards = torch.zeros(num_steps, 1)
        self.value_preds = torch.zeros(num_steps + 1, 1)
        self.action_log_probs = torch.zeros(num_steps, 1)
        if action_space.__class__.__name__ == "Discrete":
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, action_shape)
        if action_space.__class__.__name__ == "Discrete":
            self.actions = self.actions.long()
        self.done = False

        self.num_steps = num_steps
        self.step = 0

    def insert(
        self, obs, actions, action_log_probs, value_preds, rewards,
    ):

        if self.step > self.num_steps:
            raise ValueError("RolloutStorage object is full.")

        self.obs[self.step + 1].copy_(obs)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)

        self.step += 1

    def feed_forward_generator(self, minibatch_size):
        batch_size = self.rewards.size()[0]  # This is args.rollout_length
        if minibatch_size > batch_size:
            raise ValueError(
                "Minibatch size (%d) is required to be no larger than"
                " rollout_length (%d)" % (minibatch_size, batch_size)
            )

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)), minibatch_size, drop_last=True
        )
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[1:])[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]

            yield indices, obs_batch, value_preds_batch, actions_batch, old_action_log_probs_batch

    def insert_rollout(self, new_rollout: "RolloutStorage", pos: int):
        """
        Insert the values from one RolloutStorage object into ``self`` at position
        ``pos``, ignoring the values from after the last step.
        """

        end = pos + new_rollout.step
        self.obs[pos:end] = new_rollout.obs[: new_rollout.step]
        self.value_preds[pos:end] = new_rollout.value_preds[: new_rollout.step]
        self.actions[pos:end] = new_rollout.actions[: new_rollout.step]
        self.action_log_probs[pos:end] = new_rollout.action_log_probs[
            : new_rollout.step
        ]
        self.rewards[pos:end] = new_rollout.rewards[: new_rollout.step]


def combine_rollouts(individual_rollouts: List[RolloutStorage]) -> RolloutStorage:
    """
    Given a list of individual RolloutStorage objects, returns a single combined
    RolloutStorage object.
    """

    if len(individual_rollouts) == 0:
        raise ValueError("Received empty list of rollouts.")

    rollout_length = sum([rollout.step for rollout in individual_rollouts])
    rollouts = RolloutStorage(
        num_steps=rollout_length,
        observation_space=individual_rollouts[0].observation_space,
        action_space=individual_rollouts[0].action_space,
    )

    current_pos = 0
    for rollout in individual_rollouts:
        rollouts.insert_rollout(rollout, current_pos)
        current_pos += rollout.step

    # Set combined rollout_step.
    rollouts.step = current_pos

    return rollouts
