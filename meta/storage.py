import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutStorage:
    def __init__(
        self, num_steps, obs_shape, action_space,
    ):
        self.obs = torch.zeros(num_steps + 1, *obs_shape)
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
        self.masks = torch.ones(num_steps + 1, 1)

        self.num_steps = num_steps
        self.step = 0

    def insert(
        self, obs, actions, action_log_probs, value_preds, rewards, masks,
    ):
        self.obs[self.step + 1].copy_(obs)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])

    def feed_forward_generator(self, minibatch_size):
        batch_size = self.rewards.size()[0]  # This is args.rollout_length
        if minibatch_size > batch_size:
            raise ValueError(
                "Minibatch size (%d) is required to be no larger than"
                " num_steps (%d)" % (minibatch_size, num_steps)
            )

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)), minibatch_size, drop_last=True
        )
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[1:])[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]

            yield indices, obs_batch, actions_batch, value_preds_batch, masks_batch, old_action_log_probs_batch
