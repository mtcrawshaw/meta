import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal

from meta.network import PolicyNetwork


class PPOPolicy:
    """ A policy class for PPO. """

    def __init__(
        self,
        observation_space,
        action_space,
        clip_param,
        ppo_epoch,
        minibatch_size,
        gamma,
        gae_lambda,
        use_proper_time_limits,
        value_loss_coef,
        entropy_coef,
        lr=None,
        eps=None,
        max_grad_norm=None,
        use_clipped_value_loss=True,
    ):

        # Set policy state.
        self.observation_space = observation_space
        self.action_space = action_space
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.use_proper_time_limits = use_proper_time_limits
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # Initialize policy network and optimizer.
        self.policy_network = PolicyNetwork(observation_space, action_space)

        if action_space.__class__.__name__ == "Discrete":
            self.num_outputs = action_space.n
            self.distribution_cls = Categorical
        elif action_space.__class__.__name__ == "Box":
            self.num_outputs = action_space.shape[0]
            self.distribution_cls = Normal
        else:
            raise NotImplementedError

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr, eps=eps)

    def act(self, inputs):
        value_pred, action_probs = self.policy_network(inputs)
        dist = self.distribution_cls(**action_probs)

        if self.distribution_cls == Categorical:
            action = dist.sample().unsqueeze(-1)
            action_log_probs = (
                dist.log_prob(action.squeeze(-1))
                .view(action.size(0), -1)
                .sum(-1)
                .unsqueeze(-1)
            )
        elif self.distribution_cls == Normal:
            action = dist.sample()
            action_log_probs = dist.log_prob(action).sum(-1, keepdim=True)
        else:
            raise NotImplementedError

        return value_pred, action, action_log_probs

    def get_value(self, inputs):
        value, _ = self.policy_network(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        value, action_probs = self.policy_network(inputs)
        dist = self.distribution_cls(**action_probs)

        if self.distribution_cls == Categorical:
            action_log_probs = (
                dist.log_prob(action.squeeze(-1))
                .view(action.size(0), -1)
                .sum(-1)
                .unsqueeze(-1)
            )
        elif self.distribution_cls == Normal:
            action_log_probs = dist.log_prob(action).sum(-1, keepdim=True)

        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy

    def compute_returns(self, rollouts):

        with torch.no_grad():
            next_value = self.get_value(rollouts.obs[-1]).detach()

        num_steps, num_processes = rollouts.rewards.size()[0:2]
        returns = torch.zeros(num_steps + 1, num_processes, 1)
        if self.use_proper_time_limits:
            rollouts.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(rollouts.rewards.size(0))):
                delta = (
                    rollouts.rewards[step]
                    + self.gamma
                    * rollouts.value_preds[step + 1]
                    * rollouts.masks[step + 1]
                    - rollouts.value_preds[step]
                )
                gae = (
                    delta
                    + self.gamma * self.gae_lambda * rollouts.masks[step + 1] * gae
                )
                gae = gae * rollouts.bad_masks[step + 1]
                returns[step] = gae + rollouts.value_preds[step]
        else:
            rollouts.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(rollouts.rewards.size(0))):
                delta = (
                    rollouts.rewards[step]
                    + self.gamma
                    * rollouts.value_preds[step + 1]
                    * rollouts.masks[step + 1]
                    - rollouts.value_preds[step]
                )
                gae = (
                    delta
                    + self.gamma * self.gae_lambda * rollouts.masks[step + 1] * gae
                )
                returns[step] = gae + rollouts.value_preds[step]

        return returns

    def update(self, rollouts):

        # Compute returns and advantages.
        returns = self.compute_returns(rollouts)
        advantages = returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        num_updates = 0

        for _ in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(self.minibatch_size)

            for sample in data_generator:
                (
                    batch_indices,
                    obs_batch,
                    actions_batch,
                    value_preds_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                ) = sample
                returns_batch = returns[:-1].view(-1, 1)[batch_indices]
                advantages_batch = advantages.view(-1, 1)[batch_indices]

                # Reshape to do in a single forward pass for all steps
                (values, action_log_probs, dist_entropy,) = self.evaluate_actions(
                    obs_batch, actions_batch
                )

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * advantages_batch
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                    * advantages_batch
                )
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (
                        values - value_preds_batch
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - returns_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - returns_batch).pow(2)
                    value_loss = (
                        0.5 * torch.max(value_losses, value_losses_clipped).mean()
                    )
                else:
                    value_loss = 0.5 * (returns_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                (
                    value_loss * self.value_loss_coef
                    + action_loss
                    - dist_entropy * self.entropy_coef
                ).backward()
                nn.utils.clip_grad_norm_(
                    self.policy_network.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

                num_updates += 1

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
