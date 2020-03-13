import torch
import torch.nn as nn
import torch.optim as optim

from meta.network import PolicyNetwork
from meta.utils import AddBias, init


class PPO:
    def __init__(
        self,
        actor_critic,
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

        self.actor_critic = actor_critic

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

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def compute_returns(self, rollouts):

        with torch.no_grad():
            next_value = self.actor_critic.get_value(
                rollouts.obs[-1], rollouts.masks[-1],
            ).detach()

        num_steps, num_processes = rollouts.rewards.size()[0:2]
        returns = torch.zeros(num_steps + 1, num_processes, 1)
        if self.use_proper_time_limits:
            rollouts.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(rollouts.rewards.size(0))):
                delta = (
                    rollouts.rewards[step]
                    + self.gamma * rollouts.value_preds[step + 1] * rollouts.masks[step + 1]
                    - rollouts.value_preds[step]
                )
                gae = delta + self.gamma * self.gae_lambda * rollouts.masks[step + 1] * gae
                gae = gae * rollouts.bad_masks[step + 1]
                returns[step] = gae + rollouts.value_preds[step]
        else:
            rollouts.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(rollouts.rewards.size(0))):
                delta = (
                    rollouts.rewards[step]
                    + self.gamma * rollouts.value_preds[step + 1] * rollouts.masks[step + 1]
                    - rollouts.value_preds[step]
                )
                gae = delta + self.gamma * self.gae_lambda * rollouts.masks[step + 1] * gae
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
                (
                    values,
                    action_log_probs,
                    dist_entropy,
                ) = self.actor_critic.evaluate_actions(
                    obs_batch, masks_batch, actions_batch
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
                    self.actor_critic.parameters(), self.max_grad_norm
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


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space):
        super(Policy, self).__init__()

        if len(obs_shape) == 3:
            num_inputs = obs_shape[0] * obs_shape[1] * obs_shape[2]
        elif len(obs_shape) == 1:
            num_inputs = obs_shape[0]
        else:
            raise NotImplementedError

        self.base = PolicyNetwork(num_inputs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.hidden_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.hidden_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.hidden_size, num_outputs)
        else:
            raise NotImplementedError

    def act(self, inputs, masks, deterministic=False):
        value, actor_features, = self.base(inputs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs

    def get_value(self, inputs, masks):
        value, _ = self.base(inputs, masks)
        return value

    def evaluate_actions(self, inputs, masks, action):
        value, actor_features = self.base(inputs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy


#
# Standardize distribution interfaces
#

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entrop(self):
        return super.entropy().sum(-1)

    def mode(self):
        return self.mean


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01
        )

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Bernoulli, self).__init__()

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)
