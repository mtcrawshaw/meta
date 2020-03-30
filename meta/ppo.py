from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
from gym.spaces import Space, Box, Discrete

from meta.network import PolicyNetwork
from meta.storage import RolloutStorage, combine_rollouts


class PPOPolicy:
    """ A policy class for PPO. """

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        minibatch_size: int,
        num_ppo_epochs: int = 4,
        lr: float = 7e-4,
        eps: float = 1e-5,
        value_loss_coeff: float = 0.5,
        entropy_loss_coeff: float = 0.01,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_param: float = 0.2,
        max_grad_norm: float = 0.5,
        clip_value_loss: bool = True,
        num_layers: int = 3,
        hidden_size: int = 64,
        normalize_advantages: float = True,
    ):
        """
        init function for PPOPolicy.

        Arguments
        ---------
        observation_space : Space
            Environment's observation space.
        action_space : Space
            Environment's action space.
        minibatch_size : int
            Size of minibatches for training on rollout data.
        num_ppo_epochs : int
            Number of training steps of surrogate loss for each rollout.
        lr : float
            Learning rate.
        eps : float
            Epsilon value used for numerical stability. Usually 1e-8.
        value_loss_coeff : float
            Coefficient for value loss in training objective.
        entropy_loss_coeff : float
            Coefficient for entropy loss in training objective.
        gamma : float
            Discount factor.
        gae_lambda : float
            Lambda parameter for GAE (used in equation (11) in PPO paper).
        clip_param : float
            Clipping parameter for PPO surrogate loss.
        max_grad_norm : float
            Maximum norm of loss gradients for update.
        clip_value_loss : float
            Whether or not to clip the value loss.
        num_layers : int
            Number of layers in actor/critic network.
        hidden_size : int
            Hidden size of actor/critic network.
        normalize_advantages : float
            Whether or not to normalize advantages.
        """

        # Set policy state.
        self.observation_space = observation_space
        self.action_space = action_space
        self.minibatch_size = minibatch_size
        self.num_ppo_epochs = num_ppo_epochs
        self.lr = lr
        self.eps = eps
        self.value_loss_coeff = value_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.max_grad_norm = max_grad_norm
        self.clip_value_loss = clip_value_loss
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.normalize_advantages = normalize_advantages

        # Initialize policy network and optimizer.
        self.policy_network = PolicyNetwork(
            observation_space=observation_space,
            action_space=action_space,
            num_layers=num_layers,
            hidden_size=hidden_size,
        )
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr, eps=eps)

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Arguments
        ---------
        obs : torch.Tensor
            Observation to sample action from.

        Returns
        -------
        value_pred : torch.Tensor
            Value prediction from critic portion of policy.
        action : torch.Tensor
            Action sampled from distribution defined by policy output.
        action_log_prob : torch.Tensor
            Log probability of sampled action.
        """

        # Pass through network to get value prediction and action probabilities.
        value_pred, action_probs = self.policy_network(obs)

        # Create action distribution object from probabilities.
        if isinstance(self.action_space, Discrete):
            action_dist = Categorical(**action_probs)
            action = action_dist.sample().unsqueeze(-1)
            action_log_probs = (
                action_dist.log_prob(action.squeeze(-1))
                .view(action.size(0), -1)
                .sum(-1)
            )
        elif isinstance(self.action_space, Box):
            action_dist = Normal(**action_probs)
            action = action_dist.sample()
            action_log_probs = action_dist.log_prob(action).sum(-1, keepdim=True)
        else:
            raise ValueError("Action space '%r' unsupported." % type(self.action_space))

        return value_pred, action, action_log_probs

    def evaluate_actions(
        self, obs_batch: torch.Tensor, actions_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get values, log probabilities, and action distribution entropies for a batch of
        observations and actions.

        Arguments
        ---------
        obs_batch : torch.Tensor
            Observations to compute action distribution for.
        actions_batch : torch.Tensor
            Actions to compute log probability of under computed distribution.

        Returns
        -------
        value : torch.Tensor
            Value predictions of ``obs_batch`` under current policy.
        action_log_probs: torch.Tensor
            Log probabilities of ``actions_batch`` under action distributions resulting
            from ``obs_batch``.
        action_dist_entropy : torch.Tensor
            Entropies of action distributions resulting from ``obs_batch``.
        """

        value, action_probs = self.policy_network(obs_batch)

        # Create action distribution object from probabilities and compute log
        # probabilities.
        if isinstance(self.action_space, Discrete):
            action_dist = Categorical(**action_probs)
            action_log_probs = (
                action_dist.log_prob(actions_batch.squeeze(-1))
                .view(actions_batch.size(0), -1)
                .sum(-1)
                .unsqueeze(-1)
            )
        elif isinstance(self.action_space, Box):
            action_dist = Normal(**action_probs)
            action_log_probs = action_dist.log_prob(actions_batch).sum(-1, keepdim=True)
        else:
            raise ValueError("Action space '%r' unsupported." % type(self.action_space))

        action_dist_entropy = action_dist.entropy()

        return value, action_log_probs, action_dist_entropy

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get value prediction from an observation.

        Arguments
        ---------
        obs : torch.Tensor
            Observation to get value prediction for.

        Returns
        -------
        value_pred : torch.Tensor
            Value prediction from critic portion of policy.
        """

        value_pred, _ = self.policy_network(obs)
        return value_pred

    def compute_returns_advantages(
        self, individual_rollouts: List[RolloutStorage]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute returns corresponding to equations (11) and (12) in the PPO paper, for
        each rollout in ``invidial_rollouts``.
        """

        total_length = sum(rollout.step for rollout in individual_rollouts)
        returns = torch.zeros(total_length, 1)
        advantages = torch.zeros(total_length, 1)
        current_pos = 0

        for rollout in individual_rollouts:

            # Compute returns.
            with torch.no_grad():
                rollout.value_preds[rollout.step] = self.get_value(
                    rollout.obs[rollout.step]
                ).detach()
            gae = 0
            for t in reversed(range(rollout.step)):
                delta = rollout.rewards[t]
                if not (t == rollout.step - 1 and rollout.done):
                    delta += self.gamma * rollout.value_preds[t + 1]
                delta -= rollout.value_preds[t]

                gae = self.gamma * self.gae_lambda * gae + delta
                returns[t + current_pos] = gae + rollout.value_preds[t]

            # Compute advantages.
            end_pos = current_pos + rollout.step
            advantages[current_pos:end_pos] = (
                returns[current_pos:end_pos] - rollout.value_preds[: rollout.step]
            )

            current_pos += rollout.step

        # Normalize advantages.
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + self.eps
            )

        return returns, advantages

    def update(self, individual_rollouts: List[RolloutStorage]) -> Dict[str, float]:
        """
        Train policy with PPO from rollout information in ``individual rollouts``.

        Arguments
        ---------
        individual_rollouts : List[RolloutStorage]
            List of storage containers holding rollout information to train from. Each
            RolloutStorage object holds observation, action, reward, etc. from a single
            episode.

        Returns
        -------
        loss_items : Dict[str, float]
            Dictionary from loss names (e.g. action) to float loss values.
        """

        # Combine rollouts into one object and compute returns/advantages.
        rollouts = combine_rollouts(individual_rollouts)
        returns, advantages = self.compute_returns_advantages(individual_rollouts)

        # Run multiple training steps on surrogate loss.
        loss_names = ["action", "value", "entropy", "total"]
        loss_items = {loss_name: 0.0 for loss_name in loss_names}
        num_updates = 0
        for _ in range(self.num_ppo_epochs):
            data_generator = rollouts.feed_forward_generator(self.minibatch_size)

            for sample in data_generator:
                (
                    batch_indices,
                    obs_batch,
                    actions_batch,
                    value_preds_batch,
                    old_action_log_probs_batch,
                ) = sample
                returns_batch = returns.view(-1, 1)[batch_indices]
                advantages_batch = advantages.view(-1, 1)[batch_indices]

                # Reshape to do in a single forward pass for all steps
                (
                    values,
                    action_log_probs,
                    action_dist_entropy,
                ) = self.evaluate_actions(obs_batch, actions_batch)

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * advantages_batch
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                    * advantages_batch
                )
                action_loss = -torch.min(surr1, surr2).mean()

                if self.clip_value_loss:
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
                entropy_loss = action_dist_entropy.mean()

                # Optimizer step.
                self.optimizer.zero_grad()
                loss = (
                    value_loss * self.value_loss_coeff
                    + action_loss
                    - entropy_loss * self.entropy_loss_coeff
                )
                loss.backward()
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(
                        self.policy_network.parameters(), self.max_grad_norm
                    )
                self.optimizer.step()

                # Get loss values.
                loss_items["action"] += action_loss.item()
                loss_items["value"] += value_loss.item()
                loss_items["entropy"] += entropy_loss.item()
                loss_items["total"] += loss.item()
                num_updates += 1

        # Take average of loss values over all updates.
        for loss_name in loss_names:
            loss_items[loss_name] /= num_updates

        return loss_items
