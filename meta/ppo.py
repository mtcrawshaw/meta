from functools import reduce
from typing import Union, Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
from gym.spaces import Space, Box, Discrete

from meta.network import PolicyNetwork
from meta.storage import RolloutStorage
from meta.utils import convert_to_tensor, init


class PPOPolicy:
    """ A policy class for PPO. """

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        rollout_length: int,
        num_ppo_epochs: int,
        lr: float,
        eps: float,
        value_loss_coeff: float,
        entropy_loss_coeff: float,
        gamma: float,
        gae_lambda: float,
        minibatch_size: int,
        clip_param: float,
        max_grad_norm: float,
        clip_value_loss: bool,
        num_layers: int,
        hidden_size: int,
        normalize_advantages: float,
    ):
        """
        init function for PPOPolicy.

        Arguments
        ---------
        observation_space : Space
            Environment's observation space.
        action_space : Space
            Environment's action space.
        rollout_length: int
            Total number of environent timesteps (possibly over multiple episodes) per
            update.
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
        minibatch_size : int
            Size of minibatches for training on rollout data.
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
        self.rollout_length = rollout_length
        self.num_ppo_epochs = num_ppo_epochs
        self.lr = lr
        self.eps = eps
        self.value_loss_coeff = value_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.minibatch_size = minibatch_size
        self.clip_param = clip_param
        self.max_grad_norm = max_grad_norm
        self.clip_value_loss = clip_value_loss
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.normalize_advantages = normalize_advantages

        # Instantiate policy network and optimizer.
        self.policy_network = PolicyNetwork(
            observation_space,
            action_space,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
        )
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr, eps=eps)

    def act(self, obs: Union[np.ndarray, int, float]):
        """
        Sample action from policy.

        Arguments
        ---------
        obs : np.ndarray or int or float
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
        tensor_obs = convert_to_tensor(obs)
        value_pred, action_probs = self.policy_network(tensor_obs)

        # Create action distribution object from probabilities.
        if isinstance(self.action_space, Discrete):
            action_dist = Categorical(**action_probs)
        elif isinstance(self.action_space, Box):
            action_dist = Normal(**action_probs)
        else:
            raise ValueError("Action space '%r' unsupported." % type(self.action_space))

        # Sample action.
        action = action_dist.sample()

        # Compute log probability of action. We sum over ``element_log_probs``
        # to convert element-wise log probs into a joint log prob.
        action_log_prob = action_dist.log_prob(action)
        # action_log_prob = action_log_prob.sum(-1)

        return value_pred, action, action_log_prob

    def evaluate_actions(self, obs_batch: torch.Tensor, actions_batch: torch.Tensor):
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

        # Create action distribution object from probabilities.
        if isinstance(self.action_space, Discrete):
            action_dist = Categorical(**action_probs)
        elif isinstance(self.action_space, Box):
            action_dist = Normal(**action_probs)
        else:
            raise ValueError("Action space '%r' unsupported." % type(self.action_space))

        # Compute log probabilities and action distribution entropies. We sum over
        # ``element_log_probs`` here to convert element-wise log probs into a joint
        # log prob.
        actions_batch = torch.squeeze(actions_batch)
        value = torch.squeeze(value)
        action_log_probs = action_dist.log_prob(actions_batch)
        action_dist_entropy = action_dist.entropy()

        return value, action_log_probs, action_dist_entropy

    def get_value(self, obs: Union[np.ndarray, int, float]):
        """
        Get value prediction from an observation.

        Arguments
        ---------
        obs : np.ndarray or int or float
            Observation to get value prediction for.

        Returns
        -------
        value_pred : torch.Tensor
            Value prediction from critic portion of policy.
        """

        tensor_obs = convert_to_tensor(obs)
        value_pred, _ = self.policy_network(tensor_obs)
        return value_pred

    def update(self, individual_rollouts: List[RolloutStorage]):
        """
        Train policy with PPO from rollout information in ``rollouts``.

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

        if len(individual_rollouts) == 0:
            raise ValueError("Received empty list of rollouts.")

        # Compute returns corresponding to equations (11) and (12) in the PPO paper, for
        # each rollout in ``rollouts``, and build up one large RolloutStorage object for
        # all episodes.
        rollouts = RolloutStorage(
            rollout_length=self.rollout_length,
            observation_space=self.observation_space,
            action_space=self.action_space,
        )
        returns = torch.zeros(self.rollout_length)
        advantages = torch.zeros(self.rollout_length)
        current_pos = 0
        for rollout in individual_rollouts:

            # Compute returns.
            with torch.no_grad():
                rollout.value_preds[rollout.rollout_step] = self.get_value(
                    rollout.obs[rollout.rollout_step]
                )
            gae = 0
            for t in reversed(range(rollout.rollout_step)):
                delta = (
                    rollout.rewards[t]
                    + self.gamma * rollout.value_preds[t + 1]
                    - rollout.value_preds[t]
                )
                gae = self.gamma * self.gae_lambda * gae + delta
                returns[t + current_pos] = gae + rollout.value_preds[t]

            # Compute advantages.
            end_pos = current_pos + rollout.rollout_step
            advantages[current_pos:end_pos] = (
                returns[current_pos:end_pos]
                - rollout.value_preds[: rollout.rollout_step]
            )

            # Copy episode into combined rollout object.
            rollouts.insert_rollout(rollout, current_pos)
            current_pos += rollout.rollout_step

        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + self.eps
            )

        # Run multiple training steps on surrogate loss.
        loss_names = ["action", "value", "entropy", "total"]
        loss_items = {loss_name: 0.0 for loss_name in loss_names}
        num_updates = 0
        for _ in range(self.num_ppo_epochs):

            minibatch_generator = rollouts.minibatch_generator(self.minibatch_size)
            for minibatch in minibatch_generator:

                # Get batch of rollout data and construct corresponding batch of
                # returns and advantages.
                (
                    batch_indices,
                    obs_batch,
                    value_preds_batch,
                    actions_batch,
                    old_action_log_probs_batch,
                    rewards_batch,
                ) = minibatch
                returns_batch = returns[batch_indices]
                advantages_batch = advantages[batch_indices]

                # Compute new values, action log probs, and dist entropies.
                (
                    values_batch,
                    action_log_probs_batch,
                    action_dist_entropy_batch,
                ) = self.evaluate_actions(obs_batch, actions_batch)

                # Compute action loss, value loss, and entropy loss.
                ratio = torch.exp(action_log_probs_batch - old_action_log_probs_batch)
                surrogate1 = ratio * advantages_batch
                surrogate2 = (
                    torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                    * advantages_batch
                )
                action_loss = torch.min(surrogate1, surrogate2).mean()
                if self.clip_value_loss:
                    value_losses = (returns_batch - values_batch).pow(2)
                    clipped_value_preds = value_preds_batch + torch.clamp(
                        values_batch - value_preds_batch,
                        -self.clip_param,
                        self.clip_param,
                    )
                    clipped_value_losses = (returns_batch - clipped_value_preds).pow(2)
                    value_loss = (
                        0.5 * torch.max(value_losses, clipped_value_losses).mean()
                    )
                else:
                    value_loss = 0.5 * (returns_batch - values_batch).pow(2).mean()
                entropy_loss = action_dist_entropy_batch.mean()

                # Optimizer step.
                self.optimizer.zero_grad()
                loss = -(
                    action_loss
                    - self.value_loss_coeff * value_loss
                    + self.entropy_loss_coeff * entropy_loss
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