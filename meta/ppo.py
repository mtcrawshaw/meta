""" Definition of PPOPolicy, an object to perform acting and training with PPO. """

from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
from gym.spaces import Space, Box, Discrete

from meta.network import PolicyNetwork
from meta.storage import RolloutStorage
from meta.utils import combine_first_two_dims


class PPOPolicy:
    """ A policy class for PPO. """

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        num_minibatch: int,
        num_processes: int,
        rollout_length: int,
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
        recurrent: bool = False,
        normalize_advantages: float = True,
        device: torch.device = None,
    ) -> None:
        """
        init function for PPOPolicy.

        Arguments
        ---------
        observation_space : Space
            Environment's observation space.
        action_space : Space
            Environment's action space.
        num_minibatch : int
            Number of minibatches for training on rollout data.
        num_processes : int
            Number of processes to simultaneously gather training data.
        rollout_length: int
            Length of the rollout between each update.
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
        recurrent : bool
            Whether or not to add recurrent layer to policy network.
        normalize_advantages : float
            Whether or not to normalize advantages.
        device : torch.device
            Which device to perform update on (forward pass is always on CPU).
        """

        # Set policy state.
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_minibatch = num_minibatch
        self.num_processes = num_processes
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
        self.recurrent = recurrent
        self.hidden_size = hidden_size
        self.normalize_advantages = normalize_advantages
        self.device = device if device is not None else torch.device("cpu")
        self.train = True

        # Initialize policy network and optimizer.
        self.policy_network = PolicyNetwork(
            observation_space=observation_space,
            action_space=action_space,
            num_processes=num_processes,
            rollout_length=rollout_length,
            num_layers=num_layers,
            hidden_size=hidden_size,
            recurrent=recurrent,
            device=device,
        )
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr, eps=eps)

    def act(
        self, obs: torch.Tensor, hidden_state: torch.Tensor, done: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Arguments
        ---------
        obs : torch.Tensor
            Observation to sample action from.
        hidden_state : torch.Tensor
            Hidden state to use for recurrent layer of policy, if necessary.
        done : torch.Tensor
            Whether or not the previous environment step was terminal. We use this to
            clear the hidden state of the policy when necessary, if it is recurrent.

        Returns
        -------
        value_pred : torch.Tensor
            Value prediction from critic portion of policy.
        action : torch.Tensor
            Action sampled from distribution defined by policy output.
        action_log_prob : torch.Tensor
            Log probability of sampled action.
        hidden_state : torch.Tensor
            New hidden state after forward pass.
        """

        # Pass through network to get value prediction and action probabilities.
        value_pred, action_dist, hidden_state = self.policy_network(
            obs, hidden_state, done
        )

        # Sample action and compute log probabilities.
        if self.train:
            action = action_dist.sample()
        else:
            # If evaluating, select action with highest probability.
            if isinstance(action_dist, Categorical):
                action = action_dist.probs.argmax(dim=-1)
            elif isinstance(action_dist, Normal):
                action = action_dist.mean
            else:
                raise ValueError(
                    "Unsupported distribution type '%s'." % type(action_dist)
                )
        action_log_prob = action_dist.log_prob(action)

        # We sum over ``action_log_prob`` to convert element-wise log probs into a joint
        # log prob, if the action is a vector. Note that we only sum over the last
        # dimension, so if the action space has two or more dimensions then this will
        # probably fail.
        if isinstance(self.action_space, Box):
            action_log_prob = action_log_prob.sum(-1, keepdim=True)
        elif not isinstance(self.action_space, Discrete):
            raise ValueError("Action space '%r' unsupported." % type(self.action_space))

        # Keep sizes consistent.
        if action_log_prob.shape[1:] == torch.Size([]):
            action_log_prob = action_log_prob.view(-1, 1)

        return value_pred, action, action_log_prob, hidden_state

    def evaluate_actions(
        self,
        obs_batch: torch.Tensor,
        hidden_states_batch: torch.Tensor,
        actions_batch: torch.Tensor,
        dones_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get values, log probabilities, and action distribution entropies for a batch of
        observations and actions.

        Arguments
        ---------
        obs_batch : torch.Tensor
            Observations to compute action distribution for.
        actions_batch : torch.Tensor
            Actions to compute log probability of under computed distribution.
        hidden_states_batch : torch.Tensor
            Hidden states to use for recurrent layer of policy, if necessary.
        dones_batch : torch.Tensor
            Whether or not each environment step in the batch is terminal.

        Returns
        -------
        value : torch.Tensor
            Value predictions of ``obs_batch`` under current policy.
        action_log_probs: torch.Tensor
            Log probabilities of ``actions_batch`` under action distributions resulting
            from ``obs_batch``.
        action_dist_entropy : torch.Tensor
            Entropies of action distributions resulting from ``obs_batch``.
        hidden_states_batch : torch.Tensor
            Updated hidden states after forward pass.
        """

        value, action_dist, hidden_states_batch = self.policy_network(
            obs_batch, hidden_states_batch, dones_batch
        )

        # Create action distribution object from probabilities and compute log
        # probabilities.
        if isinstance(self.action_space, Discrete):
            action_log_probs = action_dist.log_prob(actions_batch.squeeze(-1))
        elif isinstance(self.action_space, Box):
            action_log_probs = action_dist.log_prob(actions_batch).sum(-1)
        else:
            raise ValueError("Action space '%r' unsupported." % type(self.action_space))

        action_dist_entropy = action_dist.entropy()

        # This is to account for the fact that value has one more dimension than
        # value_preds_batch in update(), since value is generated from input obs_batch,
        # which has one more dimension than obs, which is the input that generated
        # value_preds_batch.
        value = torch.squeeze(value, -1)

        return value, action_log_probs, action_dist_entropy, hidden_states_batch

    def get_value(
        self, obs: torch.Tensor, hidden_state: torch.Tensor, done: torch.Tensor
    ) -> torch.Tensor:
        """
        Get value prediction from an observation.

        Arguments
        ---------
        obs : torch.Tensor
            Observation to get value prediction for.
        hidden_state : torch.Tensor
            Hidden state to use for recurrent layer of policy, if necessary.
        done : torch.Tensor
            Whether or not the previous environment step was terminal. We use this to
            clear the hidden state when necessary.

        Returns
        -------
        value_pred : torch.Tensor
            Value prediction from critic portion of policy.
        """

        value_pred, _, _ = self.policy_network(obs, hidden_state, done)
        return value_pred

    def compute_returns_advantages(
        self, rollout: RolloutStorage
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute returns corresponding to equations (11) and (12) in the PPO paper, for
        each episode in ``rollouts``.

        Parameters
        ----------
        rollout : RolloutStorage
            Storage container for rollout information.

        Returns
        -------
        returns : torch.Tensor
            Tensor holding returns.
        advantages : torch.Tensor
            Tensor holding advantage estimates using GAE.
        """

        returns = torch.zeros(
            rollout.rollout_step, rollout.num_processes, 1, device=self.device
        )
        advantages = torch.zeros(
            rollout.rollout_step, rollout.num_processes, 1, device=self.device
        )

        # Get value prediction of very last observation for return computation.
        with torch.no_grad():
            rollout.value_preds[rollout.rollout_step] = self.get_value(
                rollout.obs[rollout.rollout_step],
                rollout.hidden_states[rollout.rollout_step],
                rollout.dones[rollout.rollout_step],
            )

        # Compute returns.
        gae = 0
        for t in reversed(range(rollout.rollout_step)):

            # We initially set delta = 0 to avoid setting delta = rollout.rewards[t],
            # and coupling delta with rollout.rewards[t].
            delta = 0
            delta += rollout.rewards[t]
            delta += (
                self.gamma * rollout.value_preds[t + 1] * (1 - rollout.dones[t + 1])
            )
            delta -= rollout.value_preds[t]

            gae = (
                1 - rollout.dones[t + 1]
            ) * self.gamma * self.gae_lambda * gae + delta
            returns[t] = gae + rollout.value_preds[t]

        # Compute advantages.
        advantages = returns - rollout.value_preds[: rollout.rollout_step]

        # Normalize advantages if necessary.
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + self.eps
            )

        # Reshape returns and advantages. In the feedforward case, we
        # completely flatten returns and advantages to match the rest of the data.
        # In the recurrent case, we have to preserve the temporal dimension.
        if not self.recurrent:
            returns = returns.view(rollout.rollout_step * rollout.num_processes)
            advantages = advantages.view(rollout.rollout_step * rollout.num_processes)

        return returns, advantages

    def update(self, rollout: RolloutStorage) -> Dict[str, float]:
        """
        Train policy with PPO from rollout information in ``rollouts``.

        Arguments
        ---------
        rollouts : RolloutStorage
            Storage container holding rollout information to train from. Contains
            observations, actions, rewards, etc. for one or more episodes.

        Returns
        -------
        loss_items : Dict[str, float]
            Dictionary from loss names (e.g. action) to float loss values.
        """

        # Combine rollouts into one object and compute returns/advantages.
        returns, advantages = self.compute_returns_advantages(rollout)

        # Run multiple training steps on surrogate loss.
        loss_names = ["action", "value", "entropy", "total"]
        loss_items = {loss_name: 0.0 for loss_name in loss_names}
        num_updates = 0
        for _ in range(self.num_ppo_epochs):

            # Set minibatch generator based on whether or not are training a recurrent
            # policy.
            if self.recurrent:
                minibatch_generator = rollout.recurrent_minibatch_generator(
                    self.num_minibatch
                )
            else:
                minibatch_generator = rollout.feedforward_minibatch_generator(
                    self.num_minibatch
                )

            for minibatch in minibatch_generator:

                # Get batch of rollout data and construct corresponding batch of returns
                # and advantages. Since returns and advantages still have a temporal
                # dimension in the recurrent case, we have to do some more manipulation
                # to construct their batches. They are each of size (T * N), where T is
                # the length of each rollout and N is the number of trajectories per
                # batch.
                (
                    batch_indices,
                    obs_batch,
                    value_preds_batch,
                    actions_batch,
                    old_action_log_probs_batch,
                    dones_batch,
                    hidden_states_batch,
                ) = minibatch
                if self.recurrent:
                    returns_batch = combine_first_two_dims(
                        returns[:, batch_indices]
                    ).squeeze(-1)
                    advantages_batch = combine_first_two_dims(
                        advantages[:, batch_indices]
                    ).squeeze(-1)
                else:
                    returns_batch = returns[batch_indices]
                    advantages_batch = advantages[batch_indices]

                # Compute new values, action log probs, and dist entropies.
                (
                    values_batch,
                    action_log_probs_batch,
                    action_dist_entropy_batch,
                    _,
                ) = self.evaluate_actions(
                    obs_batch, hidden_states_batch, actions_batch, dones_batch
                )

                values_batch = values_batch.squeeze(-1)

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

                # Backward pass.
                self.optimizer.zero_grad()
                loss = -(
                    action_loss
                    - self.value_loss_coeff * value_loss
                    + self.entropy_loss_coeff * entropy_loss
                )
                loss.backward()

                # Clip gradient.
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(
                        self.policy_network.parameters(), self.max_grad_norm
                    )

                # Optimizer step.
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
