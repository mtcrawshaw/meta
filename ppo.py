from functools import reduce
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
from gym.spaces import Space, Box, Discrete

from storage import RolloutStorage
from utils import convert_to_tensor, init


class PPOPolicy:
    """ A policy class for PPO. """

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        num_ppo_epochs: int,
        lr: float,
        eps: float,
        value_loss_coeff: float,
        entropy_loss_coeff: float,
    ):
        """
        init function for PPOPolicy.

        Arguments
        ---------
        observation_space : Space
            Environment's observation space.
        action_space : Space
            Environment's action space.
        num_ppo_epochs : int
            Number of training steps of surrogate loss for each rollout.
        lr : float
            Learning rate.
        eps : float
            Epsilon value for Adam, used for numerical stability. Usually 1e-8.
        value_loss_coeff : float
            Coefficient for value loss in training objective.
        entropy_loss_coeff : float
            Coefficient for entropy loss in training objective.
        """

        # Set policy state.
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_ppo_epochs = num_ppo_epochs
        self.lr = lr
        self.eps = eps
        self.value_loss_coeff = value_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff

        # Instantiate policy network and optimizer.
        self.policy_network = PolicyNetwork(observation_space, action_space)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr, eps=eps)

    def act(self, obs: Union[np.ndarray, int, float]):
        """
        Sample action from policy.

        Arguments
        ---------
        obs: np.ndarray or int or float
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
        element_log_probs = action_dist.log_prob(action)
        action_log_prob = element_log_probs.sum(-1)

        return value_pred, action, action_log_prob

    def update(self, rollouts: RolloutStorage):
        """
        Train policy with PPO from rollout information in ``rollouts``.

        Arguments
        ---------
        rollouts : RolloutStorage
            Storage container holding rollout information to train from.
        """

        # TODO: Compute advantages.
        advantages = None

        # Run multiple training steps on surrogate loss.
        for _ in range(self.num_ppo_epochs):

            # TODO: Sample data from rollouts.

            # TODO: Compute value loss, action loss, and entropy loss.
            value_loss = torch.Tensor([0.0])
            action_loss = torch.Tensor([0.0])
            entropy_loss = torch.Tensor([0.0])

            # Optimizer step
            self.optimizer.zero_grad()
            loss = -(
                action_loss
                - self.value_loss_coeff * value_loss
                + self.entropy_loss_coeff * entropy_loss
            )
            # loss.backward()
            # self.optimizer.step()


class PolicyNetwork(nn.Module):
    """ MLP network parameterizing the policy. """

    def __init__(
        self, observation_space: Space, action_space: Space, hidden_size: int = 64
    ):
        """
        init function for PolicyNetwork.

        Arguments
        ---------
        observation_space : Space
            Environment's observation space.
        action_space : Space
            Environment's action space.
        hidden_size : int
        """

        super(PolicyNetwork, self).__init__()
        self.action_space = action_space
        self.observation_space = observation_space

        # Calculate the input/output size.
        self.input_size = get_space_size(observation_space)
        self.output_size = get_space_size(action_space)

        # Instantiate modules.
        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
        )

        self.hidden_size = hidden_size
        self.actor = nn.Sequential(
            init_(nn.Linear(self.input_size, self.hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(self.hidden_size, self.hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(self.hidden_size, self.output_size)),
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(self.input_size, self.hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(self.hidden_size, self.hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(self.hidden_size, 1)),
        )

        # Extra parameter vector for standard deviations in the case that
        # the policy distribution is Gaussian.
        if isinstance(action_space, Box):
            self.logstd = torch.zeros(self.output_size)

    def forward(self, obs: torch.Tensor):
        """
        Forward pass definition for PolicyNetwork.

        Arguments
        ---------
        obs : torch.Tensor
            Observation to be used as input to policy network.

        Returns
        -------
        value_pred : torch.Tensor
            Predicted value output from critic.
        action_probs : Dict[str, torch.Tensor]
            Parameterization of policy distribution. Keys and values match
            argument structure for init functions of torch.Distribution.
        """

        value_pred = self.critic(obs)
        actor_output = self.actor(obs)

        if isinstance(self.action_space, Discrete):
            # Matches torch.distribution.Categorical
            action_probs = {"logits": actor_output}
        elif isinstance(self.action_space, Box):
            # Matches torch.distribution.Normal
            action_probs = {"loc": actor_output, "scale": self.logstd.exp()}

        return value_pred, action_probs


def get_space_size(space: Space):
    """ Get the size of a gym.spaces Space. """

    if isinstance(space, Discrete):
        size = 1
    elif isinstance(space, Box):
        size = reduce(lambda a, b: a * b, space.shape)
    else:
        raise ValueError("Unsupported space type: %s." % type(space))

    return size
