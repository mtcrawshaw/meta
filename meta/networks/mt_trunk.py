"""
Definition of MultiTaskTrunkNetwork, a module used to parameterize a multi-task
actor/critic policy with some number of shared layers at the beginning followed by a
task-specific output head for each task.
"""

from typing import Tuple, List

import torch
import torch.nn as nn
from torch.distributions import Distribution, Categorical, Normal
import numpy as np
from gym.spaces import Space, Box, Discrete

from meta.networks.base import BaseNetwork, init_base, init_final, init_recurrent
from meta.utils.utils import AddBias


class MultiTaskTrunkNetwork(BaseNetwork):
    """
    Module used to parameterize a multi-task actor/critic policy with a shared trunk.
    """

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        num_processes: int,
        rollout_length: int,
        num_tasks: int,
        num_shared_layers: int = 3,
        num_task_layers: int = 1,
        hidden_size: int = 64,
        recurrent: bool = False,
        device: torch.device = None,
    ) -> None:

        self.num_tasks = num_tasks
        self.num_shared_layers = num_shared_layers
        self.num_task_layers = num_task_layers
        if self.num_shared_layers < 1 or self.num_task_layers < 1:
            raise ValueError(
                "Number of shared layers and task-specific layers in network should "
                "each be at least 1. Given values are: %d, %d"
                % (self.num_shared_layers, self.num_task_layers)
            )

        # We only support environments whose observation spaces are flat vectors.
        if len(observation_space.shape) != 1:
            raise NotImplementedError

        super(MultiTaskTrunkNetwork, self).__init__(
            observation_space=observation_space,
            action_space=action_space,
            num_processes=num_processes,
            rollout_length=rollout_length,
            hidden_size=hidden_size,
            recurrent=recurrent,
            device=device,
        )

    def initialize_network(self) -> None:
        """ Initialize layers of network. """

        # Initialize recurrent layer, if necessary.
        if self.recurrent:
            self.gru = init_recurrent(nn.GRU(self.input_size, self.hidden_size))
            self.hidden_state = torch.zeros(self.hidden_size)

        # Initialize shared trunk of actor and critic.
        actor_trunk_layers = []
        critic_trunk_layers = []
        for i in range(self.num_shared_layers):

            # Calcuate input size of layer.
            layer_input_size = (
                self.input_size if i == 0 and not self.recurrent else self.hidden_size
            )

            # Initialize each layer.
            actor_trunk_layers.append(
                init_base(nn.Linear(layer_input_size, self.hidden_size))
            )
            critic_trunk_layers.append(
                init_base(nn.Linear(layer_input_size, self.hidden_size))
            )

            # Activation functions.
            actor_trunk_layers.append(nn.Tanh())
            critic_trunk_layers.append(nn.Tanh())

        self.actor_trunk = nn.Sequential(*actor_trunk_layers)
        self.critic_trunk = nn.Sequential(*critic_trunk_layers)

        # Initialize task-specific output heads for each task.
        self.actor_output_heads = []
        self.critic_output_heads = []
        for task in range(self.num_tasks):

            actor_task_layers = []
            critic_task_layers = []
            for i in range(self.num_task_layers):

                # Calculate output size of actor/critic layers.
                actor_output_size = (
                    self.hidden_size
                    if i != self.num_task_layers - 1
                    else self.output_size
                )
                critic_output_size = (
                    self.hidden_size if i != self.num_task_layers - 1 else 1
                )

                # Initialize each layer. We initialize the last layer of the actor with
                # a different init function (see meta/networks/base.py).
                init_actor = init_final if i == self.num_task_layers - 1 else init_base
                actor_task_layers.append(
                    init_actor(nn.Linear(self.hidden_size, actor_output_size))
                )
                critic_task_layers.append(
                    init_base(nn.Linear(self.hidden_size, critic_output_size))
                )

                # Activation functions.
                if i != self.num_task_layers - 1:
                    actor_task_layers.append(nn.Tanh())
                    critic_task_layers.append(nn.Tanh())

            self.actor_output_heads.append(nn.Sequential(*actor_task_layers))
            self.critic_output_heads.append(nn.Sequential(*critic_task_layers))

        # Extra parameter vector for standard deviations in the case that
        # the policy distribution is Gaussian.
        if isinstance(self.action_space, Box):
            self.output_logstd = []
            for task in range(self.num_tasks):
                self.output_logstd.append(AddBias(torch.zeros(self.output_size)))

    def forward(
        self, obs: torch.Tensor, hidden_state: torch.Tensor, done: torch.Tensor,
    ) -> Tuple[torch.Tensor, Distribution, torch.Tensor]:
        """
        Forward pass definition for MultiTaskTrunkNetwork. It is expected that each
        observation is the concatenation of an environment observation with a one-hot
        vector denoting the index of the task that the observation came from, that each
        observation is a flat vector, and that the last ``self.num_tasks`` elements of
        this vector make up the one-hot task vector.

        Arguments
        ---------
        obs : torch.Tensor
            Observation to be used as input to policy network. If the observation space
            is discrete, this function expects the environment portion of ``obs`` to be
            a one-hot vector.
        hidden_state : torch.Tensor
            Hidden state to use for recurrent layer, if necessary.
        done : torch.Tensor
            Whether or not the last step was a terminal step. We use this to clear the
            hidden state of the network when necessary, if it is recurrent.

        Returns
        -------
        value_pred : torch.Tensor
            Predicted value output from critic.
        action_dist : torch.distributions.Distribution
            Distribution over action space to sample from.
        hidden_state : torch.Tensor
            New hidden state after forward pass.
        """

        x = obs

        # Pass through recurrent layer, if necessary.
        if self.recurrent:
            x, hidden_state = self.recurrent_forward(x, hidden_state, done)

        # Get task indices from each observation. We take the one-hot task vector from
        # the end of each observation in the batch and aggregate the task indices. Here
        # we commented out the check that these vectors are actually one-hot, just to
        # save time.
        task_index_pos = self.observation_space.shape[0]
        nonzero_pos = obs[:, task_index_pos:].nonzero()
        # assert nonzero_pos[:, 0] == torch.arange(obs.shape[0], device="cuda")
        task_indices = nonzero_pos[:, 1]

        # Pass through shared trunks.
        actor_trunk_output = self.actor_trunk(x)
        critic_trunk_output = self.critic_trunk(x)

        # Pass each individual trunk output through its task-specific output head. This
        # is a naive implementation because we can actually batch the individual outputs
        # by task to speed things up. We will do this later.
        actor_outputs = []
        critic_outputs = []
        assert task_indices.shape[0] == actor_trunk_output.shape[0]
        assert task_indices.shape[0] == critic_trunk_output.shape[0]
        for i in range(task_indices.shape[0]):
            task_index = task_indices[i]
            actor_out = actor_trunk_output[i]
            critic_out = critic_trunk_output[i]
            actor_outputs.append(self.actor_output_heads[task_index](actor_out))
            critic_outputs.append(self.critic_output_heads[task_index](critic_out))
        actor_output = torch.stack(actor_outputs)
        critic_outputs = torch.stack(critic_outputs)

        # Construct action distribution from actor output.
        action_dist = self.get_action_distribution(actor_output)

        return value_pred, action_dist, hidden_state
