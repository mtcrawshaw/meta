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
        actor_heads_list = []
        critic_heads_list = []
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

            actor_heads_list.append(nn.Sequential(*actor_task_layers))
            critic_heads_list.append(nn.Sequential(*critic_task_layers))

        self.actor_output_heads = nn.ModuleList(actor_heads_list)
        self.critic_output_heads = nn.ModuleList(critic_heads_list)

        # Extra parameter vector for standard deviations in the case that
        # the policy distribution is Gaussian.
        if isinstance(self.action_space, Box):
            logstd_list = []
            for task in range(self.num_tasks):
                logstd_list.append(AddBias(torch.zeros(self.output_size)))
            self.output_logstd = nn.ModuleList(logstd_list)

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
        task_index_pos = self.observation_space.shape[0] - self.num_tasks
        nonzero_pos = obs[:, task_index_pos:].nonzero()
        # assert nonzero_pos[:, 0] == torch.arange(obs.shape[0], device="cuda")
        task_indices = nonzero_pos[:, 1]

        # Pass through shared trunks.
        actor_trunk_output = self.actor_trunk(x)
        critic_trunk_output = self.critic_trunk(x)

        # To forward pass through the output heads, we first partition the inputs by
        # their task, so that we can perform only one forward pass through each
        # task-specific output head.
        task_batch_indices = [
            (task_indices == task).nonzero().squeeze(-1)
            for task in range(self.num_tasks)
        ]
        actor_batch_outputs = []
        critic_batch_outputs = []
        for task in range(self.num_tasks):

            # Construct the batch of trunk outputs for a given task head.
            single_batch_indices = task_batch_indices[task]

            if len(single_batch_indices) == 0:

                # Don't perform forward pass if there are no inputs for the current
                # task.
                actor_batch_outputs.append(torch.Tensor([0.0]))
                critic_batch_outputs.append(torch.Tensor([0.0]))
                continue

            # Pass batch of trunk outputs through task head.
            actor_batch = actor_trunk_output[single_batch_indices]
            critic_batch = critic_trunk_output[single_batch_indices]
            actor_batch_outputs.append(self.actor_output_heads[task](actor_batch))
            critic_batch_outputs.append(self.critic_output_heads[task](critic_batch))

        # Reconstruct batched outputs into a single tensor each for actor/critic.
        actor_outputs = []
        critic_outputs = []
        batch_counters = [0 for _ in range(self.num_tasks)]
        for task in task_indices:
            batch_counter = batch_counters[task]
            actor_outputs.append(actor_batch_outputs[task][batch_counter])
            critic_outputs.append(critic_batch_outputs[task][batch_counter])
            batch_counters[task] += 1
        actor_output = torch.stack(actor_outputs)
        value_pred = torch.stack(critic_outputs)

        # Construct action distribution from actor outputs.
        if isinstance(self.action_space, Discrete):
            action_dist = Categorical(logits=actor_output)
        elif isinstance(self.action_space, Box):
            action_logstds = []
            logstd_shape = actor_output.shape[1:]
            for i in range(task_indices.shape[0]):
                task_index = task_indices[i]
                action_logstds.append(
                    self.output_logstd[task_index](
                        torch.zeros(logstd_shape, device=self.device)
                    )
                )
            action_logstd = torch.stack(action_logstds)
            action_dist = Normal(loc=actor_output, scale=action_logstd.exp())
        else:
            raise NotImplementedError

        return value_pred, action_dist, hidden_state
