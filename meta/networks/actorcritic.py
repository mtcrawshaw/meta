"""
Definition of ActorCriticNetwork, a module used to parameterize a vanilla actor/critic
policy.
"""

from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.distributions import Distribution, Categorical, Normal
from gym.spaces import Space, Box, Discrete

from meta.networks.mlp import MLPNetwork
from meta.networks.recurrent import RecurrentBlock
from meta.networks.trunk import MultiTaskTrunkNetwork
from meta.networks.splitting import (
    MultiTaskSplittingNetworkV1,
    MultiTaskSplittingNetworkV2,
    MetaSplittingNetwork,
)
from meta.utils.utils import AddBias, get_space_size, get_space_shape


class ActorCriticNetwork(nn.Module):
    """ Module used to parameterize an actor/critic policy. """

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        num_processes: int,
        rollout_length: int,
        architecture_config: Dict[str, Any],
        device: torch.device = None,
    ) -> None:

        super(ActorCriticNetwork, self).__init__()

        # Set state.
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_processes = num_processes
        self.rollout_length = rollout_length
        self.device = device if device is not None else torch.device("cpu")
        self.architecture_type = architecture_config["type"]
        self.recurrent = architecture_config["recurrent"]
        self.recurrent_hidden_size = architecture_config["recurrent_hidden_size"]

        # Compute input and output sizes.
        self.input_size = get_space_size(observation_space)
        self.output_size = get_space_size(action_space)

        # Initialize network.
        self.initialize_network(architecture_config)

        # Move to device.
        self.to(device)

    def initialize_network(self, architecture_config: Dict[str, Any]) -> None:
        """
        Initialize pieces of the network. These are recurrent block (optional), actor,
        and critic networks.
        """

        # Initialize recurrent block, if necessary.
        if architecture_config["recurrent"]:

            # Correct recurrent input size if we should exclude task index. The line
            # where we change the observation shape to reflect the exclusion assumes
            # that len(observation.shape) == 1, since this is the only supported case
            # for the trunk and splitting architecture.
            input_size = self.input_size
            observation_shape = get_space_shape(self.observation_space, "obs")
            if (
                architecture_config["type"] in ["trunk", "splitting_v1", "splitting_v2"]
                and not architecture_config["include_task_index"]
            ):
                input_size -= architecture_config["num_tasks"]
                observation_shape = (
                    observation_shape[0] - architecture_config["num_tasks"],
                )

            self.recurrent_block = RecurrentBlock(
                input_size=input_size,
                hidden_size=self.recurrent_hidden_size,
                observation_shape=observation_shape,
                num_processes=self.num_processes,
                rollout_length=self.rollout_length,
                device=self.device,
            )

        # Initialize actor and critic networks.
        actor_kwargs = dict(architecture_config["actor_config"])
        critic_kwargs = dict(architecture_config["critic_config"])
        if architecture_config["type"] == "mlp":
            self.actor = MLPNetwork(
                input_size=self.input_size
                if not self.recurrent
                else self.recurrent_hidden_size,
                output_size=self.output_size,
                device=self.device,
                **actor_kwargs,
            )
            self.critic = MLPNetwork(
                input_size=self.input_size
                if not self.recurrent
                else self.recurrent_hidden_size,
                output_size=1,
                device=self.device,
                **critic_kwargs,
            )

            # Extra parameter vector for standard deviations in the case that
            # the policy distribution is Gaussian.
            if isinstance(self.action_space, Box):
                self.logstd = AddBias(torch.zeros(self.output_size))

        elif architecture_config["type"] in ["trunk", "splitting_v1", "splitting_v2"]:

            # We only support environments whose observation spaces are flat vectors.
            if (
                not isinstance(self.observation_space, Box)
                or len(self.observation_space.shape) != 1
            ):
                raise NotImplementedError
            self.num_tasks = architecture_config["num_tasks"]
            actor_kwargs["num_tasks"] = self.num_tasks
            critic_kwargs["num_tasks"] = self.num_tasks
            self.include_task_index = architecture_config["include_task_index"]

            # Correct input size if we should exclude task index from the input and the
            # architecture isn't recurrent. This is because: when we are excluding the
            # task input and we have a recurrent block at the beginning of the
            # architecture, we exclude the task index from the recurrent input, not the
            # input to the trunk/splitting network.
            input_size = self.input_size
            if self.recurrent:
                input_size = self.recurrent_hidden_size
            elif not self.include_task_index:
                input_size -= self.num_tasks

            if architecture_config["type"] == "trunk":
                cls = MultiTaskTrunkNetwork
            elif architecture_config["type"] == "splitting_v1":
                cls = MultiTaskSplittingNetworkV1
            elif architecture_config["type"] == "splitting_v2":
                cls = MultiTaskSplittingNetworkV2
            else:
                raise NotImplementedError

            self.actor = cls(
                input_size=input_size,
                output_size=self.output_size,
                device=self.device,
                **actor_kwargs,
            )
            self.critic = cls(
                input_size=input_size,
                output_size=1,
                device=self.device,
                **critic_kwargs,
            )

            # Extra parameter vectors (one for each task) for standard deviations in the
            # case that the policy distribution is Gaussian.
            if isinstance(self.action_space, Box):
                logstd_list = []
                for _ in range(architecture_config["num_tasks"]):
                    logstd_list.append(AddBias(torch.zeros(self.output_size)))
                self.output_logstd = nn.ModuleList(logstd_list)

        else:
            raise ValueError(
                "Unsupported architecture type: %s" % str(architecture_config["type"])
            )

    def forward(
        self, obs: torch.Tensor, hidden_state: torch.Tensor, done: torch.Tensor,
    ) -> Tuple[torch.Tensor, Distribution, torch.Tensor]:
        """
        Forward pass definition for ActorCriticNetwork.

        Arguments
        ---------
        obs : torch.Tensor
            Observation to be used as input to policy network. If the observation space
            is discrete, this function expects ``obs`` to be a one-hot vector.
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

        # Exclude task index from obs, if necessary.
        if (
            self.architecture_type
            in ["trunk", "splitting_v1", "splitting_v2", "meta_splitting"]
            and not self.include_task_index
        ):
            task_index_pos = self.input_size - self.num_tasks
            x = x[:, :task_index_pos]

        # Pass through recurrent layer, if necessary.
        if self.recurrent:
            x, hidden_state = self.recurrent_block(x, hidden_state, done)

        # Pass through actor and critic networks. We do this separately depending on the
        # architecture type, since the multi-task networks need the task index info
        # included in obs in order to feed each observation to the correct output head,
        # and this information isn't present in `x`.
        if self.architecture_type == "mlp":
            value_pred = self.critic(x)
            actor_output = self.actor(x)

        elif self.architecture_type in [
            "trunk",
            "splitting_v1",
            "splitting_v2",
            "meta_splitting",
        ]:
            task_index_pos = self.input_size - self.num_tasks
            task_indices = obs[:, task_index_pos:].nonzero()[:, 1]
            value_pred = self.critic(x, task_indices)
            actor_output = self.actor(x, task_indices)

        else:
            raise NotImplementedError

        # Construct action distribution from actor output.
        if isinstance(self.action_space, Discrete):
            action_dist = Categorical(logits=actor_output)
        elif isinstance(self.action_space, Box):

            # Use logstd to compute standard deviation of action distribution.
            if self.architecture_type == "mlp":
                action_logstd = self.logstd(
                    torch.zeros(actor_output.size(), device=self.device)
                )

            elif self.architecture_type in [
                "trunk",
                "splitting_v1",
                "splitting_v2",
                "meta_splitting",
            ]:

                # In the multi-task case, we have to do account for the fact that each
                # output head has its own copy of `logstd`.
                action_logstds = []
                logstd_shape = actor_output.shape[1:]
                for i in range(len(task_indices)):
                    task_index = task_indices[i]
                    action_logstds.append(
                        self.output_logstd[task_index](
                            torch.zeros(logstd_shape, device=self.device)
                        )
                    )
                action_logstd = torch.stack(action_logstds)

            else:
                raise NotImplementedError

            action_dist = Normal(loc=actor_output, scale=action_logstd.exp())

        else:
            raise NotImplementedError

        return value_pred, action_dist, hidden_state

    def meta_conversion(self, num_test_tasks: int) -> None:
        """
        Convert the actor and the critic models to their meta-learning counterparts. For
        all architectures besides splitting networks, this is not implemented. For
        splitting networks, we convert the actor and critic models from multi-task
        splitting networks to meta splitting networks.

        Arguments
        ---------
        num_test_tasks : int
            Number of tasks for the model to perform at meta-test time.
        """

        # Convert `self.actor` and `self.critic` from multi-task splitting networks into
        # meta-splitting networks.
        if self.architecture_type in ["splitting_v1", "splitting_v2"]:
            self.actor = MetaSplittingNetwork(
                self.actor, num_test_tasks, device=self.device
            )
            self.critic = MetaSplittingNetwork(
                self.critic, num_test_tasks, device=self.device
            )
            logstd_list = []
            for _ in range(self.num_tasks):
                logstd_list.append(AddBias(torch.zeros(self.output_size, device=self.device)))
            self.output_logstd = nn.ModuleList(logstd_list)

            self.architecture_type = "meta_splitting"

        else:
            raise NotImplementedError
