"""
Definition of MultiTaskTrunkNetwork, a module used to parameterize a multi-task network
with some number of shared layers at the beginning followed by a task-specific output
head for each task.
"""

from itertools import product
from typing import Iterator, Union, List

import torch
import torch.nn as nn

from meta.networks.utils import get_fc_layer, init_downscale, init_base, Parallel
from meta.utils.estimate import RunningStats


class MultiTaskTrunkNetwork(nn.Module):
    """ Module used to parameterize a multi-task network with a shared trunk. """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_tasks: int,
        activation: str = "tanh",
        num_shared_layers: int = 3,
        num_task_layers: int = 1,
        hidden_size: Union[int, List[int]] = 64,
        downscale_last_layer: bool = False,
        parallel_branches: bool = False,
        device: torch.device = None,
        monitor_grads: bool = False,
    ) -> None:
        """
        Init function for MultiTaskTrunkNetwork.

        Parameters
        ----------
        input_size : int,
            Input size of network. When `parallel_branches=False`, this size should
            include the size of the one-hot task vector appended to observations.
        output_size : int,
            Output size of network.
        num_tasks : int,
            Number of tasks. The network will contain a number of task-specific branches
            equal to `num_tasks`.
        activation : str
            Activation function for each layer besides the last layer,
            which has no activation.
        num_shared_layers : int
            Number of layers in the shared trunk portion.
        num_task_layers : int
            Number of layers in each task-specific branch.
        hidden_size : Union[int, List[int]]
            Number of units in each hidden layer. This can either be specified with a
            single integer, in which case all hidden layers will have the same size, or
            with a list of integers denoting the hidden sizes of each individual layer.
            The length of this list must be equal to `num_shared_layers +
            num_task_layers - 1`.
        downscale_last_layer : bool
            Whether or not to downscale the variance of initialized weights in the last
            layer of the network. This is suggested in https://arxiv.org/abs/2006.05990.
        parallel_branches : bool
            If `True`, each input will be passed through each task-specific branch, as
            is common in multi-task computer vision. If `False`, each input will only be
            routed through the branch corresponding to that input's task, as is common
            in multi-task RL. Default is `False`.
        device : torch.device
            Torch device to train on.
        monitor_grads : bool
            Whether or not to monitor the presence of conflicting gradients between
            tasks. This is older, haven't touched it in awhile and it may not be
            functioning.
        """

        super(MultiTaskTrunkNetwork, self).__init__()

        # Check number of layers.
        if num_shared_layers < 1 or num_task_layers < 1:
            raise ValueError(
                "Number of shared layers and task-specific layers in network should "
                "each be at least 1. Given values are: %d, %d"
                % (num_shared_layers, num_task_layers)
            )
        num_hidden_layers = num_shared_layers + num_task_layers - 1
        if isinstance(hidden_size, list):
            assert len(hidden_size) == num_hidden_layers

        # Set state.
        self.input_size = input_size
        self.output_size = output_size
        self.num_tasks = num_tasks
        self.activation = activation
        self.num_shared_layers = num_shared_layers
        self.num_task_layers = num_task_layers
        if isinstance(hidden_size, list):
            self.hidden_size = hidden_size
        else:
            assert isinstance(hidden_size, int)
            self.hidden_size = [hidden_size] * num_hidden_layers
        self.downscale_last_layer = downscale_last_layer
        self.parallel_branches = parallel_branches
        self.monitor_grads = monitor_grads

        # Set device.
        self.device = device if device is not None else torch.device("cpu")

        # Generate network layers.
        self.initialize_network()

        if self.monitor_grads:

            # Compute max shared layer size.
            self.shared_layer_sizes = [
                sum([param.nelement() for param in self.trunk[i].parameters()])
                for i in range(self.num_shared_layers)
            ]
            self.max_shared_layer_size = int(max(self.shared_layer_sizes))

            # Initialize running statistics for gradient conflicts between tasks.
            self.grad_conflict_stats = RunningStats(
                shape=(self.num_tasks, self.num_tasks, self.num_shared_layers),
                device=self.device,
            )
            self.layer_grad_conflicts = torch.zeros(
                self.num_shared_layers, device=self.device
            )

        # Move model to device.
        self.to(self.device)

    def initialize_network(self) -> None:
        """ Initialize layers of network. """

        # Initialize shared trunk.
        trunk_layers = []
        for i in range(self.num_shared_layers):

            # Calcuate input and output size of layer.
            layer_input_size = self.input_size if i == 0 else self.hidden_size[i - 1]
            layer_output_size = self.hidden_size[i]

            # Initialize layer.
            trunk_layers.append(
                get_fc_layer(
                    in_size=layer_input_size,
                    out_size=layer_output_size,
                    activation=self.activation,
                    layer_init=init_base,
                )
            )

        self.trunk = nn.Sequential(*trunk_layers)

        # Initialize task-specific output heads for each task.
        heads_list = []
        for _ in range(self.num_tasks):

            task_layers = []
            for i in range(self.num_task_layers):

                # Calculate input and output size of layer.
                layer_idx = i + self.num_shared_layers
                layer_input_size = self.hidden_size[layer_idx - 1]
                layer_output_size = (
                    self.hidden_size[layer_idx]
                    if i != self.num_task_layers - 1
                    else self.output_size
                )

                # Determine init function for layer.
                if i == self.num_task_layers - 1 and self.downscale_last_layer:
                    layer_init = init_downscale
                else:
                    layer_init = init_base

                # Initialize layer.
                task_layers.append(
                    get_fc_layer(
                        in_size=layer_input_size,
                        out_size=layer_output_size,
                        activation=self.activation
                        if i != self.num_task_layers - 1
                        else None,
                        layer_init=layer_init,
                    )
                )

            heads_list.append(nn.Sequential(*task_layers))

        # Construct task-specific heads, either in parallel or as individual modules.
        if self.parallel_branches:
            self.output_heads = Parallel(heads_list, combine_dim=1, new_dim=True)
        else:
            self.output_heads = nn.ModuleList(heads_list)

        # Save number of shared and task-specific parameters.
        self.num_shared_params = sum([p.numel() for p in self.trunk.parameters()])
        self.num_specific_params = {
            task: sum([p.numel() for p in self.output_heads[task].parameters()])
            for task in range(self.num_tasks)
        }

    def forward(
        self, inputs: torch.Tensor, task_indices: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass definition for MultiTaskTrunkNetwork.

        Arguments
        ---------
        inputs : torch.Tensor
            Inputs to trunk network.
        task_indices : torch.Tensor
            If `self.parallel_branches=False`, then `task_indices` should be a tensor of
            shape `(self.num_tasks)`, where `task_indices[i]` is an integer which
            denotes the task corresponding to `inputs[i]`. This input will only be
            routed through the corresonding task branch. We NOT explicitly check this
            condition so it is up to you to make sure that `task_indices` is correctly
            specified in this case. If `self.parallel_branches=True`, then
            `task_indices` is simply ignored.

        Returns
        -------
        outputs: torch.Tensor
            Output of shared trunk network when given `inputs` as input.
        """

        # Pass input through shared trunk.
        trunk_output = self.trunk(inputs)

        if self.parallel_branches:
            outputs = self.output_heads(trunk_output)

        else:

            # To forward pass through the output heads, we first partition the inputs by
            # their task, so that we can perform only one forward pass through each
            # task-specific output head.
            task_batch_indices = [
                (task_indices == task).nonzero().squeeze(-1)
                for task in range(self.num_tasks)
            ]
            batch_outputs = []
            for task in range(self.num_tasks):

                # Construct the batch of trunk outputs for a given task head.
                single_batch_indices = task_batch_indices[task]

                # Don't perform forward pass if there are no inputs for the current
                # task.
                if len(single_batch_indices) == 0:
                    batch_outputs.append(torch.Tensor([0.0]))
                    continue

                # Pass batch of trunk outputs through task head.
                task_batch = trunk_output[single_batch_indices]
                batch_outputs.append(self.output_heads[task](task_batch))

            # Reconstruct batched outputs into a single tensor.
            outputs_list = []
            batch_counters = [0 for _ in range(self.num_tasks)]
            for task in task_indices:
                batch_counter = batch_counters[task]
                outputs_list.append(batch_outputs[task][batch_counter])
                batch_counters[task] += 1
            outputs = torch.stack(outputs_list)

        return outputs

    def last_shared_params(self) -> Iterator[nn.Parameter]:
        """
        Return a list of the parameters of the last layer in `self` whose parameters are
        shared between multiple tasks.
        """
        return self.trunk[-1].parameters()

    def shared_params(self) -> Iterator[nn.parameter.Parameter]:
        """ Iterator over parameters which are shared between all tasks. """
        return self.trunk.parameters()

    def specific_params(self, task: int) -> Iterator[nn.parameter.Parameter]:
        """ Iterator over task-specific parameters. """
        return self.output_heads[task].parameters()

    def check_conflicting_grads(self, task_losses: torch.Tensor) -> None:
        """
        Determine whether there are conflicting gradients between the task losses at
        each shared layer. This is purely for observation and investigating the
        multi-task training dynamics.
        """

        # Compute task-specific gradients for the shared layers.
        task_grads = torch.zeros(
            (self.num_tasks, self.num_shared_layers, self.max_shared_layer_size),
            device=self.device,
        )
        for task in range(self.num_tasks):

            self.zero_grad()
            task_losses[task].backward(retain_graph=True)

            for layer in range(self.num_shared_layers):
                param_grad_list = []
                for param in self.trunk[layer].parameters():
                    param_grad_list.append(param.grad.view(-1))
                layer_grad = torch.cat(param_grad_list)
                task_grads[task, layer, : len(layer_grad)] = layer_grad

        self.measure_conflicts_from_grads(task_grads)

    def measure_conflicts_from_grads(self, task_grads: torch.Tensor) -> None:
        """
        Determine whether there are conflict gradients between the task losses at each
        shared layer, given the task-specific gradients.

        Arguments
        ---------
        task_grads : torch.Tensor
            Tensor of shape `(self.num_tasks, self.num_shared_layers,
            self.max_shared_layer_size)`, holds the task-specific gradients for each
            task at each layer.
        """

        # Get indices of tasks with non-zero gradients, i.e. the tasks that have data in
        # the batch from which the losses were computed, and pairs of tasks that both
        # have non-zero gradients.
        task_flags = (task_grads.reshape(self.num_tasks, -1) != 0.0).any(dim=1)
        task_pair_flags = task_flags.unsqueeze(0) * task_flags.unsqueeze(1)
        task_pair_flags = task_pair_flags.unsqueeze(-1)
        task_pair_flags = task_pair_flags.expand(-1, -1, self.num_shared_layers)

        # Compute whether gradients are conflicting between each pair of tasks at each
        # shared layer.
        conflict_flags = torch.zeros(
            (self.num_tasks, self.num_tasks, self.num_shared_layers),
            device=self.device,
        )
        for task1, task2 in product(range(self.num_tasks), range(self.num_tasks)):
            conflict_flags[task1, task2] = torch.sum(
                task_grads[task1] * task_grads[task2], dim=1
            )
            conflict_flags[task1, task2] = conflict_flags[task1, task2] < 0.0

        # Update running statistics measuring frequency of gradient conflicts.
        self.grad_conflict_stats.update(conflict_flags, task_pair_flags)
        self.layer_grad_conflicts = torch.zeros_like(self.layer_grad_conflicts)
        total_sample_sizes = torch.zeros(self.num_shared_layers, device=self.device)
        for task1 in range(self.num_tasks - 1):
            for task2 in range(task1 + 1, self.num_tasks):
                self.layer_grad_conflicts += (
                    self.grad_conflict_stats.mean[task1, task2]
                    * self.grad_conflict_stats.sample_size[task1, task2]
                )
                total_sample_sizes += self.grad_conflict_stats.sample_size[task1, task2]
        self.layer_grad_conflicts /= total_sample_sizes
