"""
Definition of MultiTaskTrunkNetwork, a module used to parameterize a multi-task network
with some number of shared layers at the beginning followed by a task-specific output
head for each task.
"""

from itertools import product
from typing import Callable

import torch
import torch.nn as nn

from meta.utils.estimate import RunningStats


class MultiTaskTrunkNetwork(nn.Module):
    """
    Module used to parameterize a multi-task network with a shared trunk. `init_base` is
    the initialization function used to initialize all layers except for the last, and
    `init_final` is the initialization function used to initialize the last layer. Note
    that `input_size` should be the size of the observation AFTER the one-hot task
    vector is appended to it.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        init_base: Callable[[nn.Module], nn.Module],
        init_final: Callable[[nn.Module], nn.Module],
        num_tasks: int,
        num_shared_layers: int = 3,
        num_task_layers: int = 1,
        hidden_size: int = 64,
        device: torch.device = None,
        measure_conflicting_grads: bool = False,
    ) -> None:

        super(MultiTaskTrunkNetwork, self).__init__()

        # Check number of layers.
        if num_shared_layers < 1 or num_task_layers < 1:
            raise ValueError(
                "Number of shared layers and task-specific layers in network should "
                "each be at least 1. Given values are: %d, %d"
                % (num_shared_layers, num_task_layers)
            )

        # Set state.
        self.input_size = input_size
        self.output_size = output_size
        self.init_base = init_base
        self.init_final = init_final
        self.num_tasks = num_tasks
        self.num_shared_layers = num_shared_layers
        self.num_task_layers = num_task_layers
        self.hidden_size = hidden_size
        self.measure_conflicting_grads = measure_conflicting_grads

        # Set device.
        self.device = device if device is not None else torch.device("cpu")

        # Generate network layers.
        self.initialize_network()

        if self.measure_conflicting_grads:

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

            layer_modules = []

            # Calcuate input size of layer.
            layer_input_size = self.input_size if i == 0 else self.hidden_size

            # Initialize layer.
            layer_modules.append(
                self.init_base(nn.Linear(layer_input_size, self.hidden_size))
            )

            # Activation functions.
            layer_modules.append(nn.Tanh())

            trunk_layers.append(nn.Sequential(*layer_modules))

        self.trunk = nn.Sequential(*trunk_layers)

        # Initialize task-specific output heads for each task.
        heads_list = []
        for _ in range(self.num_tasks):

            task_layers = []
            for i in range(self.num_task_layers):

                layer_modules = []

                # Calculate output size of layer.
                output_size = (
                    self.hidden_size
                    if i != self.num_task_layers - 1
                    else self.output_size
                )

                # Determine init function for layer.
                layer_init = (
                    self.init_base if i < self.num_task_layers - 1 else self.init_final
                )

                # Initialize layer.
                layer_modules.append(
                    layer_init(nn.Linear(self.hidden_size, output_size))
                )

                # Activation function.
                if i != self.num_task_layers - 1:
                    layer_modules.append(nn.Tanh())

                task_layers.append(nn.Sequential(*layer_modules))

            heads_list.append(nn.Sequential(*task_layers))

        self.output_heads = nn.ModuleList(heads_list)

    def forward(self, inputs: torch.Tensor, task_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass definition for MultiTaskTrunkNetwork.

        Arguments
        ---------
        inputs : torch.Tensor
            Inputs to trunk network. Each input should be a flat vector that ends with a
            one-hot vector (of length self.num_tasks) denoting the task index
            corresponding to the input. We do NOT explicitly check this condition so it
            is up to you to make sure it is true when using this method.

        task_indices : torch.Tensor
            Task index for each input in ``inputs`` as an integer.

        Returns
        -------
        outputs: torch.Tensor
            Output of shared trunk network when given ``inputs`` as input.
        """

        # Pass input through shared trunk.
        trunk_output = self.trunk(inputs)

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

            # Don't perform forward pass if there are no inputs for the current task.
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
