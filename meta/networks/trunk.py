"""
Definition of MultiTaskTrunkNetwork, a module used to parameterize a multi-task network
with some number of shared layers at the beginning followed by a task-specific output
head for each task.
"""

from typing import Callable

import torch
import torch.nn as nn


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

        # Set device.
        self.device = device if device is not None else torch.device("cpu")

        # Generate network layers.
        self.initialize_network()

        # Move model to device.
        self.to(self.device)

    def initialize_network(self) -> None:
        """ Initialize layers of network. """

        # Initialize shared trunk.
        trunk_layers = []
        for i in range(self.num_shared_layers):

            # Calcuate input size of layer.
            layer_input_size = self.input_size if i == 0 else self.hidden_size

            # Initialize layer.
            trunk_layers.append(
                self.init_base(nn.Linear(layer_input_size, self.hidden_size))
            )

            # Activation functions.
            trunk_layers.append(nn.Tanh())

        self.trunk = nn.Sequential(*trunk_layers)

        # Initialize task-specific output heads for each task.
        heads_list = []
        for _ in range(self.num_tasks):

            task_layers = []
            for i in range(self.num_task_layers):

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
                task_layers.append(layer_init(nn.Linear(self.hidden_size, output_size)))

                # Activation function.
                if i != self.num_task_layers - 1:
                    task_layers.append(nn.Tanh())

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
