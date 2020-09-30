"""
Definition of SplittingMLPNetwork, a multi-layer perceptron splitting network.
"""

from copy import deepcopy
from typing import Callable, List

import torch
import torch.nn as nn


class SplittingMLPNetwork(nn.Module):
    """
    Module used to parameterize a splitting MLP. `init_base` is the initialization
    function used to initialize all layers except for the last, and `init_final` is the
    initialization function used to initialize the last layer.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        init_base: Callable[[nn.Module], nn.Module],
        init_final: Callable[[nn.Module], nn.Module],
        num_tasks: int,
        num_layers: int = 3,
        hidden_size: int = 64,
        device: torch.device = None,
    ) -> None:

        super(SplittingMLPNetwork, self).__init__()

        # Check number of layers.
        if num_layers < 1:
            raise ValueError(
                "Number of layers in network should be at least 1. Given value is: %d"
                % num_layers
            )

        # Set state.
        self.input_size = input_size
        self.output_size = output_size
        self.init_base = init_base
        self.init_final = init_final
        self.num_tasks = num_tasks
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Set device.
        self.device = device if device is not None else torch.device("cpu")

        # Generate network layers.
        self.initialize_network()

        # Move model to device.
        self.to(self.device)

    def initialize_network(self) -> None:
        """ Initialize layers of network. """

        # Initialize layers.
        region_list = []
        for i in range(self.num_layers):

            layer_modules = []

            # Calcuate input and output size of layer.
            layer_input_size = self.input_size if i == 0 else self.hidden_size
            layer_output_size = (
                self.output_size if i == self.num_layers - 1 else self.hidden_size
            )

            # Determine init function for layer.
            layer_init = self.init_base if i < self.num_layers - 1 else self.init_final

            # Initialize layer.
            layer_modules.append(
                layer_init(nn.Linear(layer_input_size, layer_output_size))
            )

            # Activation function.
            if i != self.num_layers - 1:
                layer_modules.append(nn.Tanh())

            # Combine linear transformation and activation into a single layer and add
            # to list of layers. Note that each element of `layers` is a list of layers,
            # initialized with only a single element. This is because each element of
            # `layers` represents all copies of a layer used by different subsets of all
            # tasks, but initially all layers are shared by all tasks, so we only have
            # one copy per region.
            region = nn.ModuleList([nn.Sequential(*layer_modules)])
            region_list.append(region)

        self.regions = nn.ModuleList(region_list)
        self.num_regions = len(self.regions)

        # Initialize splitting maps. These are the variables that define the parameter
        # sharing structure between tasks.
        self.maps = [SplittingMap(self.num_tasks) for _ in range(self.num_layers)]

        # Find size of biggest region. We use this to initialize tensors that hold
        # gradients with respect to specific regions.
        region_sizes = [
            sum([param.nelement() for param in self.regions[i][0].parameters()])
            for i in range(self.num_regions)
        ]
        self.max_region_size = max(region_sizes)

    def forward(self, inputs: torch.Tensor, task_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass definition for SplittingMLPNetwork. For each layer of the network,
        we aggregate the inputs by their assigned copy of each region, and pass the
        inputs through the corresponding copy.

        Implementation note: As I see it, there are two ways that we can reasonably
        implement this function. The first is, at each region, sorting the inputs by
        copy index, batching the inputs by copy, performing one forward pass per copy,
        and restoring the original order of the inputs. The second is by, just once at
        the beginning of the forward pass, sorting the inputs by task, then performing
        one forward pass per task at each region, then restoring the original order of
        the inputs just once at the end of the forward pass. The trade off between them
        is that the first involves sorting/unsorting at each region but only uses one
        forward pass per copy per region, while the second only sorts/unsorts once but
        performs one forward pass per task per region. We have used the first method
        here, in the name of minimizing the number of forward passes (which are likely
        going to add more computational burden than sorting), but I could see an
        argument for the second implementation.

        Arguments
        ---------
        inputs : torch.Tensor
            Input to splitting MLP network.
        task_indices : torch.Tensor
            Task index for each input in `inputs` as a integer.

        Returns
        -------
        outputs : torch.Tensor
            Output of splitting MLP network when given `inputs` as input.
        """

        assert len(inputs) == len(task_indices)

        # Pass through each splitting layer.
        x = inputs
        for layer in range(self.num_layers):

            # Sort the inputs by copy index and construct reverse permutation to later
            # restore the original order.
            copy_indices = self.maps[layer].module[task_indices]
            sorted_copy_indices, copy_permutation = torch.sort(copy_indices)
            _, copy_reverse = torch.sort(copy_permutation)
            x = x[copy_permutation]

            # Pass through each copy of the region and stack outputs.
            copy_outputs = []
            for copy in range(self.maps[layer].num_copies):
                batch_indices = (sorted_copy_indices == copy).nonzero().squeeze(-1)
                batch = x[batch_indices]
                copy_outputs.append(self.regions[layer][copy](batch))
            x = torch.cat(copy_outputs)

            # Restore original order of inputs.
            x = x[copy_reverse]

        return x

    def check_for_split(self, task_losses: torch.Tensor) -> None:
        """
        Determine whether any splits should occur based on the task-specific losses from
        the current batch. To do this, for each region we update our statistics that
        estimate the pairwise differences of task gradients, and we check whether there
        is a statistically significant difference in the gradient distribution between
        two tasks. If so, we perform a split.
        """

        task_grads = self.get_task_grads(task_losses)

    def get_task_grads(self, task_losses: torch.Tensor) -> torch.Tensor:
        """
        Compute the task-specific gradients for each task at each region.

        Arguments
        ---------
        task_losses : torch.Tensor
            A tensor of shape `(self.num_tasks,)` holding the task-specific losses for a
            batch.

        Returns
        -------
        task_grads : torch.Tensor
            A tensor of size `(self.num_tasks, self.num_regions, self.max_region_size)`.
            `task_grads[i, j]` that holds the gradient of task loss `i` with respect to
            region `j` padded with zeros to fit the size of the tensor.
        """

        task_grads = torch.zeros(
            (self.num_tasks, self.num_regions, self.max_region_size)
        )
        for task in range(self.num_tasks):

            self.zero_grad()
            task_losses[task].backward(retain_graph=True)

            for region in range(self.num_regions):
                param_grad_list = []
                copy = int(self.maps[region].module[task])
                for param in self.regions[region][copy].parameters():
                    param_grad_list.append(param.grad.view(-1))
                region_grad = torch.cat(param_grad_list)
                task_grads[task, region, : len(region_grad)] = region_grad

        return task_grads

    def split(
        self, region: int, copy: int, group_1: List[int], group_2: List[int]
    ) -> None:
        """
        Split the `copy`-th copy of region `region`. The tasks with indices in `group_1`
        will remain tied to copy `copy`, while the tasks with indices in `group_2` will
        be assigned to the new copy. It is required that the combined task indices of
        `group_1` and `group_2` make up all tasks assigned to copy `copy` at region
        `region`.
        """

        # Split the map that describes the splitting structure, so that tasks with
        # indices in `group_2` are assigned to the new copy.
        self.maps[region].split(copy, group_1, group_2)

        # Create a new module and add to parameters.
        new_copy = deepcopy(self.regions[region][copy])
        self.regions[region].append(new_copy)


class SplittingMap:
    """
    Data structure used to encode the splitting structure of a single region of a
    splitting network.
    """

    def __init__(self, num_tasks: int) -> None:
        """ Init function for Splitting Variable. """

        self.num_tasks = num_tasks
        self.num_copies = 1
        self.module = torch.zeros(self.num_tasks)

    def split(self, copy: int, group_1: List[int], group_2: List[int]) -> None:
        """
        Split a module with index `copy` into two modules, one corresponding to tasks
        with inidices in `group_1` and the other to `group_2`. Note that to call this
        function, it must be that the combined indices of `group_1` and `group_2` form
        the set of indices i with self.module_map[i] = copy.
        """

        # Check that precondition is satisfied.
        assert set(group_1 + group_2) == set(
            [i for i in range(self.num_tasks) if self.module[i] == copy]
        )

        self.num_copies += 1
        for task_index in group_2:
            self.module[task_index] = self.num_copies - 1
