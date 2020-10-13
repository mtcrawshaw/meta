"""
Definition of SplittingMLPNetwork, a multi-layer perceptron splitting network.
"""

import math
from itertools import product
from copy import deepcopy
from typing import Callable, List, Tuple

import numpy as np
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.nn.functional as F

from meta.utils.estimate import RunningStats


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
        split_alpha: float = 0.05,
        split_step_threshold: int = 30,
        ema_alpha: float = 0.999,
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
        self.split_alpha = split_alpha
        self.split_step_threshold = split_step_threshold
        self.ema_alpha = ema_alpha

        # Set device.
        self.device = device if device is not None else torch.device("cpu")

        # Generate network layers.
        self.initialize_network()

        # Initialize running estimates of gradient statistics.
        self.grad_diff_stats = RunningStats(
            shape=(self.num_tasks, self.num_tasks, self.num_regions),
            ema_alpha=self.ema_alpha,
        )
        self.grad_stats = RunningStats(
            shape=(self.num_tasks, self.total_region_size),
            compute_stdev=True,
            condense_dims=(1,),
            ema_alpha=self.ema_alpha,
        )
        self.num_steps = 0

        # Compute critical value of z-statistic based on given value of `split_alpha`.
        self.critical_z = norm.ppf(1 - self.split_alpha)

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

        # Store size of each region. We use this info to initialize tensors that hold
        # gradients with respect to specific regions.
        self.region_sizes = torch.Tensor(
            [
                sum([param.nelement() for param in self.regions[i][0].parameters()])
                for i in range(self.num_regions)
            ]
        )
        self.region_sizes = self.region_sizes.to(dtype=torch.long)
        self.max_region_size = int(max(self.region_sizes))
        self.total_region_size = int(sum(self.region_sizes))

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
        estimate the pairwise differences of task gradients, and we compute a z-score
        over these differences that determine whether there is a statistically
        significant difference in the gradient distribution between two tasks at a given
        region. If so, we perform a split of those tasks at the given region.
        """

        self.num_steps += 1

        # Compute task-specific gradients.
        task_grads = self.get_task_grads(task_losses)

        # Update running estimates of gradient statistics.
        self.update_grad_stats(task_grads)

        # Compute test statistics regarding difference of task gradient distributions.
        if self.num_steps >= self.split_step_threshold:
            z = self.get_split_statistics()

            # Check if `z` is large enough to warrant any splits.
            split_coords = (z > self.critical_z).nonzero()

            # Perform any necessary splits. Notice that we only do this for if `task1,
            # task` current share the same copy of `region`, and if `task1 < task2`,
            # since we don't want to split for both coords (task1, task2, region) and
            # (task2, task1, region).
            for task1, task2, region in split_coords:
                if self.maps[region].module[task1] != self.maps[region].module[task2]:
                    continue
                if task1 >= task2:
                    continue
                copy = self.maps[region].module[task1]

                # Partition tasks into groups by distance to task1 and task2. Notice
                # that we have to filter the groups of tasks so that we are only
                # including tasks that currently share the same copy of `region` with
                # `task1` and `task2`.
                group1 = (z[task1, :, region] < z[task2, :, region]).nonzero()
                group1 = group1.squeeze(-1).tolist()
                group1 = [
                    task for task in group1 if self.maps[region].module[task] == copy
                ]
                group2 = [
                    task
                    for task in range(self.num_tasks)
                    if task not in group1 and self.maps[region].module[task] == copy
                ]

                # Execute split.
                self.split(region, copy, group1, group2)

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

    def update_grad_stats(self, task_grads: torch.Tensor) -> None:
        """
        Update our running estimates of gradient statistics. We keep running estimates
        of the mean of the squared difference between task gradients at each region, and
        an estimate of the standard deviation of the gradient of each weight.
        """

        # Compute pairwise differences between task-specific gradients.
        task_grad_diffs = self.get_task_grad_diffs(task_grads)

        # Update our estimates of the mean pairwise distance between tasks and the
        # standard deviation of the gradient of each individual weight. Since
        # `task_grads` is a single tensor padded with zeros, we extract the non-pad
        # values before updating `self.grad_stats`. The non-pad values are extracted
        # into a tensor of shape `(self.num_tasks, self.total_region_size)`.
        flattened_grad = torch.cat(
            [task_grads[:, r, : self.region_sizes[r]] for r in range(self.num_regions)],
            dim=1,
        )
        self.grad_diff_stats.update(task_grad_diffs)
        self.grad_stats.update(flattened_grad)

    def get_task_grad_diffs(self, task_grads: torch.Tensor) -> None:
        """
        Compute the squared pairwise differences between task-specific gradients.

        Arguments
        ---------
        task_grads : torch.Tensor
            A tensor of size `(self.num_tasks, self.num_regions, self.max_region_size)`.
            `task_grads[i, j]` that holds the gradient of task loss `i` with respect to
            region `j` padded with zeros to fit the size of the tensor.

        Returns
        -------
        task_grad_diffs : torch.Tensor
            A tensor of size `(self.num_tasks, self.num_tasks, self.num_regions)`.
            `task_grad_diffs[i, j, k]` holds the squared norm of the difference between
            the task-specific gradients for tasks `i` and `j` at region `k`.
        """

        task_grad_diffs = torch.zeros(self.num_tasks, self.num_tasks, self.num_regions)

        # Compute pairwise difference for gradients at each region.
        for region in range(self.num_regions):

            # Use F.pdist to get pairwise distance of grads at each region.
            region_grads = task_grads[:, region, :]
            region_grad_diffs = torch.pow(F.pdist(region_grads), 2)

            # Reshape pairwise distances. What we get back will fill the flattened upper
            # triangle of `task_grad_diffs[:, :, region]`, which has shape
            # `(self.num_tasks * (self.num_tasks - 1) / 2)` so we do some weirdness to
            # reshape it quickly. We fill the upper triangle of `task_grad_diffs`, then
            # we add the tensor to its transpose to fill the entire thing.
            pos = 0
            for i, n in enumerate(reversed(range(self.num_tasks))):
                task_grad_diffs[i, i + 1 :, region] = region_grad_diffs[pos : pos + n]
                pos += n

        # Convert upper triangular tensor to symmetric tensor.
        task_grad_diffs += torch.transpose(task_grad_diffs, 0, 1)

        return task_grad_diffs

    def get_split_statistics(self) -> torch.Tensor:
        """
        Compute the z-statistic for each pair of tasks at each region. Intuitively, the
        magnitude of this value represents the difference in the distributions of task
        gradients between each pair of tasks at each region. If a z-score is large
        enough, then we will perform a split for the corresponding tasks/region.

        Returns
        -------
        z : torch.Tensor
            Tensor of size (self.num_tasks, self.num_tasks, self.num_regions), where
            `z[i, j, k]` holds the z-score of the mean difference in task gradients of
            tasks `i, j` at region `k`.
        """

        est_grad_var = torch.mean(self.grad_stats.var)
        mu = 2 * self.region_sizes * est_grad_var
        mu = mu.expand(self.num_tasks, self.num_tasks, -1)
        float_region_sizes = self.region_sizes.to(dtype=torch.float32)
        sigma = 2 * torch.sqrt(2 * float_region_sizes) * est_grad_var
        sigma = sigma.expand(self.num_tasks, self.num_tasks, -1)
        z = (self.grad_diff_stats.mean - mu) / (
            sigma / math.sqrt(self.grad_stats.sample_size)
        )

        return z

    def split(
        self, region: int, copy: int, group1: List[int], group2: List[int]
    ) -> None:
        """
        Split the `copy`-th copy of region `region`. The tasks with indices in `group1`
        will remain tied to copy `copy`, while the tasks with indices in `group2` will
        be assigned to the new copy. It is required that the combined task indices of
        `group1` and `group2` make up all tasks assigned to copy `copy` at region
        `region`.
        """

        # Split the map that describes the splitting structure, so that tasks with
        # indices in `group2` are assigned to the new copy.
        self.maps[region].split(copy, group1, group2)

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
        self.module = torch.zeros(self.num_tasks, dtype=torch.long)

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
