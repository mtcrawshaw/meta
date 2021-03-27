"""
Definition of GradMonitor, which is used to analyze the relationship between gradients
of task-specific losses.
"""

from itertools import product
import csv

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from meta.networks.mlp import MLPNetwork
from meta.utils.estimate import RunningStats


class GradMonitor:
    """
    Tool to analyze the relationship between gradients of task-specific losses.
    Currently we are only able to analyze gradients for an MLPNetwork model.
    """

    def __init__(
        self,
        network: MLPNetwork,
        num_tasks: int,
        metric: str = "sqeuclidean",
        ema_alpha: float = 0.999,
        cap_sample_size: bool = True,
        device: torch.device = None,
    ) -> None:
        """ Init function for GradMonitor. """

        # Set state.
        self.network = network
        self.num_tasks = num_tasks
        self.metric = metric
        self.ema_alpha = ema_alpha
        self.cap_sample_size = cap_sample_size
        self.num_layers = self.network.num_layers

        # Check for valid metric type.
        assert self.metric in ["sqeuclidean", "cosine"]

        # Set device.
        self.device = device if device is not None else torch.device("cpu")

        # Compute max layer size.
        self.layer_sizes = [
            sum([param.nelement() for param in self.network.layers[i].parameters()])
            for i in range(self.num_layers)
        ]
        self.max_layer_size = int(max(self.layer_sizes))

        # Initialize running statistics for task gradient differences.
        self.grad_stats = RunningStats(
            shape=(self.num_tasks, self.num_tasks, self.num_layers),
            cap_sample_size=self.cap_sample_size,
            ema_alpha=self.ema_alpha,
            device=self.device,
        )

        # Initialize history of gradient statistics.
        self.grad_mean_history = []

    def update_grad_stats(
        self, task_losses: torch.Tensor = None, task_grads: torch.Tensor = None
    ) -> None:
        """
        Update statistics over pairs of gradients of task losses at each layer. One of
        `task_losses` and `task_grads` should not be none. If `task_grads` is not None,
        then `task_losses` is ignored.
        """

        # Get task-specific gradients if they aren't given.
        if task_grads is None:
            assert task_losses is not None
            task_grads = self.get_task_grads(task_losses)

        # Compute distances between pairs of gradients.
        task_grad_diffs = self.get_task_grad_diffs(task_grads)

        # Get indices of tasks with non-zero gradients. A task will have zero gradients
        # when the current batch doesn't contain any data from that task, and in that
        # case we do not want to update the gradient stats for this task.
        task_flags = (task_grads.view(self.num_tasks, -1) != 0.0).any(dim=1)
        task_pair_flags = task_flags.unsqueeze(0) * task_flags.unsqueeze(1)
        task_pair_flags = task_pair_flags.unsqueeze(-1)
        task_pair_flags = task_pair_flags.expand(-1, -1, self.num_layers)

        # Update running stats with new task_grad_distances and add to history.
        self.grad_stats.update(task_grad_diffs, task_pair_flags)
        self.grad_mean_history.append(self.grad_stats.mean.numpy())

    def get_task_grads(self, task_losses: torch.Tensor) -> torch.Tensor:
        """ Compute and return task-specific gradients from losses. """

        # Compute task-specific gradients at each network layer.
        task_grads = torch.zeros(
            (self.num_tasks, self.num_layers, self.max_layer_size), device=self.device,
        )
        for task in range(self.num_tasks):

            self.network.zero_grad()
            task_losses[task].backward(retain_graph=True)

            for layer in range(self.num_layers):
                param_grad_list = []
                for param in self.network.layers[layer].parameters():
                    param_grad_list.append(param.grad.view(-1))
                layer_grad = torch.cat(param_grad_list)
                task_grads[task, layer, : len(layer_grad)] = layer_grad

        return task_grads

    def get_task_grad_diffs(self, task_grads: torch.Tensor) -> torch.Tensor:
        """ Compute and return pairwise differences between task-specific gradients. """

        task_grad_diffs = torch.zeros(
            (self.num_tasks, self.num_tasks, self.num_layers),
        )

        # Compute pairwise distances between gradients.
        for layer in range(self.num_layers):
            layer_grads = task_grads[:, layer, : self.layer_sizes[layer]]
            layer_grad_diffs = torch.pow(F.pdist(layer_grads), 2)

            # Unflatten return value of `F.pdist` into the shape we need.
            pos = 0
            for i, n in enumerate(reversed(range(self.num_tasks))):
                task_grad_diffs[i, i + 1 :, layer] = layer_grad_diffs[pos : pos + n]
                pos += n

        # Convert upper triangular tensor to symmetric tensor.
        task_grad_diffs += torch.transpose(task_grad_diffs, 0, 1)

        # Convert Euclidean distance to cosine distance, if necessary.
        if self.metric == "cosine":

            # Compute sum/product of norms of gradients for each pair (task1/task2, layer).
            pair_norm_sums = []
            pair_norm_products = []
            for layer in range(self.num_layers):
                layer_norms = torch.sum(task_grads[:, layer] ** 2, axis=-1)
                pair_layer_norms = torch.cartesian_prod(layer_norms, layer_norms)
                pair_layer_norm_sum = torch.sum(pair_layer_norms, axis=1)
                pair_layer_norm_sum = pair_layer_norm_sum.reshape(
                    self.num_tasks, self.num_tasks
                )
                pair_layer_norm_product = torch.prod(pair_layer_norms, axis=1)
                pair_layer_norm_product = pair_layer_norm_product.reshape(
                    self.num_tasks, self.num_tasks
                )
                pair_norm_sums.append(pair_layer_norm_sum)
                pair_norm_products.append(pair_layer_norm_product)
            pair_norm_sums = torch.stack(pair_norm_sums, axis=-1)
            pair_norm_products = torch.stack(pair_norm_products, axis=-1)

            # Subtract sum of grad norms from norm of grad difference, divide by sqrt of
            # product of grad norms, then divide by -2 to yield cosine distance.
            task_grad_diffs -= pair_norm_sums
            task_grad_diffs /= torch.sqrt(pair_norm_products)
            task_grad_diffs /= -2.0

            # Get rid of any infs or nans that may have appeared from zero division.
            bad_idxs = torch.isnan(task_grad_diffs) + torch.isinf(task_grad_diffs)
            task_grad_diffs[bad_idxs] = 0.0

        elif self.metric != "sqeuclidean":
            raise NotImplementedError

        return task_grad_diffs

    def plot_stats(self, plot_path: str) -> None:
        """ Plot the history gradient statistics throughout training. """

        # Convert stats histories into numpy arrays.
        num_steps = len(self.grad_mean_history)
        mean_history = np.array(self.grad_mean_history)

        # Format paths of output plots.
        dot = plot_path.rfind(".")
        path_template = plot_path[:dot] + "_%s" + plot_path[dot:]

        # Create a separate plot for each layer.
        for layer in range(self.num_layers):

            # Create plot for given layer.
            fig, ax = plt.subplots()
            for task1 in range(self.num_tasks - 1):
                for task2 in range(task1, self.num_tasks):
                    xs = np.arange(num_steps)
                    ys = mean_history[:, task1, task2, layer]
                    ax.plot(xs, ys)

            plt.savefig(path_template % str(layer))

    def write_table(self, table_path: str) -> None:
        """ Write out a sorted table of final grad diff means as a CSV. """

        # Get final mean grad diff of each pair of tasks, summed across layers.
        num_pairs = int((self.num_tasks - 1) * self.num_tasks / 2)
        pair_grad_diffs = np.zeros(num_pairs)
        task_pairs = np.zeros((num_pairs, 2))
        pos = 0
        for task1 in range(self.num_tasks - 1):
            for task2 in range(task1 + 1, self.num_tasks):
                pair_grad_diffs[pos] = np.sum(self.grad_mean_history[-1][task1, task2])
                task_pairs[pos] = [task1, task2]
                pos += 1

        # Sort grad diffs of each task pair.
        sort_order = np.argsort(pair_grad_diffs)
        sort_order = np.flip(sort_order)
        sorted_pair_grad_diffs = pair_grad_diffs[sort_order]
        sorted_task_pairs = task_pairs[sort_order].astype(int)

        # Write out table.
        with open(table_path, "w") as table_file:
            csv_writer = csv.writer(table_file)
            for (task1, task2), grad_diff in zip(
                sorted_task_pairs, sorted_pair_grad_diffs
            ):
                csv_writer.writerow([task1, task2, grad_diff])
