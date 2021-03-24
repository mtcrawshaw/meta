"""
Definition of GradMonitor, which is used to analyze the relationship between gradients
of task-specific losses.
"""

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
        ema_alpha: float = 0.999,
        cap_sample_size: bool = True,
        device: torch.device = None,
    ) -> None:
        """ Init function for GradMonitor. """

        # Set state.
        self.network = network
        self.num_tasks = self.network.num_tasks
        self.num_layers = self.network.num_layers
        self.ema_alpha = ema_alpha
        self.cap_sample_size = cap_sample_size

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
        self.grad_history = []

    def update_grad_stats(self, task_losses: torch.Tensor) -> None:
        """ Update statistics over pairs of gradients of task losses at each layer. """

        # Get task-specific gradients.
        task_grads = self.get_task_grads(task_losses)

        # Compute distances between pairs of gradients.
        task_grad_distances = self.get_task_grad_distances(task_grads)

        # Update running stats with new task_grad_distances and add to history.
        self.grad_stats.update(task_grad_distances)
        self.grad_history.append(self.grad_stats.mean)

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
                for param in self.network[layer].parameters():
                    param_grad_list.append(param.grad.view(-1))
                layer_grad = torch.cat(param_grad_list)
                task_grads[task, layer, : len(layer_grad)] = layer_grad

        return task_grads

    def get_task_grad_distances(self, task_grads: torch.Tensor) -> torch.Tensor:
        """ Compute and return pairwise distances between task-specific gradients. """

        task_grad_distances = torch.zeros(
            (self.num_tasks, self.num_tasks, self.num_layers),
        )

        # Compute pairwise distances between gradients.
        for layer in self.num_layers:
            layer_grads = task_grads[:, layer, : self.layer_sizes[layer]]
            layer_grad_distances = torch.pow(F.pdist(layer_grads), 2)

            # Unflatten return value of `F.pdist` into the shape we need.
            pos = 0
            for i, n in enumerate(reversed(range(self.num_tasks))):
                task_grad_distances[i, i + 1 :, layer] = layer_grad_distances[
                    pos : pos + n
                ]
                pos += n

        # Convert upper triangular tensor to symmetric tensor.
        task_grad_distances += torch.transpose(task_grad_diffs, 0, 1)

        return task_grad_distances

    def plot_stats(self, plot_path: str) -> None:
        """ Plot the history gradient statistics throughout training. """

        raise NotImplementedError
