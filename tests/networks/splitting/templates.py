"""
Templates for tests in tests/networks/splitting/.
"""

from math import sqrt
from itertools import product
from typing import List, Dict, Any, Callable
from gym.spaces import Box

import numpy as np
import torch

from meta.networks.utils import init_base
from meta.networks.splitting import (
    BaseMultiTaskSplittingNetwork,
    MultiTaskSplittingNetworkV1,
    MultiTaskSplittingNetworkV2,
    MetaSplittingNetwork,
)
from meta.utils.estimate import alpha_to_threshold
from tests.helpers import DEFAULT_SETTINGS, get_obs_batch


TOL = 2e-3


def gradients_template(
    settings: Dict[str, Any], splits_args: List[Dict[str, Any]]
) -> None:
    """
    Template to test that `get_task_grads()` correctly computes task-specific gradients
    at each region of the network. For simplicity we compute the loss as half of the
    squared norm of the output, and we make the following assumptions: each layer has
    the same size, the activation function is Tanh for each layer, and the final layer
    has no activation.
    """

    # Set up case.
    dim = settings["obs_dim"] + settings["num_tasks"]
    observation_subspace = Box(low=-np.inf, high=np.inf, shape=(settings["obs_dim"],))
    observation_subspace.seed(DEFAULT_SETTINGS["seed"])
    hidden_size = dim

    # Construct network.
    network = BaseMultiTaskSplittingNetwork(
        input_size=dim,
        output_size=dim,
        num_tasks=settings["num_tasks"],
        num_layers=settings["num_layers"],
        hidden_size=hidden_size,
        device=settings["device"],
    )

    # Split the network according to `splits_args`.
    for split_args in splits_args:
        network.split(**split_args)

    # Re-initialize the new copies so different tasks will actually have different
    # corresponding functions.
    state_dict = network.state_dict()
    for region in range(network.num_regions):
        for copy in range(1, int(network.splitting_map.num_copies[region])):
            weight_name = "regions.%d.%d.0.weight" % (region, copy)
            bias_name = "regions.%d.%d.0.bias" % (region, copy)
            state_dict[weight_name] = torch.rand(state_dict[weight_name].shape)
            state_dict[bias_name] = torch.rand(state_dict[bias_name].shape)
    network.load_state_dict(state_dict)

    # Register forward hooks to get activations later from each copy of each region.
    activation = {}

    def get_activation(name):
        def hook(model, ins, outs):
            activation[name] = outs.detach()

        return hook

    for region in range(network.num_regions):
        for copy in range(int(network.splitting_map.num_copies[region])):
            name = "regions.%d.%d" % (region, copy)
            network.regions[region][copy].register_forward_hook(get_activation(name))

    # Construct batch of observations concatenated with one-hot task vectors.
    obs, task_indices = get_obs_batch(
        batch_size=settings["num_processes"],
        obs_space=observation_subspace,
        num_tasks=settings["num_tasks"],
    )

    # Get output of network and compute task gradients.
    output = network(obs, task_indices)
    task_losses = torch.zeros(settings["num_tasks"])
    for task in range(settings["num_tasks"]):
        for current_out, current_task in zip(output, task_indices):
            if current_task == task:
                task_losses[task] += 0.5 * torch.sum(current_out ** 2)

    task_grads = network.get_task_grads(task_losses)

    def get_task_activations(r, t, tasks):
        """ Helper function to get activations from specific regions. """

        c = network.splitting_map.copy[r, t]
        copy_indices = network.splitting_map.copy[r, tasks]
        sorted_copy_indices, copy_permutation = torch.sort(copy_indices)
        sorted_tasks = tasks[copy_permutation]
        batch_indices = (sorted_copy_indices == c).nonzero().squeeze(-1)
        task_batch_indices = sorted_tasks[batch_indices]
        current_task_indices = (task_batch_indices == t).nonzero().squeeze(-1)
        activations = activation["regions.%d.%d" % (r, c)][current_task_indices]

        return activations

    # Compute expected gradients.
    state_dict = network.state_dict()
    expected_task_grads = torch.zeros(
        (settings["num_tasks"], network.num_regions, network.max_region_size)
    )
    for task in range(settings["num_tasks"]):

        # Get output from current task.
        task_input_indices = (task_indices == task).nonzero().squeeze(-1)
        task_output = output[task_input_indices]

        # Clear local gradients.
        local_grad = {}

        for region in reversed(range(network.num_regions)):

            # Get copy index and layer input.
            copy = network.splitting_map.copy[region, task]
            if region > 0:
                layer_input = get_task_activations(region - 1, task, task_indices)
            else:
                layer_input = obs[task_input_indices]

            # Compute local gradient first.
            if region == network.num_regions - 1:
                local_grad[region] = -task_output
            else:
                layer_output = get_task_activations(region, task, task_indices)
                local_grad[region] = torch.zeros(len(layer_output), dim)
                next_copy = network.splitting_map.copy[region + 1, task]
                weights = state_dict["regions.%d.%d.0.weight" % (region + 1, next_copy)]
                for i in range(dim):
                    for j in range(dim):
                        local_grad[region][:, i] += (
                            local_grad[region + 1][:, j] * weights[j, i]
                        )
                local_grad[region] = local_grad[region] * (1 - layer_output ** 2)

            # Compute gradient from local gradients.
            grad = torch.zeros(dim, dim + 1)
            for i in range(dim):
                for j in range(dim):
                    grad[i, j] = torch.sum(
                        -local_grad[region][:, i] * layer_input[:, j]
                    )
                grad[i, dim] = torch.sum(-local_grad[region][:, i])

            # Rearrange weights and biases. Should be all weights, then all biases.
            weights = torch.reshape(grad[:, :-1], (-1,))
            biases = torch.reshape(grad[:, -1], (-1,))
            grad = torch.cat([weights, biases])
            expected_task_grads[task, region, : len(grad)] = grad

    # Test gradients.
    assert torch.allclose(task_grads, expected_task_grads, atol=2e-5)


def backward_template(
    settings: Dict[str, Any], splits_args: List[Dict[str, Any]]
) -> None:
    """
    Template to test that the backward() function correctly computes gradients. We don't
    actually compare the gradients against baseline values, instead we just check that
    the gradient of the loss for task i is non-zero for all copies that i is assigned
    to, and zero for all copies i isn't assigned to, for each i. To keep things simple,
    we define each task loss as the squared norm of the output for inputs from the given
    task.
    """

    # Set up case.
    dim = settings["obs_dim"] + settings["num_tasks"]
    observation_subspace = Box(low=-np.inf, high=np.inf, shape=(settings["obs_dim"],))
    observation_subspace.seed(DEFAULT_SETTINGS["seed"])
    hidden_size = dim

    # Construct network.
    network = BaseMultiTaskSplittingNetwork(
        input_size=dim,
        output_size=dim,
        num_tasks=settings["num_tasks"],
        num_layers=settings["num_layers"],
        hidden_size=hidden_size,
        device=settings["device"],
    )

    # Split the network according to `splits_args`.
    for split_args in splits_args:
        network.split(**split_args)

    # Re-initialize the new copies so different tasks will actually have different
    # corresponding functions.
    state_dict = network.state_dict()
    for region in range(network.num_regions):
        for copy in range(1, int(network.splitting_map.num_copies[region])):
            weight_name = "regions.%d.%d.0.weight" % (region, copy)
            bias_name = "regions.%d.%d.0.bias" % (region, copy)
            state_dict[weight_name] = torch.rand(state_dict[weight_name].shape)
            state_dict[bias_name] = torch.rand(state_dict[bias_name].shape)
    network.load_state_dict(state_dict)

    # Construct batch of observations concatenated with one-hot task vectors.
    obs, task_indices = get_obs_batch(
        batch_size=settings["num_processes"],
        obs_space=observation_subspace,
        num_tasks=settings["num_tasks"],
    )

    # Get output of network and compute task losses.
    output = network(obs, task_indices)
    task_losses = {i: None for i in range(settings["num_tasks"])}
    for task in range(settings["num_tasks"]):
        for current_out, current_task in zip(output, task_indices):
            if current_task == task:
                if task_losses[task] is not None:
                    task_losses[task] += torch.sum(current_out ** 2)
                else:
                    task_losses[task] = torch.sum(current_out ** 2)

    # Test gradients.
    for task in range(settings["num_tasks"]):
        network.zero_grad()
        if task_losses[task] is None:
            continue

        task_losses[task].backward(retain_graph=True)
        for region in range(len(network.regions)):
            for copy in range(int(network.splitting_map.num_copies[region])):
                for param in network.regions[region][copy].parameters():
                    zero = torch.zeros(param.grad.shape)
                    if network.splitting_map.copy[region, task] == copy:
                        assert not torch.allclose(param.grad, zero)
                    else:
                        assert torch.allclose(param.grad, zero)


def grad_diffs_template(settings: Dict[str, Any], grad_type: str) -> None:
    """
    Test that `get_task_grad_diffs()` correctly computes the pairwise difference between
    task-specific gradients at each region.
    """

    # Set up case.
    dim = settings["obs_dim"] + settings["num_tasks"]
    observation_subspace = Box(low=-np.inf, high=np.inf, shape=(settings["obs_dim"],))
    observation_subspace.seed(DEFAULT_SETTINGS["seed"])
    hidden_size = dim

    # Construct network.
    network = BaseMultiTaskSplittingNetwork(
        input_size=dim,
        output_size=dim,
        num_tasks=settings["num_tasks"],
        num_layers=settings["num_layers"],
        hidden_size=hidden_size,
        metric=settings["metric"],
        device=settings["device"],
    )

    # Construct dummy task gradients.
    if grad_type == "zero":
        task_grads = torch.zeros(
            network.num_tasks, network.num_regions, network.max_region_size
        )
    elif grad_type == "rand_identical":
        task_grads = torch.rand(1, network.num_regions, network.max_region_size)
        task_grads = task_grads.expand(network.num_tasks, -1, -1)
        pass
    elif grad_type == "rand":
        pass
        task_grads = torch.rand(
            network.num_tasks, network.num_regions, network.max_region_size
        )
    else:
        raise NotImplementedError

    # Compute pairwise differences of task gradients.
    task_grad_diffs = network.get_task_grad_diffs(task_grads)

    # Check computed differences.
    for task1, task2 in product(range(network.num_tasks), range(network.num_tasks)):
        for region in range(network.num_regions):
            actual = task_grad_diffs[task1, task2, region]
            grad1 = task_grads[task1, region]
            grad2 = task_grads[task2, region]

            # Compute expected difference.
            if settings["metric"] == "sqeuclidean":
                expected = torch.sum(torch.pow(grad1 - grad2, 2))
            elif settings["metric"] == "cosine":
                grad1 /= sqrt(torch.sum(grad1 ** 2))
                grad2 /= sqrt(torch.sum(grad2 ** 2))
                bad1 = torch.isinf(grad1) + torch.isnan(grad1)
                bad2 = torch.isinf(grad2) + torch.isnan(grad2)
                bad = (bad1 + bad2).any()
                if bad:
                    expected = torch.Tensor([0.0])
                else:
                    expected = torch.sum(grad1 * grad2)
            else:
                raise NotImplementedError

            assert torch.allclose(actual, expected)


def grad_stats_template(
    settings: Dict[str, Any],
    task_grads: torch.Tensor,
    splits_args: List[Dict[str, Any]],
) -> None:
    """
    Test that `update_grad_stats()` correctly computes pairwise differences of gradients
    over multiple steps.

    Arguments
    ---------
    settings : Dict[str, Any]
        Dictionary holding misc settings for how to run trial.
    task_grads : torch.Tensor
        Tensor of size `(total_steps, network.num_tasks, network.num_regions,
        network.max_region_size)` which holds the task gradients for multiple steps that
        we will compute statistics over.
    splits_args : List[Dict[str, Any]]
        List of splits to execute on network.
    """

    dim = settings["obs_dim"] + settings["num_tasks"]

    # Construct network.
    network = BaseMultiTaskSplittingNetwork(
        input_size=dim,
        output_size=dim,
        num_tasks=settings["num_tasks"],
        num_layers=settings["num_layers"],
        hidden_size=settings["hidden_size"],
        metric=settings["metric"],
        ema_alpha=settings["ema_alpha"],
        device=settings["device"],
    )
    ema_threshold = alpha_to_threshold(settings["ema_alpha"])

    # Split the network according to `splits_args`.
    for split_args in splits_args:
        network.split(**split_args)

    # Check that the region sizes are what we think they are.
    expected_region_sizes = torch.zeros(settings["num_layers"], dtype=torch.long)
    expected_region_sizes[1:-1] = settings["hidden_size"] ** 2 + settings["hidden_size"]
    expected_region_sizes[0] = settings["hidden_size"] * (dim + 1)
    expected_region_sizes[-1] = dim * (settings["hidden_size"] + 1)
    assert torch.all(expected_region_sizes == network.region_sizes)
    region_sizes = expected_region_sizes.tolist()

    # Update the networks's gradient statistics with our constructed task gradients,
    # compute the gradient statistics at each step along the way, and compare the
    # computed statistics against the expected values.
    task_flags = torch.zeros(settings["num_steps"], network.num_tasks)
    task_pair_flags = torch.zeros(
        settings["num_steps"], network.num_tasks, network.num_tasks
    )
    for step in range(settings["num_steps"]):
        network.num_steps += 1
        network.update_grad_stats(task_grads[step])
        assert network.grad_diff_stats.shape == (
            network.num_tasks,
            network.num_tasks,
            network.num_regions,
        )

        # Set task flags, i.e. indicators for whether or not each task is included in
        # the current batch, and compute sample sizes for each task and task pair.
        task_flags[step] = torch.any(
            task_grads[step].view(network.num_tasks, -1) != 0, dim=1
        )
        task_flags[step] = task_flags[step] * 1
        task_pair_flags[step] = task_flags[step].unsqueeze(0) * task_flags[
            step
        ].unsqueeze(1)
        sample_sizes = torch.sum(task_flags[: step + 1], dim=0)
        pair_sample_sizes = torch.sum(task_pair_flags[: step + 1], dim=0)

        def grad_diff(grad1: torch.Tensor, grad2: torch.Tensor) -> float:
            """ Compute diff between two gradients based on `metric`. """

            if settings["metric"] == "sqeuclidean":
                diff = torch.sum((grad1 - grad2) ** 2)
            elif settings["metric"] == "cosine":
                grad1 /= sqrt(torch.sum(grad1 ** 2))
                grad2 /= sqrt(torch.sum(grad2 ** 2))
                diff = torch.sum(grad1 * grad2)
            else:
                raise NotImplementedError

            return diff

        # Compare networks's gradients stats to the expected value for each `(task1,
        # task2, region)`.
        for task1, task2, region in product(
            range(network.num_tasks),
            range(network.num_tasks),
            range(network.num_regions),
        ):
            region_size = int(network.region_sizes[region])

            # Computed the expected value of the mean of gradient differences between
            # `task1, task2` at region `region`.
            steps = task_pair_flags[:, task1, task2].bool()
            if not torch.any(steps):
                continue
            task1_grads = task_grads[steps, task1, region, :region_size]
            task2_grads = task_grads[steps, task2, region, :region_size]
            num_steps = len(steps.nonzero())
            if pair_sample_sizes[task1, task2] <= ema_threshold:
                diffs = torch.Tensor(
                    [
                        grad_diff(task1_grads[i], task2_grads[i])
                        for i in range(num_steps)
                    ]
                )
                exp_mean = torch.mean(diffs)
            else:
                initial_task1_grads = task1_grads[:ema_threshold]
                initial_task2_grads = task2_grads[:ema_threshold]
                diffs = torch.Tensor(
                    [
                        grad_diff(initial_task1_grads[i], initial_task2_grads[i])
                        for i in range(ema_threshold)
                    ]
                )
                exp_mean = torch.mean(diffs)
                for i in range(ema_threshold, int(pair_sample_sizes[task1, task2])):
                    task1_grad = task1_grads[i]
                    task2_grad = task2_grads[i]
                    diff = grad_diff(task1_grad, task2_grad)
                    exp_mean = exp_mean * settings["ema_alpha"] + diff * (
                        1.0 - settings["ema_alpha"]
                    )

            # Compare expected mean to network's mean.
            actual_mean = network.grad_diff_stats.mean[task1, task2, region]
            diff = abs(actual_mean - exp_mean)
            assert abs(actual_mean - exp_mean) < TOL


def split_stats_template(
    settings: Dict[str, Any],
    task_grads: torch.Tensor,
    splits_args: List[Dict[str, Any]],
) -> None:
    """
    Test that `get_split_statistics()` correctly computes the z-score over the pairwise
    differences in task gradients, assuming that none of the task gradients are zero
    across an entire task.

    Arguments
    ---------
    settings : Dict[str, Any]
        Dictionary holding misc settings for how to run trial.
    task_grads : torch.Tensor
        Tensor of size `(total_steps, network.num_tasks, network.num_regions,
        network.max_region_size)` which holds the task gradients for multiple steps that
        we will compute statistics over.
    splits_args : List[Dict[str, Any]]
        List of splits to execute on network.
    """

    dim = settings["obs_dim"] + settings["num_tasks"]

    # Construct network.
    network = MultiTaskSplittingNetworkV1(
        input_size=dim,
        output_size=dim,
        num_tasks=settings["num_tasks"],
        num_layers=settings["num_layers"],
        hidden_size=settings["hidden_size"],
        grad_var=settings["grad_var"],
        cap_sample_size=settings["cap_sample_size"],
        ema_alpha=settings["ema_alpha"],
        device=settings["device"],
    )
    ema_threshold = alpha_to_threshold(settings["ema_alpha"])

    # Split the network according to `splits_args`.
    for split_args in splits_args:
        network.split(**split_args)

    # Check that the region sizes are what we think they are.
    expected_region_sizes = torch.zeros(settings["num_layers"], dtype=torch.long)
    expected_region_sizes[1:-1] = settings["hidden_size"] ** 2 + settings["hidden_size"]
    expected_region_sizes[0] = settings["hidden_size"] * (dim + 1)
    expected_region_sizes[-1] = dim * (settings["hidden_size"] + 1)
    assert torch.all(expected_region_sizes == network.region_sizes)
    region_sizes = expected_region_sizes.tolist()

    # Update the network's gradient statistics with our constructed task gradients,
    # compute the split statistics at each step along the way, and compare the computed
    # z-scores against the expected z-scores.
    task_flags = torch.zeros(len(task_grads), network.num_tasks)
    task_pair_flags = torch.zeros(len(task_grads), network.num_tasks, network.num_tasks)
    for step in range(len(task_grads)):
        network.num_steps += 1
        network.update_grad_stats(task_grads[step])
        z = network.get_split_statistics()
        assert z.shape == (network.num_tasks, network.num_tasks, network.num_regions)

        # Set task flags, i.e. indicators for whether or not each task is included in
        # each batch, and compute sample sizes for each task and task pair.
        task_flags[step] = torch.any(
            task_grads[step].view(network.num_tasks, -1) != 0, dim=1
        )
        task_flags[step] = task_flags[step] * 1
        task_pair_flags[step] = task_flags[step].unsqueeze(0) * task_flags[
            step
        ].unsqueeze(1)
        sample_sizes = torch.sum(task_flags[: step + 1], dim=0)
        pair_sample_sizes = torch.sum(task_pair_flags[: step + 1], dim=0)

        # Compute stdev over all gradient values up to `step`. We have to do this
        # differently based on whether or not we have hit the EMA threshold for each
        # task.
        task_vars = torch.zeros(network.num_tasks)
        for task in range(network.num_tasks):
            task_steps = task_flags[:, task].bool()
            if int(sample_sizes[task]) == 0:
                task_var = 0
            elif int(sample_sizes[task]) <= ema_threshold:
                task_grad = task_grads[task_steps, task : task + 1]
                flattened_grad = get_flattened_grads(
                    task_grad, 1, region_sizes, 0, int(sample_sizes[task]),
                )[0]
                task_var = torch.var(flattened_grad, unbiased=False)
            else:
                task_grad = task_grads[task_steps, task : task + 1]
                flattened_grad = get_flattened_grads(
                    task_grad, 1, region_sizes, 0, ema_threshold
                )[0]
                grad_mean = torch.mean(flattened_grad)
                grad_square_mean = torch.mean(flattened_grad ** 2)
                for i in range(ema_threshold, int(sample_sizes[task])):
                    flattened_grad = get_flattened_grads(
                        task_grad, 1, region_sizes, i, i + 1
                    )[0]
                    new_mean = torch.mean(flattened_grad)
                    new_square_mean = torch.mean(flattened_grad ** 2)
                    grad_mean = grad_mean * settings["ema_alpha"] + new_mean * (
                        1.0 - settings["ema_alpha"]
                    )
                    grad_square_mean = grad_square_mean * settings[
                        "ema_alpha"
                    ] + new_square_mean * (1.0 - settings["ema_alpha"])
                task_var = grad_square_mean - grad_mean ** 2

            task_vars[task] = task_var

        if settings["grad_var"] is None:
            grad_var = torch.sum(task_vars * sample_sizes) / torch.sum(sample_sizes)
        else:
            grad_var = settings["grad_var"]

        # Compare `z` to the expected value for each `(task1, task2, region)`.
        for task1, task2, region in product(
            range(network.num_tasks),
            range(network.num_tasks),
            range(network.num_regions),
        ):
            region_size = int(network.region_sizes[region])

            # Computed the expected value of the mean of gradient differences between
            # `task1, task2` at region `region`.
            steps = task_pair_flags[:, task1, task2].bool()
            if not torch.any(steps):
                continue
            task1_grads = task_grads[steps, task1, region, :]
            task2_grads = task_grads[steps, task2, region, :]
            if pair_sample_sizes[task1, task2] <= ema_threshold:
                diffs = torch.sum((task1_grads - task2_grads) ** 2, dim=1)
                exp_mean = torch.mean(diffs)
            else:
                initial_task1_grads = task1_grads[:ema_threshold]
                initial_task2_grads = task2_grads[:ema_threshold]
                diffs = torch.sum(
                    (initial_task1_grads - initial_task2_grads) ** 2, dim=1
                )
                exp_mean = torch.mean(diffs)
                for i in range(ema_threshold, int(pair_sample_sizes[task1, task2])):
                    task1_grad = task1_grads[i]
                    task2_grad = task2_grads[i]
                    diff = torch.sum((task1_grad - task2_grad) ** 2)
                    exp_mean = exp_mean * settings["ema_alpha"] + diff * (
                        1.0 - settings["ema_alpha"]
                    )

            # Compute the expected z-score.
            sample_size = int(pair_sample_sizes[task1, task2])
            if settings["cap_sample_size"]:
                sample_size = min(sample_size, ema_threshold)
            exp_mu = 2 * region_size * grad_var
            exp_sigma = 2 * sqrt(2 * region_size) * grad_var
            expected_z = sqrt(sample_size) * (exp_mean - exp_mu) / exp_sigma
            assert abs(z[task1, task2, region] - expected_z) < TOL


def split_v1_template(
    settings: Dict[str, Any],
    z: torch.Tensor,
    task_grads: torch.Tensor,
    splits_args: List[Dict[str, Any]],
) -> None:
    """
    Test whether splitting decisions are made correctly when given z-scores for the
    pairwise difference in gradient distributions at each region. `task_grads` is
    essentially a dummy input, the gradient values themselves don't matter. This is just
    used to update the network's gradient statistics in order to update
    `network.grad_diff_stats.num_steps`, so the only thing about `task_grads` that
    matters is whether or not the elements are non-zero.
    """

    # Instantiate network and perform splits.
    dim = settings["obs_dim"] + settings["num_tasks"]
    network = MultiTaskSplittingNetworkV1(
        input_size=dim,
        output_size=dim,
        num_tasks=settings["num_tasks"],
        num_layers=settings["num_layers"],
        hidden_size=dim,
        ema_alpha=settings["ema_alpha"],
    )
    for split_args in splits_args:
        network.split(**split_args)

    # Compute initial splitting map.
    splitting_map = [
        [list(range(network.num_tasks))] for _ in range(network.num_regions)
    ]
    for split_args in splits_args:
        region = split_args["region"]
        copy = split_args["copy"]
        group1 = split_args["group1"]
        group2 = split_args["group2"]
        assert set(splitting_map[region][copy]) == set(group1 + group2)
        splitting_map[region][copy] = list(group1)
        splitting_map[region].append(list(group))

    def get_copy(smap: List[List[List[int]]], task: int, region: int) -> int:
        """ Helper function to get copy index for a task/region pair. """
        copies = [i for i in range(len(smap[region])) if task in smap[region][i]]
        assert len(copies) == 1
        return copies[0]

    # Check split at each step. Note that we call update_grad_stats() with a dummy value
    # to make sure that `network.grad_diff_stats.num_steps` gets increased past
    # `network.split_step_threshold`, because otherwise the network would never split.
    for step in range(z.shape[0]):

        # Perform network split.
        network.num_steps += 1
        network.update_grad_stats(task_grads[step])
        should_split = z[step] > network.critical_z
        should_split *= (
            network.grad_diff_stats.num_steps >= network.split_step_threshold
        )
        network.perform_splits(should_split)

        # Compute expected splitting map.
        for task1 in range(network.num_tasks - 1):
            for task2 in range(task1 + 1, network.num_tasks):
                for region in range(network.num_regions):
                    critical = float(z[step, task1, task2, region]) > network.critical_z
                    sample = (
                        network.grad_diff_stats.num_steps[task1, task2, region]
                        >= network.split_step_threshold
                    )
                    task1_copy = get_copy(splitting_map, task1, region)
                    task2_copy = get_copy(splitting_map, task2, region)
                    shared = task1_copy == task2_copy

                    if critical and sample and shared:
                        copy = task1_copy
                        group1 = []
                        group2 = []
                        for task in splitting_map[region][copy]:
                            if task == task1:
                                group1.append(task)
                                continue
                            if task == task2:
                                group2.append(task)
                                continue

                            if (
                                network.grad_diff_stats.mean[task, task1, region]
                                < network.grad_diff_stats.mean[task, task2, region]
                            ):
                                group1.append(task)
                            else:
                                group2.append(task)

                        splitting_map[region][copy] = list(group1)
                        splitting_map[region].append(list(group2))

        # Compare network splitting map with expected splitting map.
        for task in range(network.num_tasks):
            for region in range(network.num_regions):
                actual_copy = network.splitting_map.copy[region, task]
                expected_copy = get_copy(splitting_map, task, region)
                assert actual_copy == expected_copy


def split_v2_template(settings: Dict[str, Any], task_grads: torch.Tensor) -> None:
    """
    Test whether splitting decisions are made correctly when given task gradients at
    each step.
    """

    # Instantiate network and perform splits.
    dim = settings["obs_dim"] + settings["num_tasks"]
    network = MultiTaskSplittingNetworkV2(
        input_size=dim,
        output_size=dim,
        num_tasks=settings["num_tasks"],
        num_layers=settings["num_layers"],
        hidden_size=dim,
        ema_alpha=settings["ema_alpha"],
        split_freq=settings["split_freq"],
        splits_per_step=settings["splits_per_step"],
    )

    # Compute initial splitting map.
    splitting_map = [
        [list(range(network.num_tasks))] for _ in range(network.num_regions)
    ]

    def get_copy(smap: List[List[List[int]]], task: int, region: int) -> int:
        """ Helper function to get copy index for a task/region pair. """
        copies = [i for i in range(len(smap[region])) if task in smap[region][i]]
        assert len(copies) == 1
        return copies[0]

    # Check split at each step.
    total_steps = task_grads.shape[0]
    for step in range(total_steps):

        split = True

        # Perform one training step and compute expected splits.
        network.num_steps += 1
        if network.get_sharing_score() <= network.sharing_threshold:
            split = False
        if (
            step + 1 < network.split_step_threshold
            or (step + 1) % network.split_freq != 0
        ):
            split = False
        network.update_grad_stats(task_grads[step])
        should_split = network.determine_splits()
        network.perform_splits(should_split)

        # Compute expected splitting map. If we've completed sufficiently many steps,
        # then we will split `network.splits_per_step` regions every
        # `network.split_freq` steps.
        if split:

            # Collect list of task gradient distances for shared regions.
            task_grad_dists = []
            for task1 in range(network.num_tasks - 1):
                for task2 in range(task1 + 1, network.num_tasks):
                    for region in range(network.num_regions):

                        # Check if region is shared and has valid sample size.
                        sample = network.grad_diff_stats.sample_size[
                            task1, task2, region
                        ]
                        copy1 = get_copy(splitting_map, task1, region)
                        copy2 = get_copy(splitting_map, task2, region)
                        if sample < network.split_step_threshold or copy1 != copy2:
                            continue

                        # If so, add its normalized task gradient distance to the list.
                        grad_dist = float(
                            network.grad_diff_stats.mean[task1, task2, region]
                        )
                        region_size = int(network.region_sizes[region])
                        task_grad_dists.append(grad_dist / region_size)

            # Get a threshold on the gradient distance to decide splitting.
            if len(task_grad_dists) >= network.splits_per_step:
                distance_threshold = sorted(task_grad_dists)[-network.splits_per_step]
            elif len(task_grad_dists) > 0:
                distance_threshold = sorted(task_grad_dists)[0]
            else:
                distance_threshold = None

            # Perform splits on any region with gradient distance larger than threshold.
            for task1 in range(network.num_tasks - 1):
                for task2 in range(task1 + 1, network.num_tasks):
                    for region in range(network.num_regions):

                        # Check if region is shared and has valid sample size.
                        sample = network.grad_diff_stats.sample_size[
                            task1, task2, region
                        ]
                        copy1 = get_copy(splitting_map, task1, region)
                        copy2 = get_copy(splitting_map, task2, region)
                        if sample < network.split_step_threshold or copy1 != copy2:
                            continue

                        # Check if region's normalized gradient distance warrants a split.
                        grad_dist = float(
                            network.grad_diff_stats.mean[task1, task2, region]
                        )
                        region_size = int(network.region_sizes[region])
                        normalized_grad_dist = grad_dist / region_size
                        if normalized_grad_dist < distance_threshold:
                            continue

                        # Perform split.
                        copy = copy1
                        group1 = []
                        group2 = []
                        for task in splitting_map[region][copy]:
                            if task == task1:
                                group1.append(task)
                                continue
                            if task == task2:
                                group2.append(task)
                                continue

                            if (
                                network.grad_diff_stats.mean[task, task1, region]
                                < network.grad_diff_stats.mean[task, task2, region]
                            ):
                                group1.append(task)
                            else:
                                group2.append(task)

                        splitting_map[region][copy] = list(group1)
                        splitting_map[region].append(list(group2))

        # Compare network splitting map with expected splitting map.
        for task in range(network.num_tasks):
            for region in range(network.num_regions):
                actual_copy = network.splitting_map.copy[region, task]
                expected_copy = get_copy(splitting_map, task, region)
                assert actual_copy == expected_copy


def score_template(
    settings: Dict[str, Any], splits_args: List[Dict[str, Any]], expected_score: float
) -> None:
    """
    Test that the sharing score is correctly computed.
    """

    # Instantiate network and perform splits.
    network = BaseMultiTaskSplittingNetwork(
        input_size=settings["input_size"],
        output_size=settings["output_size"],
        num_tasks=settings["num_tasks"],
        num_layers=settings["num_layers"],
        hidden_size=settings["hidden_size"],
        ema_alpha=settings["ema_alpha"],
    )
    for split_args in splits_args:
        network.split(**split_args)

    # Compare actual sharing score with expected sharing score.
    actual_score = network.get_sharing_score()
    assert actual_score == expected_score


def meta_forward_template(
    settings: Dict[str, Any],
    state_dict: Dict[str, torch.Tensor],
    splits_args: List[Dict[str, Any]],
    alpha: List[torch.Tensor],
    get_expected_output: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> None:
    """ Test MetaSplittingNetwork.forward() correct computes network output. """

    # Construct multi-task network.
    multitask_network = BaseMultiTaskSplittingNetwork(
        input_size=settings["input_size"],
        output_size=settings["output_size"],
        num_tasks=settings["num_tasks"],
        num_layers=settings["num_layers"],
        hidden_size=settings["hidden_size"],
        device=settings["device"],
    )

    # Split the network according to `splits_args`.
    for split_args in splits_args:
        multitask_network.split(**split_args)

    # Load state dict.
    multitask_network.load_state_dict(state_dict)

    # Construct MetaSplittingNetwork from BaseMultiTaskSplittingNetwork.
    meta_network = MetaSplittingNetwork(
        multitask_network,
        num_test_tasks=settings["num_tasks"],
        device=settings["device"],
    )

    # Set alpha weights of meta network.
    for layer in range(meta_network.num_layers):
        meta_network.alpha[layer].data = alpha[layer]

    # Construct batch of observations concatenated with one-hot task vectors.
    observation_subspace = Box(low=-np.inf, high=np.inf, shape=(settings["obs_dim"],))
    observation_subspace.seed(settings["seed"])
    obs, task_indices = get_obs_batch(
        batch_size=settings["num_processes"],
        obs_space=observation_subspace,
        num_tasks=settings["num_tasks"],
    )

    # Get and test output of network.
    output = meta_network(obs, task_indices)
    expected_output = get_expected_output(obs, task_indices)
    assert torch.allclose(output, expected_output)


def meta_backward_template(
    settings: Dict[str, Any],
    splits_args: List[Dict[str, Any]],
    alpha: List[torch.Tensor],
) -> None:
    """
    Template to test that the backward() function correctly computes gradients. We don't
    actually compare the gradients against baseline values, instead we just check that
    the gradients are non-zero for each of the alpha values and zero for the parameters
    in each region.
    """

    # Construct multi-task network.
    multitask_network = BaseMultiTaskSplittingNetwork(
        input_size=settings["input_size"],
        output_size=settings["output_size"],
        num_tasks=settings["num_tasks"],
        num_layers=settings["num_layers"],
        hidden_size=settings["hidden_size"],
        device=settings["device"],
    )

    # Split the network according to `splits_args`.
    for split_args in splits_args:
        multitask_network.split(**split_args)

    # Construct MetaSplittingNetwork from BaseMultiTaskSplittingNetwork.
    meta_network = MetaSplittingNetwork(
        multitask_network,
        num_test_tasks=settings["num_tasks"],
        device=settings["device"],
    )

    # Set alpha weights of meta network.
    for layer in range(meta_network.num_layers):
        meta_network.alpha[layer].data = alpha[layer]

    # Construct batch of observations concatenated with one-hot task vectors.
    observation_subspace = Box(low=-np.inf, high=np.inf, shape=(settings["obs_dim"],))
    observation_subspace.seed(settings["seed"])
    obs, task_indices = get_obs_batch(
        batch_size=settings["num_processes"],
        obs_space=observation_subspace,
        num_tasks=settings["num_tasks"],
    )

    # Get output, compute a dummy loss, and perform backwards call.
    output = meta_network(obs, task_indices)
    loss = torch.sum(output ** 2)
    meta_network.zero_grad()
    loss.backward()

    # Check that gradients of alpha values are non-zero.
    batch_tasks = task_indices.tolist()
    for layer in range(meta_network.num_layers):
        for task in range(meta_network.num_test_tasks):
            grad = meta_network.alpha[layer].grad[:, task]
            assert grad is not None
            if task in batch_tasks:
                assert torch.all(grad != 0)
            else:
                assert torch.all(grad == 0)

    # Check that gradients of regions are zero.
    for region in range(meta_network.num_regions):
        for copy in range(int(meta_network.splitting_map.num_copies[region])):
            for param in meta_network.regions[region][copy].parameters():
                assert param.grad is None


def get_flattened_grads(
    task_grads: torch.Tensor,
    num_tasks: int,
    region_sizes: List[int],
    begin: int,
    end: int,
):
    """
    Helper function to get flattened gradients from `task_grads`. Returns a tensor of
    size `(num_tasks, (end - begin) * sum(region_sizes))`.
    """

    flattened_grads = []
    for task in range(num_tasks):
        task_grad = []
        for region, region_size in enumerate(region_sizes):
            region_grad = task_grads[begin:end, task, region, :region_size]
            region_grad = region_grad.reshape(-1)
            task_grad.append(region_grad)
        task_grad = torch.cat(task_grad)
        flattened_grads.append(task_grad)
    return torch.stack(flattened_grads)
