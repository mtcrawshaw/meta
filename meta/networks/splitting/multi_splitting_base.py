"""
Definition of BaseMultiTaskSplittingNetwork, the base class used to represent a
multi-task splitting network.
"""

import pickle
from copy import deepcopy
from typing import List, Dict, Any, Union, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from meta.networks.utils import get_fc_layer, get_conv_layer, init_base, init_downscale
from meta.utils.estimate import RunningStats
from meta.utils.logger import logger


ARCH_TYPES = ["fc", "conv"]


class BaseMultiTaskSplittingNetwork(nn.Module):
    """
    Base class used to represent a splitting MLP. This class shouldn't be used for
    training, as it will raise NotImplementedError when `check_for_split()` is called.
    Only extensions of this class should be used for training.
    """

    def __init__(
        self,
        input_size: Union[int, Tuple[int, int, int]],
        output_size: Union[int, List[int]],
        num_tasks: int = None,
        arch_type: str = "fc",
        num_layers: int = 3,
        width: int = 64,
        activation: str = "tanh",
        split_step_threshold: int = 30,
        sharing_threshold: float = 0.1,
        metric: str = "sqeuclidean",
        cap_sample_size: bool = True,
        ema_alpha: float = 0.999,
        task_specific_layers: int = 0,
        downscale_last_layer: bool = False,
        network_load: Dict[str, Any] = None,
        device: torch.device = None,
    ) -> None:
        """
        Init function for BaseMultiTaskSplittingNetwork.

        Arguments
        ---------
        input_size : Union[int, Tuple[int, int, int]]
            Input size for first layer of network. This can be an integer (vector input)
            or a tuple of 3 integers (image input). When input space is 3D, order of
            dimensions is [channels, width, height].
        output_size : Union[int, List[int]]
            Output size for last layer of network, which can be specified as a single
            integer (if all tasks have the same output size) or as a list of integers.
            If the output size is provided as a list of ints, then the number of tasks
            is inferred from the length of this list, and in this case `num_tasks`
            doesn't have to be provided. If `num_tasks` is provided in this case, it
            must match the length of `output_size`. We currently only support vector
            valued outputs.
        num_tasks : int
            Number of tasks that we are training over. Not required if `output_size` is
            a list, since the number of tasks is inferred from the length of this list.
        arch_type : str
            Type of architecture for each layer. Options are listed in `ARCH_TYPES`.
        num_layers : int
            Number of layers in network.
        width : int
            Hidden size for network in fully connected case, number of initial
            convolutional channels in convolutional case.
        split_step_threshold : int
            Number of updates before any splitting is performed. This is in place to
            make sure that we don't perform any splits based on a tiny amount of data.
        sharing_threshold : float
            Sharing score that the network can reach before splitting is disabled. The
            sharing score is computed by `self.get_sharing_score()`.
        metric : str
            Whether to use Euclidean distance or cosine distance to compare task
            gradients. Should be either "sqeuclidean" or "cosine".
        cap_sample_size : bool
            Whether or not to stop increasing the sample size when we switch to EMA.
        ema_alpha : float
            Coefficient used to compute exponential moving averages.
        task_specific_layers : int
            Number of layers at the end of the network that are task-specific. Note: if
            this value is non-zero, the network will not add any layers after the
            `num_layers` splitting layers. Instead, the last `task_specific_layers` of
            the splitting layers will be set to task-specific. Note that this argument
            is overriden if a network architecture is loaded with `network_load`.
        network_load : Dict[str, Any]
            Dictionary holding instructions about loading a pre-split architecture in.
            If no loading should be done (as is the case for the traditional algorithm
            which trains from a fully shared network), this value should be None.
            Otherwise, a pre-split architecture will be loaded and no splitting will be
            performed during training. The key/value pairs that should be in this
            dictionary to perform this are described below:
                checkpoint_path : str
                    Path to a checkpoint file from a previous training run of a
                    splitting network. The architecture will be loaded from this
                    splitting file.
                network_type : str
                    Either "actor" or "critic". This determines which network
                    architecture within the checkpoint (namely, actor or critic) to load
                    from.
                load_initialization : bool
                    Whether or not to load the initial weights of the previous network.
                    Note that if this is true, we aren't loading the weights produced by
                    training, we are literally loading the randomly initialized weights
                    from the beginning of the training run we are loading from. We do
                    this because the splitting architecture that it learned may be
                    initialization sensitive, in which case it would help to begin our
                    new training with that architecture from the same initialization.
            This is incredibly janky but it's what we got for now.
        device : torch.device
            Device to perform computation on, either `torch.device("cpu")` or
            `torch.device("cuda:0")`.
        """

        super(BaseMultiTaskSplittingNetwork, self).__init__()

        # Check for valid arguments.
        if num_layers < 1:
            raise ValueError(
                "Number of layers in network should be at least 1. Given value is: %d"
                % num_layers
            )
        if task_specific_layers > num_layers:
            raise ValueError(
                f"There must be at least as many total layers ({num_layers}) as"
                f" task-specific layers ({task_specific_layers})."
            )
        if arch_type not in ARCH_TYPES:
            raise ValueError(
                "Architecture '%s' not supported for MultiTaskSplittingNetwork."
                % arch_type
            )
        if isinstance(input_size, int):
            if arch_type not in ["fc"]:
                raise ValueError(f"Unsupported arch_type for 1D inputs: {arch_type}")
        elif isinstance(input_size, tuple):
            if len(input_size) != 3:
                raise ValueError(f"Unsupported input size: {input_size}")
            if arch_type not in ["conv"]:
                raise ValueError(f"Unsupported arch_type for 3D inputs: {arch_type}")
        else:
            raise ValueError(f"Unsupported input type: {type(input_size)}")
        if isinstance(output_size, int):
            if num_tasks is None:
                raise ValueError(
                    "Number of tasks must be provided either through `num_tasks` "
                    "argument or by specifying `output_size` as a list whose length "
                    "is equal to the number of tasks."
                )
        elif isinstance(output_size, list):
            if num_tasks is not None and len(output_size) != num_tasks:
                raise ValueError(
                    f"Argument `num_tasks` ({num_tasks}) doesn't match length of "
                    "`output_size` ({len(output_size)})."
                )
            num_tasks = len(output_size)

            # TEMP: Right now, we only support sets of tasks that have the same output
            # sizes.
            out = output_size[0]
            for x in output_size:
                assert isinstance(x, int) and x == out

        else:
            raise ValueError(f"Unsupported output type: {type(output_size)}")
        if not isinstance(output_size, int) and not isinstance(output_size, List):
            raise ValueError(f"Unsupported output type: {type(output_size)}")

        # Set state.
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.num_tasks = num_tasks
        self.arch_type = arch_type
        self.num_layers = num_layers
        self.width = width
        self.split_step_threshold = split_step_threshold
        self.sharing_threshold = sharing_threshold
        self.metric = metric
        self.cap_sample_size = cap_sample_size
        self.ema_alpha = ema_alpha
        self.task_specific_layers = task_specific_layers
        self.network_load = network_load
        self.downscale_last_layer = downscale_last_layer

        # Set device.
        self.device = device if device is not None else torch.device("cpu")

        # Generate network layers.
        self.initialize_network()

        # Initialize running estimates of pairwise differences of task gradients.
        self.grad_diff_stats = RunningStats(
            shape=(self.num_tasks, self.num_tasks, self.num_regions),
            cap_sample_size=self.cap_sample_size,
            ema_alpha=self.ema_alpha,
        )

        # Move model to device.
        self.to(self.device)

        self.num_steps = 0

    def initialize_network(self) -> None:
        """ Initialize layers of network. """

        # Initialize splitting map. Defines the parameter sharing structure between
        # tasks.
        self.num_regions = self.num_layers
        self.splitting_map = SplittingMap(
            self.num_tasks, self.num_regions, device=self.device
        )
        self.splitting_enabled = True

        # Make the output heads task-specific, if necessary.
        if self.task_specific_layers > 0:
            for i in range(self.task_specific_layers):
                layer = self.num_layers - self.task_specific_layers + i
                self.splitting_map.num_copies[layer] = self.num_tasks
                for t in range(self.num_tasks):
                    self.splitting_map.copy[layer, t] = t

        # Load splitting map, if necessary. If so, we disable splitting.
        initial_state_dict = None
        if self.network_load is not None:

            # Load previous network from checkpoint.
            checkpoint_path = self.network_load["checkpoint_path"]
            network_type = self.network_load["network_type"]
            with open(checkpoint_path, "rb") as checkpoint_file:
                checkpoint = pickle.load(checkpoint_file)
            if network_type == "actor":
                prev_network = checkpoint["policy"].policy_network.actor
            elif network_type == "critic":
                prev_network = checkpoint["policy"].policy_network.critic
            else:
                raise NotImplementedError

            assert isinstance(prev_network, BaseMultiTaskSplittingNetwork)
            assert self.num_tasks == prev_network.num_tasks
            assert self.num_regions == prev_network.num_regions
            self.splitting_map = prev_network.splitting_map
            self.splitting_enabled = False
            if self.network_load["load_initialization"]:
                initial_state_dict = prev_network.initial_state_dict

        # Initialize layers.
        region_list = []
        for i in range(self.num_layers):

            # Determine init function for layer.
            if i == self.num_layers - 1 and self.downscale_last_layer:
                layer_init = init_downscale
            else:
                layer_init = init_base

            if self.arch_type == "fc":

                # Calcuate input and output size of layer.
                if isinstance(self.output_size, int):
                    out = self.output_size
                elif isinstance(self.output_size, List):
                    out = self.output_size[0]
                else:
                    raise NotImplementedError
                layer_input_size = self.input_size if i == 0 else self.width
                layer_output_size = out if i == self.num_layers - 1 else self.width

                # Initialize copies of layer.
                layer_list = [
                    get_fc_layer(
                        in_size=layer_input_size,
                        out_size=layer_output_size,
                        activation=self.activation
                        if i != self.num_layers - 1
                        else None,
                        layer_init=layer_init,
                    )
                    for _ in range(int(self.splitting_map.num_copies[i]))
                ]

            elif self.arch_type == "conv":

                if i < self.num_layers - 1:

                    # Calculate input and output channels of layer.
                    layer_in_channels = self.input_size[0] if i == 0 else self.width
                    layer_out_channels = self.width

                    # Inititalize copies of layer.
                    layer_list = [
                        get_conv_layer(
                            in_channels=layer_in_channels,
                            out_channels=layer_out_channels,
                            activation=self.activation,
                            layer_init=layer_init,
                        )
                        for _ in range(int(self.splitting_map.num_copies[i]))
                    ]

                else:

                    # Initialize copies of final layer. Note that we insert an adaptive
                    # average pooling and flattening before the final fully connected
                    # layer, following the ResNet style output.
                    if isinstance(self.output_size, int):
                        out = self.output_size
                    elif isinstance(self.output_size, list):
                        out = self.output_size[0]
                    else:
                        raise NotImplementedError
                    layer_list = [
                        nn.Sequential(
                            nn.AdaptiveAvgPool2d((1, 1)),
                            nn.Flatten(1),
                            get_fc_layer(
                                in_size=self.width,
                                out_size=out,
                                activation=None,
                                layer_init=layer_init,
                            ),
                        )
                        for _ in range(int(self.splitting_map.num_copies[i]))
                    ]

            else:
                raise NotImplementedError

            region = nn.ModuleList(layer_list)
            region_list.append(region)

        self.regions = nn.ModuleList(region_list)

        # Load initialization from previous run, if necessary. Note that the
        # architecture of the original initialization may be different from the
        # architecture of this initialization, but that's okay. The only differences
        # should be that the current architecture is a split version of the previous
        # architecture. This means that we can simply copy the weights of any module
        # that was split and assign those weights to each copy of that module in the
        # current architecture.
        if initial_state_dict is not None:
            state_dict = {}
            for i in range(self.num_layers):
                for m in range(int(self.splitting_map.num_copies[i])):
                    for p in ["weight", "bias"]:
                        old_key = "regions.%d.0.0.%s" % (i, p)
                        key = "regions.%d.%d.0.%s" % (i, m, p)
                        if m == 0:
                            state_dict[key] = initial_state_dict[old_key]
                        else:
                            state_dict[key] = np.copy(initial_state_dict[old_key])

            # Convert values from numpy to torch Tensors.
            for key, val in state_dict.items():
                state_dict[key] = torch.Tensor(val)
            self.load_state_dict(state_dict)

        # Store size of each region. We use this info to initialize tensors that hold
        # gradients with respect to specific regions.
        self.region_sizes = torch.Tensor(
            [
                sum([param.nelement() for param in self.regions[i][0].parameters()])
                for i in range(self.num_regions)
            ]
        )
        self.region_sizes = self.region_sizes.to(dtype=torch.long, device=self.device)
        self.max_region_size = int(max(self.region_sizes))
        self.total_region_size = int(sum(self.region_sizes))

        # Store network initialization. We save the values as numpy arrays so that we
        # eliminate the torch/device dependence.
        self.initial_state_dict = {}
        for key, val in self.state_dict().items():
            self.initial_state_dict[key] = np.copy(val.numpy())

    def forward(
        self, inputs: torch.Tensor, task_indices: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass definition for BaseMultiTaskSplittingNetwork. For each layer of the
        network, we aggregate the inputs by their assigned copy of each region, and pass
        the inputs through the corresponding copy. When `task_indices` is given, the
        model produces (for each input) a prediction for only a single task (as in
        multi-task reinforcement learning). Otherwise, the model produces (for each
        input) a prediction for all tasks (as is common in multi-task supervised
        learning).

        Implementation note: As I see it, there are two ways that we can reasonably
        implement this function in the case that `task_indices` is not None. The first
        is, at each region, sorting the inputs by copy index, batching the inputs by
        copy, performing one forward pass per copy, and restoring the original order of
        the inputs. The second is by, just once at the beginning of the forward pass,
        sorting the inputs by task, then performing one forward pass per task at each
        region, then restoring the original order of the inputs just once at the end of
        the forward pass. The trade off between them is that the first involves
        sorting/unsorting at each region but only uses one forward pass per copy per
        region, while the second only sorts/unsorts once but performs one forward pass
        per task per region. We have used the first method here, in the name of
        minimizing the number of forward passes (which are likely going to add more
        computational burden than sorting), but I could see an argument for the second
        implementation.

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

        batch_size = inputs.shape[0]

        if task_indices is not None:
            assert len(inputs) == len(task_indices)

            # Pass through each splitting layer.
            x = inputs
            for layer in range(self.num_layers):

                # Sort the inputs by copy index and construct reverse permutation to later
                # restore the original order.
                copy_indices = self.splitting_map.copy[layer, task_indices]
                sorted_copy_indices, copy_permutation = torch.sort(copy_indices)
                _, copy_reverse = torch.sort(copy_permutation)
                x = x[copy_permutation]

                # Pass through each copy of the region and stack outputs.
                copy_outputs = []
                for copy in range(int(self.splitting_map.num_copies[layer])):
                    batch_indices = (sorted_copy_indices == copy).nonzero().squeeze(-1)
                    batch = x[batch_indices]
                    copy_outputs.append(self.regions[layer][copy](batch))
                x = torch.cat(copy_outputs)

                # Restore original order of inputs.
                x = x[copy_reverse]

            return x

        else:

            # Pass through each splitting layer.
            layer_inputs = torch.stack(
                [inputs.clone() for _ in range(self.num_tasks)], dim=0
            )
            for layer in range(self.num_layers):

                layer_outputs = None
                for copy in range(int(self.splitting_map.num_copies[layer])):

                    # Get tasks which are routed through the current copy, and aggregate
                    # the activations for those tasks.
                    copy_tasks = (
                        (self.splitting_map.copy[layer] == copy).nonzero().squeeze(-1)
                    )
                    copy_inputs = layer_inputs[copy_tasks]

                    # Flatten task dimension of inputs, feed through copy, then
                    # un-flatten task dimension.
                    num_tasks, num_inputs = copy_inputs.shape[:2]
                    batch = copy_inputs.view(
                        num_tasks * num_inputs, *copy_inputs.shape[2:]
                    )
                    copy_outputs = self.regions[layer][copy](batch)
                    task_outputs = copy_outputs.view(
                        num_tasks, num_inputs, *copy_outputs.shape[1:]
                    )

                    # Store task activations.
                    if layer_outputs is None:
                        layer_outputs = torch.zeros(
                            self.num_tasks, *task_outputs.shape[1:], device=self.device,
                        )
                    layer_outputs[copy_tasks] = task_outputs

                layer_inputs = layer_outputs.clone()

            # Switch first two dimensions, so that batch dimension still comes first.
            return torch.transpose(layer_outputs, 0, 1)

    def check_for_split(self, task_losses: torch.Tensor) -> bool:
        """
        Determine whether any splits should occur based on the task-specific losses from
        the current batch. To do this, we compute task-specific gradients for each task,
        update our running statistics measuring these gradients, then determine which
        regions should be split (if any) by calling self.determine_splits(), which is
        implemented differently for each subclass.
        """

        self.num_steps += 1

        # Stop splitting when the sharing score is sufficiently low.
        past_threshold = self.get_sharing_score() <= self.sharing_threshold
        if past_threshold or not self.splitting_enabled:
            return False

        # Compute task-specific gradients.
        task_grads = self.get_task_grads(task_losses)

        # Update running estimates of gradient statistics.
        self.update_grad_stats(task_grads)

        # Compute test statistics regarding difference of task gradient distributions,
        # though only if there are any task pairs whose joint sample size is larger than
        # `self.split_step_threshold`.
        split = False
        if torch.any(self.grad_diff_stats.num_steps >= self.split_step_threshold):
            should_split = self.determine_splits()
            split = self.perform_splits(should_split)

        if split:
            self.log_split()

        return split

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
            (self.num_tasks, self.num_regions, self.max_region_size),
            device=self.device,
        )

        for task in range(self.num_tasks):

            self.zero_grad()
            task_losses[task].backward(retain_graph=True)

            for region in range(self.num_regions):
                param_grad_list = []
                copy = int(self.splitting_map.copy[region, task])
                for param in self.regions[region][copy].parameters():
                    param_grad_list.append(param.grad.view(-1))
                region_grad = torch.cat(param_grad_list)
                task_grads[task, region, : len(region_grad)] = region_grad

        return task_grads

    def update_grad_stats(self, task_grads: torch.Tensor) -> None:
        """ Update our running estimates of pairwise gradient statistics. """

        # Get indices of tasks with non-zero gradients. A task will have zero gradients
        # when the current batch doesn't contain any data from that task, and in that
        # case we do not want to update the gradient stats for this task.
        task_flags = (task_grads.view(self.num_tasks, -1) != 0.0).any(dim=1)
        task_pair_flags = task_flags.unsqueeze(0) * task_flags.unsqueeze(1)
        task_pair_flags = task_pair_flags.unsqueeze(-1)
        task_pair_flags = task_pair_flags.expand(-1, -1, self.num_regions)

        # Compute pairwise differences between task-specific gradients and update
        # running stats.
        task_grad_diffs = self.get_task_grad_diffs(task_grads)
        self.grad_diff_stats.update(task_grad_diffs, task_pair_flags)

    def get_task_grad_diffs(self, task_grads: torch.Tensor) -> torch.Tensor:
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

        task_grad_diffs = torch.zeros(
            self.num_tasks, self.num_tasks, self.num_regions, device=self.device
        )

        # Check for a valid metric type.
        if self.metric not in ["sqeuclidean", "cosine"]:
            raise NotImplementedError

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

        # Convert Euclidean distance to cosine distance, if necessary.
        if self.metric == "cosine":

            # Collect pairs of gradient norms (per task pair) at each layer. This is
            # kind of janky because you can only use torch.cartesian_prod() with 1D
            # tensors.
            pair_norms = []
            region_norms = torch.sum(task_grads ** 2, axis=-1)
            for region in range(self.num_regions):
                pair_norms.append(
                    torch.cartesian_prod(
                        region_norms[:, region], region_norms[:, region]
                    )
                )
            pair_norms = torch.stack(pair_norms, axis=1)

            # Compute sum/product of gradient norms of each layer/task pair.
            pair_norm_sum = torch.sum(pair_norms, axis=-1)
            pair_norm_sum = pair_norm_sum.reshape(
                self.num_tasks, self.num_tasks, self.num_regions
            )
            pair_norm_prod = torch.prod(pair_norms, axis=-1)
            pair_norm_prod = pair_norm_prod.reshape(
                self.num_tasks, self.num_tasks, self.num_regions
            )

            # Subtract sum of grad norms from norm of grad difference, divide by sqrt of
            # product of grad norms, then divide by -2 to yield cosine distance.
            task_grad_diffs -= pair_norm_sum
            task_grad_diffs /= torch.sqrt(pair_norm_prod)
            task_grad_diffs /= -2.0

            # Rescale linearly so that 1 is mapped to 0 and -1 is mapped to 1. This way
            # the distances are the same order as Euclidean distance (lower means
            # closer) and a distance of zero means the values are identical.
            task_grad_diffs = (-task_grad_diffs + 1) / 2.0

            # Get rid of any infs or nans that may have appeared from zero division.
            bad_idxs = torch.isnan(task_grad_diffs) + torch.isinf(task_grad_diffs)
            task_grad_diffs[bad_idxs] = 0.0

        return task_grad_diffs

    def determine_splits(self) -> torch.Tensor:
        """
        Determine which regions (if any) should be split based on the current gradient
        statistics. This function is implemented differently for each subclass.
        """

        raise NotImplementedError

    def perform_splits(self, should_split: torch.Tensor) -> bool:
        """
        Perform any splits as determined by `should_split`. Returns true if any splits
        occur, and false otherwise.
        """

        # Perform any necessary splits. Notice that we only do this for if `task1,
        # task` current share the same copy of `region`, and if `task1 < task2`,
        # since we don't want to split for both coords (task1, task2, region) and
        # (task2, task1, region).
        split = False
        split_coords = should_split.nonzero()
        for task1, task2, region in split_coords:
            if (
                self.splitting_map.copy[region, task1]
                != self.splitting_map.copy[region, task2]
            ):
                continue
            if task1 >= task2:
                continue
            copy = self.splitting_map.copy[region, task1]

            # Partition tasks into groups by distance to task1 and task2. Notice
            # that we have to filter the groups of tasks so that we are only
            # including tasks that currently share the same copy of `region` with
            # `task1` and `task2`.
            group1 = (
                self.grad_diff_stats.mean[task1, :, region]
                < self.grad_diff_stats.mean[task2, :, region]
            ).nonzero()
            group1 = group1.squeeze(-1).tolist()
            group1 = [
                task for task in group1 if self.splitting_map.copy[region, task] == copy
            ]
            group2 = [
                task
                for task in range(self.num_tasks)
                if task not in group1 and self.splitting_map.copy[region, task] == copy
            ]

            # Ensure that `task1` and `task2` are assigned to the correct groups. This
            # can go wrong in the above code if multiple task gradient distances are
            # exactly the same.
            t1 = int(task1)
            t2 = int(task2)
            if t1 in group2:
                group2.remove(t1)
                group1.append(t1)
            if t2 in group1:
                group1.remove(t2)
                group2.append(t2)

            # Execute split.
            self.split(region, copy, group1, group2)
            split = True

        return split

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
        self.splitting_map.split(region, copy, group1, group2)

        # Create a new module and add to parameters.
        new_copy = deepcopy(self.regions[region][copy])
        self.regions[region].append(new_copy)

    def get_sharing_score(self) -> float:
        """
        Compute and return the sharing score of the network, which is a value ranging
        from 0 to 1. A score of 0 means that no parameters are shared between any tasks,
        and a score of 1 means that all parameters are shared between all tasks. The
        sharing score is roughly the degree of parameter sharing between all tasks.
        """

        region_scores = (self.num_tasks - self.splitting_map.num_copies) / (
            self.num_tasks - 1
        )
        sharing_score = torch.sum(region_scores * self.region_sizes)
        sharing_score /= self.total_region_size
        return sharing_score

    def log_split(self) -> None:
        """
        Log out network information.
        """

        msg = "Step %d\n" % self.num_steps
        msg += "Sharing score: %f\n" % self.get_sharing_score()
        msg += "Architecture:\n"
        msg += self.architecture_str() + "\n"

        msg += "\n"
        logger.log(msg)

    def architecture_str(self) -> str:
        """ Return a string representation of the current splitting architecture. """

        return self.splitting_map.architecture_str()


class SplittingMap:
    """
    Data structure used to encode the splitting structure of a splitting network.
    """

    def __init__(
        self, num_tasks: int, num_regions: int, device: torch.device = None
    ) -> None:
        """ Init function for SplittingMap. """

        self.num_tasks = num_tasks
        self.num_regions = num_regions
        self.device = device if device is not None else torch.device("cpu")
        self.num_copies = torch.ones(self.num_regions, device=self.device)
        self.copy = torch.zeros(
            self.num_regions, self.num_tasks, dtype=torch.long, device=self.device
        )

    def __eq__(self, other: "SplittingMap") -> bool:
        """ Definition of `self == other`. """

        eq = True
        eq = eq and self.num_tasks == other.num_tasks
        eq = eq and self.num_regions == other.num_regions
        eq = eq and torch.all(self.num_copies == other.num_copies)
        eq = eq and torch.all(self.copy == other.copy)
        return eq

    def split(
        self, region: int, copy: int, group_1: List[int], group_2: List[int]
    ) -> None:
        """
        Split copy `copy` at region `region` into two modules, one corresponding to
        tasks with inidices in `group_1` and the other to `group_2`. Note that to call
        this function, it must be that the combined indices of `group_1` and `group_2`
        form the set of indices i with self.module_map[i] = copy.
        """

        # Check that precondition is satisfied.
        assert set(group_1 + group_2) == set(
            [i for i in range(self.num_tasks) if self.copy[region, i] == copy]
        )

        self.num_copies[region] += 1
        for task in group_2:
            self.copy[region, task] = self.num_copies[region] - 1

    def shared_regions(self) -> torch.Tensor:
        """
        Returns a tensor of flags representing which regions are shared by which tasks.

        Returns
        -------
        is_shared : torch.Tensor
            Tensor of shape `(self.num_tasks, self.num_tasks, self.num_regions)` so that
            `is_shared[i, j, k]` holds 1/True if tasks `i, j` are shared at region `j`,
            and 0/False otherwise.
        """

        is_shared = torch.zeros(
            self.num_tasks, self.num_tasks, self.num_regions, device=self.device
        )

        # Janky way to compute the tensor described above as fast as I can think of.
        for region in range(self.num_regions):
            r = self.copy[region]
            is_shared[:, :, region] = (r.unsqueeze(0) + (-r).unsqueeze(1)) == 0
        is_shared *= (1.0 - torch.eye(self.num_tasks, device=self.device)).unsqueeze(-1)

        return is_shared

    def architecture_str(self) -> str:
        """ Return a string representation of the current splitting architecture. """

        msg = ""
        for region in range(self.num_regions):
            msg += "Region %d: " % region
            copies = [
                [
                    task
                    for task in range(self.num_tasks)
                    if self.copy[region, task] == copy
                ]
                for copy in range(int(self.num_copies[region]))
            ]
            msg += str(copies) + "\n"

        return msg
