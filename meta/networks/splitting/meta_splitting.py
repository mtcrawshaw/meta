"""
Definition of MetaSplittingNetwork, a trained BaseMultiTaskSplittingNetwork which will
be adapted at meta-test time.
"""

import torch
import torch.nn as nn

from meta.networks.splitting import BaseMultiTaskSplittingNetwork


class MetaSplittingNetwork(nn.Module):
    """ Module used to parameterize a splitting MLP at meta-test time. """

    def __init__(
        self, split_net: BaseMultiTaskSplittingNetwork, device: torch.device = None,
    ) -> None:
        """
        Init function for MetaTaskSplittingNetwork.

        Arguments
        ---------
        split_net : BaseMultiTaskSplittingNetwork
            A trained multi-task splitting network that will be used as the base for the
            meta splitting network represented by this object. We copy the weights and
            sharing structure from `split_net`, as well as settings such as input size,
            number of tasks, etc. It's important to note that we do not create copies of
            the parameters of `split_net`, we just save references to them. This means
            that after you pass a trained splitting network here to create a meta
            splitting network, you shouldn't use that trained splitting network for
            anything else, or things could get hairy.
        device : torch.device = None
            Device to perform computation on, either `torch.device("cpu")` or
            `torch.device("cuda:0")`.
        """

        super(MetaSplittingNetwork, self).__init__()

        # Copy state from `split_net`.
        self.input_size = split_net.input_size
        self.output_size = split_net.output_size
        self.num_tasks = split_net.num_tasks
        self.num_layers = split_net.num_layers
        self.hidden_size = split_net.hidden_size

        # Set device.
        self.device = device if device is not None else torch.device("cpu")

        # Set network weights.
        self.initialize_network(split_net)

        # Move model to device.
        self.to(self.device)

    def initialize_network(self, split_net: BaseMultiTaskSplittingNetwork) -> None:
        """ Initialize weights of network from the already trained `split_net`. """

        # Copy regions from `split_net`, and freeze them.
        self.num_regions = split_net.num_regions
        self.regions = split_net.regions
        for param in self.regions.parameters():
            param.requires_grad = False

        # Copy splitting map from `split_net`.
        self.splitting_map = split_net.splitting_map

        # Initialize alpha values. These are linear combination weights for each copy of
        # a region.
        alpha_list = []
        for i in range(self.num_regions):
            copies = int(self.splitting_map.num_copies[i])
            alpha_list.append(
                nn.Parameter(1.0 / copies * torch.ones(copies, self.num_tasks))
            )
        self.alpha = nn.ParameterList(alpha_list)

    def forward(self, inputs: torch.Tensor, task_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass definition for MetaSplittingNetwork. The output of each layer is a
        linear combination of the outputs of each copy of that layer, as defined by the
        `split_net` used to instantiate this network. The weights of the linear
        combination are learned per task, stored in `self.alpha`.

        Arguments
        ---------
        inputs : torch.Tensor
            Input to meta splitting network.
        task_indices : torch.Tensor
            Task index for each input in `inputs` as a integer.

        Returns
        -------
        outputs : torch.Tensor
            Output of meta splitting network when given `inputs` as input.
        """

        assert len(inputs) == len(task_indices)

        # Pass through each layer.
        x = inputs
        for layer in range(self.num_layers):

            # Pass through each copy of the region and stack outputs.
            copy_outputs = []
            for copy in range(int(self.splitting_map.num_copies[layer])):
                copy_outputs.append(self.regions[layer][copy](x))
            copy_outputs = torch.stack(copy_outputs)

            # Combine copy outputs with a linear combination.
            input_alphas = self.alpha[layer][:, task_indices].unsqueeze(-1)
            x = torch.sum(copy_outputs * input_alphas, dim=0)

        return x

    def architecture_str(self) -> str:
        """ Return a string representation of the current splitting architecture.  """

        return self.splitting_map.architecture_str()
