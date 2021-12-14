""" Unit tests for meta/train/optimize.py. """

import torch
from torch import nn

from meta.networks import MLPNetwork, ConvNetwork
from meta.train.optimize import get_PSI_optimizer


NUM_NETWORKS = 10
NUM_STEPS = 10
HIDDEN_SIZE = 8
BATCH_SIZE = 8
LR = 1e-4
TOL = 5e-4
MIN_SCALE = 0.5
OUTPUT_SIZE = 5

MLP_INPUT_SIZE = 10
MLP_NUM_LAYERS = 4

CONV_INPUT_SIZE = (3, 5, 5)
CONV_BACKBONE_LAYERS = 2
CONV_HEAD_LAYERS = 2
CONV_CHANNELS = 8

MOMENTUM = 0.9


def test_PSI_invariance_mlp():
    """
    Test that the PSI optimizer preserves equivalence of a set of MLP networks with
    scaled weights over multiple optimization steps.
    """

    # Initialize networks, set the weights/biases of the non-first networks to a scaled
    # version of the weights/biases of the first network, and set all other parameters
    # equal to that of the first network.
    networks = []
    for _ in range(NUM_NETWORKS):
        networks.append(
            MLPNetwork(
                input_size=MLP_INPUT_SIZE,
                output_size=OUTPUT_SIZE,
                num_layers=MLP_NUM_LAYERS,
                hidden_size=HIDDEN_SIZE,
                batch_norm=True,
            )
        )

    original = networks[0]
    for i in range(1, NUM_NETWORKS):

        copy = networks[i]
        scaled_param_names = []

        for l in range(copy.num_layers - 1):
            original_layer = original.layers[l]
            copy_layer = copy.layers[l]
            assert isinstance(original_layer[0], nn.Linear)
            assert isinstance(original_layer[1], nn.BatchNorm1d)
            assert isinstance(copy_layer[0], nn.Linear)
            assert isinstance(copy_layer[1], nn.BatchNorm1d)

            # Scale parameters of copy network.
            num_neurons = original_layer[0].out_features
            scale = torch.rand(num_neurons) + MIN_SCALE
            copy_layer[0].weight.data = original_layer[0].weight.data * scale.unsqueeze(
                -1
            )
            copy_layer[0].bias.data = original_layer[0].bias.data * scale

            # Save names of scaled params.
            name_prefix = f"layers.{l}.0."
            scaled_param_names.append(name_prefix + "weight")
            scaled_param_names.append(name_prefix + "bias")

        # Copy over remaining parameters without scaling.
        original_state = original.state_dict()
        for name, p in copy.named_parameters():
            if name not in scaled_param_names:
                p.data = original_state[name]

    # Initialize dummy training data and loss function.
    inputs = torch.normal(
        mean=torch.zeros(BATCH_SIZE, MLP_INPUT_SIZE),
        std=torch.ones(BATCH_SIZE, MLP_INPUT_SIZE),
    )
    outputs = torch.normal(
        mean=torch.zeros(BATCH_SIZE, OUTPUT_SIZE),
        std=torch.ones(BATCH_SIZE, OUTPUT_SIZE),
    )
    criterion = nn.MSELoss()

    # Check that networks are equivalent. If this isn't the case, the test is invalid.
    original_output = original(inputs)
    for i in range(1, NUM_NETWORKS):
        output = networks[i](inputs)
        assert torch.allclose(output, original_output, atol=TOL)

    # Initialize PSI optimizer for each network.
    optimizers = [get_PSI_optimizer(net, LR, momentum=0) for net in networks]

    # Update each of the networks with the PSI optimizer.
    for step in range(NUM_STEPS):
        for i in range(NUM_NETWORKS):
            net = networks[i]
            opt = optimizers[i]

            opt.zero_grad()
            preds = networks[i](inputs)
            loss = criterion(preds, outputs)
            loss.backward()
            opt.step()

    # Check that networks are still equivalent and different from original networks.
    new_output = original(inputs)
    for i in range(1, NUM_NETWORKS):
        output = networks[i](inputs)
        assert torch.allclose(output, new_output, atol=TOL)

    assert not torch.allclose(original_output, new_output, atol=TOL)


def test_PSI_invariance_conv():
    """
    Test that the PSI optimizer preserves equivalence of a set of convolutional networks
    with scaled weights over multiple optimization steps.
    """

    # Initialize networks, set the weights/biases of the non-first networks to a scaled
    # version of the weights/biases of the first network, and set all other parameters
    # equal to that of the first network.
    networks = []
    for _ in range(NUM_NETWORKS):
        networks.append(
            ConvNetwork(
                input_size=CONV_INPUT_SIZE,
                num_conv_layers=CONV_BACKBONE_LAYERS,
                initial_channels=CONV_CHANNELS,
                num_fc_layers=CONV_HEAD_LAYERS,
                fc_hidden_size=HIDDEN_SIZE,
                output_size=OUTPUT_SIZE,
                batch_norm=True,
            )
        )

    original = networks[0]
    for i in range(1, NUM_NETWORKS):

        copy = networks[i]
        scaled_param_names = []

        for l in range(copy.num_conv_layers):
            original_layer = original.conv[l]
            copy_layer = copy.conv[l]
            assert isinstance(original_layer[0], nn.Conv2d)
            assert isinstance(original_layer[1], nn.BatchNorm2d)
            assert isinstance(copy_layer[0], nn.Conv2d)
            assert isinstance(copy_layer[1], nn.BatchNorm2d)

            # Scale parameters of copy network.
            num_neurons = original_layer[0].out_channels
            scale = torch.rand(num_neurons) + MIN_SCALE
            copy_layer[0].weight.data = original_layer[0].weight.data * scale.view(
                num_neurons, 1, 1, 1
            )
            copy_layer[0].bias.data = original_layer[0].bias.data * scale

            # Save names of scaled params.
            name_prefix = f"conv.{l}.0."
            scaled_param_names.append(name_prefix + "weight")
            scaled_param_names.append(name_prefix + "bias")

        for l in range(copy.num_fc_layers - 1):
            original_layer = original.fc[l]
            copy_layer = copy.fc[l]
            assert isinstance(original_layer[0], nn.Linear)
            assert isinstance(original_layer[1], nn.BatchNorm1d)
            assert isinstance(copy_layer[0], nn.Linear)
            assert isinstance(copy_layer[1], nn.BatchNorm1d)

            # Scale parameters of copy network.
            num_neurons = original_layer[0].out_features
            scale = torch.rand(num_neurons) + MIN_SCALE
            copy_layer[0].weight.data = original_layer[0].weight.data * scale.unsqueeze(
                -1
            )
            copy_layer[0].bias.data = original_layer[0].bias.data * scale

            # Save names of scaled params.
            name_prefix = f"fc.{l}.0."
            scaled_param_names.append(name_prefix + "weight")
            scaled_param_names.append(name_prefix + "bias")

        # Copy over remaining parameters without scaling.
        original_state = original.state_dict()
        for name, p in copy.named_parameters():
            if name not in scaled_param_names:
                p.data = original_state[name]

    # Initialize dummy training data and loss function.
    inputs = torch.normal(
        mean=torch.zeros(BATCH_SIZE, *CONV_INPUT_SIZE),
        std=torch.ones(BATCH_SIZE, *CONV_INPUT_SIZE),
    )
    outputs = torch.normal(
        mean=torch.zeros(BATCH_SIZE, OUTPUT_SIZE),
        std=torch.ones(BATCH_SIZE, OUTPUT_SIZE),
    )
    criterion = nn.MSELoss()

    # Check that networks are equivalent. If this isn't the case, the test is invalid.
    original_output = original(inputs)
    for i in range(1, NUM_NETWORKS):
        output = networks[i](inputs)
        assert torch.allclose(output, original_output, atol=TOL)

    # Initialize PSI optimizer for each network.
    optimizers = [get_PSI_optimizer(net, LR, momentum=0) for net in networks]

    # Update each of the networks with the PSI optimizer.
    for step in range(NUM_STEPS):
        for i in range(NUM_NETWORKS):
            net = networks[i]
            opt = optimizers[i]

            opt.zero_grad()
            preds = networks[i](inputs)
            loss = criterion(preds, outputs)
            loss.backward()
            opt.step()

    # Check that networks are still equivalent and different from original networks.
    new_output = original(inputs)
    for i in range(1, NUM_NETWORKS):
        output = networks[i](inputs)
        assert torch.allclose(output, new_output, atol=TOL)

    assert not torch.allclose(original_output, new_output, atol=TOL)


def test_PSI_invariance_mlp_momentum():
    """
    Test that the PSI optimizer preserves equivalence of a set of MLP networks with
    scaled weights over multiple optimization steps with momentum.
    """

    # Initialize networks, set the weights/biases of the non-first networks to a scaled
    # version of the weights/biases of the first network, and set all other parameters
    # equal to that of the first network.
    networks = []
    for _ in range(NUM_NETWORKS):
        networks.append(
            MLPNetwork(
                input_size=MLP_INPUT_SIZE,
                output_size=OUTPUT_SIZE,
                num_layers=MLP_NUM_LAYERS,
                hidden_size=HIDDEN_SIZE,
                batch_norm=True,
            )
        )

    original = networks[0]
    for i in range(1, NUM_NETWORKS):

        copy = networks[i]
        scaled_param_names = []

        for l in range(copy.num_layers - 1):
            original_layer = original.layers[l]
            copy_layer = copy.layers[l]
            assert isinstance(original_layer[0], nn.Linear)
            assert isinstance(original_layer[1], nn.BatchNorm1d)
            assert isinstance(copy_layer[0], nn.Linear)
            assert isinstance(copy_layer[1], nn.BatchNorm1d)

            # Scale parameters of copy network.
            num_neurons = original_layer[0].out_features
            scale = torch.rand(num_neurons) + MIN_SCALE
            copy_layer[0].weight.data = original_layer[0].weight.data * scale.unsqueeze(
                -1
            )
            copy_layer[0].bias.data = original_layer[0].bias.data * scale

            # Save names of scaled params.
            name_prefix = f"layers.{l}.0."
            scaled_param_names.append(name_prefix + "weight")
            scaled_param_names.append(name_prefix + "bias")

        # Copy over remaining parameters without scaling.
        original_state = original.state_dict()
        for name, p in copy.named_parameters():
            if name not in scaled_param_names:
                p.data = original_state[name]

    # Initialize dummy training data and loss function.
    inputs = torch.normal(
        mean=torch.zeros(BATCH_SIZE, MLP_INPUT_SIZE),
        std=torch.ones(BATCH_SIZE, MLP_INPUT_SIZE),
    )
    outputs = torch.normal(
        mean=torch.zeros(BATCH_SIZE, OUTPUT_SIZE),
        std=torch.ones(BATCH_SIZE, OUTPUT_SIZE),
    )
    criterion = nn.MSELoss()

    # Check that networks are equivalent. If this isn't the case, the test is invalid.
    original_output = original(inputs)
    for i in range(1, NUM_NETWORKS):
        output = networks[i](inputs)
        assert torch.allclose(output, original_output, atol=TOL)

    # Initialize PSI optimizer for each network.
    optimizers = [get_PSI_optimizer(net, LR, momentum=MOMENTUM) for net in networks]

    # Update each of the networks with the PSI optimizer.
    for step in range(NUM_STEPS):
        for i in range(NUM_NETWORKS):
            net = networks[i]
            opt = optimizers[i]

            opt.zero_grad()
            preds = networks[i](inputs)
            loss = criterion(preds, outputs)
            loss.backward()
            opt.step()

    # Check that networks are still equivalent and different from original networks.
    new_output = original(inputs)
    for i in range(1, NUM_NETWORKS):
        output = networks[i](inputs)
        assert torch.allclose(output, new_output, atol=TOL)

    assert not torch.allclose(original_output, new_output, atol=TOL)
