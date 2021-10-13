""" Unit tests for meta/networks/parallel.py. """

import torch
import torch.nn as nn

from meta.networks.utils import Parallel


INPUT_SIZE = 10
OUTPUT_SIZE = 8
NUM_LAYERS = 5
BATCH_SIZE = 6


def test_new_dim():
    """
    Test that outputs from Parallel are correctly computed and stacked when
    `new_dim=True`.
    """

    # Create network.
    modules = [nn.Linear(INPUT_SIZE, OUTPUT_SIZE) for _ in range(NUM_LAYERS)]
    parallel = Parallel(modules, new_dim=True)

    # Construct batch of inputs.
    inputs = 2 * torch.rand((BATCH_SIZE, INPUT_SIZE)) - 1

    # Pass inputs through modules.
    outputs = parallel(inputs)

    # Check that outputs were correctly stacked.
    assert outputs.shape == (NUM_LAYERS, BATCH_SIZE, OUTPUT_SIZE)

    # Check that outputs were correctly computed.
    for layer in range(NUM_LAYERS):
        assert torch.allclose(outputs[layer], modules[layer](inputs))


def test_no_new_dim():
    """
    Test that outputs from Parallel are correctly computed and stacked when
    `new_dim=False`.
    """

    # Create network.
    modules = [nn.Linear(INPUT_SIZE, OUTPUT_SIZE) for _ in range(NUM_LAYERS)]
    parallel = Parallel(modules, new_dim=False)

    # Construct batch of inputs.
    inputs = 2 * torch.rand((BATCH_SIZE, INPUT_SIZE)) - 1

    # Pass inputs through modules.
    outputs = parallel(inputs)

    # Check that outputs were correctly stacked.
    assert outputs.shape == (NUM_LAYERS * BATCH_SIZE, OUTPUT_SIZE)

    # Check that outputs were correctly computed.
    for layer in range(NUM_LAYERS):
        layer_start = layer * BATCH_SIZE
        layer_end = (layer + 1) * BATCH_SIZE
        assert torch.allclose(outputs[layer_start:layer_end], modules[layer](inputs))
