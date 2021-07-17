"""
Playing around with Complete Knowledge Distillation.

Current state of the script: See how well we can approximate the output of a fully
connected neural network by a power series.
"""

from math import factorial
from typing import List

import torch
import torch.nn as nn
from sympy import Symbol, Expr, expand


INPUT_SIZE = 4
OUTPUT_SIZE = 2
NUM_LAYERS = 3
HIDDEN_SIZE = 3
MAX_PS_DEGREE = 3
BATCH_SIZE = 4


class ExpActivation(nn.Module):
    """ Module that computes the element-wise exponential of the input. """

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Forward function for ExpActivation. """
        return torch.exp(inputs)


def network_to_expr(
    network: nn.Sequential,
    inputs: List[Expr],
    weights: List[List[List[Symbol]]],
    biases: List[List[Symbol]],
) -> List[Expr]:
    """
    Convert the module `network` into an (approximately) equivalent power-series
    expression with SymPy. This expression is a list of polynomials in the input symbols
    and the network weights. Note that the input size to `network` must be equal to the
    length of `inputs`, and `network` must be an instance of `nn.Sequential`, whose
    submodules are also an `nn.Sequential` made of `[nn.Linear, ExpActivation]`, or
    `[nn.Linear]`.
    """

    assert isinstance(network, nn.Sequential)

    for i, layer in enumerate(network):

        assert isinstance(layer, nn.Sequential)
        assert len(layer) in [1, 2]

        # Check that input size of `layer` matches that the given list of inputs.
        linear = layer[0]
        assert isinstance(linear, nn.Linear)
        assert linear.in_features == len(inputs)

        # Compute expression for each output unit. Each output is the dot product of
        # the inputs with the weights of the corresponding output unit, plus the
        # bias of that unit.
        outputs = []
        for j in range(linear.out_features):
            outputs.append(
                sum(
                    inputs[k] * weights[i][j][k] for k in range(linear.in_features)
                )
                + biases[i][j]
            )

        if len(layer) == 2:

            activation = layer[1]
            assert isinstance(activation, ExpActivation)

            # Compute activation for each output unit. Each output is the exponential of
            # the corresponding sum, though this is approximated using the power-series
            # representation of the exponential function.
            for i in range(len(outputs)):
                outputs[i] = sum(
                    outputs[i] ** m / factorial(m)
                    for m in range(MAX_PS_DEGREE)
                )
                outputs[i] = expand(outputs[i])

        inputs = outputs

    return outputs


def estimate_output(input_data: torch.Tensor, network_expr: Expr) -> torch.Tensor:
    """
    Estimate the output of the neural network whose approximate expression is
    `network_expr` by evaluating the expression on input `input_data.
    """
    raise NotImplementedError


def main():
    """ Main function for ckd_sandbox.py. """

    # Construct network.
    layers = []
    for i in range(NUM_LAYERS):

        # Construct linear layer, adding activation if necessary.
        layer = []
        layer_in = INPUT_SIZE if i == 0 else HIDDEN_SIZE
        layer_out = OUTPUT_SIZE if i == NUM_LAYERS - 1 else HIDDEN_SIZE
        layer.append(nn.Linear(layer_in, layer_out))
        if i < NUM_LAYERS - 1:
            layer.append(ExpActivation())
        layers.append(nn.Sequential(*layer))

    network = nn.Sequential(*layers)

    # Construct SymPy symbols for network inputs, weights, and biases.
    input_symbols = [Symbol(f"x{i}") for i in range(INPUT_SIZE)]
    weight_symbols = []
    bias_symbols = []
    for i in range(NUM_LAYERS):
        layer_in = INPUT_SIZE if i == 0 else HIDDEN_SIZE
        layer_out = OUTPUT_SIZE if i == NUM_LAYERS - 1 else HIDDEN_SIZE
        weight_symbols.append([])
        bias_symbols.append([])
        for j in range(layer_out):
            weight_symbols[i].append([])
            for k in range(layer_in):
                weight_symbols[i][j].append(Symbol(f"w{i}_{j}_{k}"))
            bias_symbols[i].append(Symbol(f"b{i}_{j}"))

    # Get power series representation of output.
    network_expr = network_to_expr(network, input_symbols, weight_symbols, bias_symbols)

    # Generate input data to test network approximation.
    mean = torch.zeros((BATCH_SIZE, INPUT_SIZE))
    std = torch.ones((BATCH_SIZE, INPUT_SIZE))
    input_data = torch.normal(mean, std)

    # Get actual network output, estimated output, and compare.
    actual_output = network(input_data)
    estimated_output = estimate_output(input_data, network_expr)
    error = torch.mean((actual_output - estimated_output) ** 2).item()
    print(f"Mean error: {error}")


if __name__ == "__main__":
    main()
