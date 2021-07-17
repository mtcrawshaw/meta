"""
Playing around with Complete Knowledge Distillation.

Current state of the script: See how well we can approximate the output of a fully
connected neural network by a power series.
"""

from math import factorial
from typing import List

import torch
import torch.nn as nn
import sympy
from sympy import expand, Symbol, Expr, Number, Add, Mul, Pow


INPUT_SIZE = 2
OUTPUT_SIZE = 2
NUM_LAYERS = 3
HIDDEN_SIZE = 2
MAX_PS_DEGREE = 4
BATCH_SIZE = 5


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
                sum(inputs[k] * weights[i][j][k] for k in range(linear.in_features))
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
                    outputs[i] ** m / factorial(m) for m in range(MAX_PS_DEGREE)
                )

        inputs = outputs

    # Expand outputs.
    for i in range(len(outputs)):
        outputs[i] = expand(outputs[i])

    return outputs


def is_constant(expr: Expr) -> bool:
    """
    Whether or not `expr` is a constant, i.e. a Number or Symbol.
    """
    return any(issubclass(expr.func, cls) for cls in [Number, Symbol])


def is_monomial(expr: Expr) -> bool:
    """
    Whether or not `expr` is a monomial, i.e. a product of constants and symbols.
    """

    monomial = True

    if is_constant(expr):
        pass

    elif expr.func == Pow:
        monomial = is_monomial(expr.args[0]) and is_constant(expr.args[1])

    elif expr.func == Mul:
        for subexpr in expr.args:
            constant = is_constant(subexpr)
            constant_power = (
                subexpr.func == Pow
                and is_constant(subexpr.args[0])
                and is_constant(subexpr.args[1])
            )
            monomial = monomial and (constant or constant_power)
            if not monomial:
                break

    else:
        monomial = False

    return monomial


def evaluate_symbol(
    input_data: torch.Tensor, network: nn.Module, symbol: Symbol
) -> torch.Tensor:
    """
    Evaluate a symbol representing an input element or a network parameter, and return
    the corresponding torch Tensor.
    """

    if symbol.name.startswith("x"):
        idx = int(symbol.name[1:])
        val = input_data[idx]

    elif symbol.name.startswith("w"):
        first_pos = symbol.name.find("_")
        second_pos = symbol.name.find("_", first_pos + 1)
        layer = int(symbol.name[1:first_pos])
        out_unit = int(symbol.name[first_pos + 1 : second_pos])
        in_unit = int(symbol.name[second_pos + 1 :])
        val = network[layer][0].weight[out_unit, in_unit]

    elif symbol.name.startswith("b"):
        pos = symbol.name.find("_")
        layer = int(symbol.name[1:pos])
        out_unit = int(symbol.name[pos + 1 :])
        val = network[layer][0].bias[out_unit]

    else:
        raise ValueError(
            "Can only handle symbols with names starting with 'x', 'w', or 'b'."
        )

    return val


def evaluate_monomial(
    input_data: torch.Tensor, network: nn.Module, monomial: Expr
) -> torch.Tensor:
    """
    Numberically evaluate a monomial expression by substituting torch tensors in for the
    symbols, returning a torch Tensor.
    """

    if issubclass(monomial.func, Number):
        val = torch.Tensor([float(monomial)])

    elif monomial.func == Symbol:
        val = evaluate_symbol(input_data, network, monomial)

    elif monomial.func == Pow:
        base, exp = monomial.args
        if issubclass(base.func, Number):
            base_val = torch.Tensor([float(base)])
        else:
            base_val = evaluate_symbol(input_data, network, base)
        exp_val = torch.Tensor([float(exp)])
        val = torch.pow(base_val, exp_val)

    elif monomial.func == Mul:
        val = torch.Tensor([1])
        for subexpr in monomial.args:
            if issubclass(subexpr.func, Number):
                val *= torch.Tensor([float(subexpr)])
            elif subexpr.func == Symbol:
                val *= evaluate_symbol(input_data, network, subexpr)
            elif subexpr.func == Pow:
                base, exp = subexpr.args
                if issubclass(base.func, Number):
                    base_val = torch.Tensor([float(base)])
                else:
                    base_val = evaluate_symbol(input_data, network, base)
                exp_val = torch.Tensor([float(exp)])
                val *= torch.pow(base_val, exp_val)
            else:
                assert False
    else:
        assert False

    return val


def estimate_output(
    input_data: torch.Tensor, network: nn.Module, network_expr: List[Expr]
) -> torch.Tensor:
    """
    Estimate the output of the neural network whose approximate expression is
    `network_expr` by evaluating the expression on input `input_data`. Input data should
    have shape `(INPUT_SIZE,)`.
    """

    # Check that `network_expr` is a sum of monomials.
    for expr in network_expr:
        assert expr.func == Add
        for subexpr in expr.args:
            assert is_monomial(subexpr)

    # Construct output from network expression.
    estimated_output = torch.zeros(len(network_expr))
    for i, expr in enumerate(network_expr):
        for subexpr in expr.args:
            estimated_output[i] += evaluate_monomial(
                input_data, network, subexpr
            ).squeeze()

    return estimated_output


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
    estimated_output = torch.zeros_like(actual_output)
    for i in range(input_data.shape[0]):
        estimated_output[i] = estimate_output(input_data[i], network, network_expr)
    error = torch.mean((actual_output - estimated_output) ** 2).item()
    print(f"Mean error: {error}")


if __name__ == "__main__":
    main()
