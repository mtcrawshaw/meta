"""
Playing around with Complete Knowledge Distillation.

Current state of the script: Approximate the difference between the outputs of two
neural networks with a single power series.
"""

import random
from math import factorial, pi
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import sympy
from sympy import expand, Symbol, Expr, Number, Add, Mul, Pow


INPUT_SIZE = 2
OUTPUT_SIZE = 2
NUM_TEACHER_LAYERS = 1
TEACHER_HIDDEN_SIZE = 1
NUM_STUDENT_LAYERS = 1
STUDENT_HIDDEN_SIZE = 1

MAX_PS_DEGREE = 3
BATCH_SIZE = 5

SEED = 0


class ExpActivation(nn.Module):
    """ Module that computes the element-wise exponential of the input. """

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Forward function for ExpActivation. """
        return torch.exp(inputs)


def double_factorial(i: int) -> int:
    """ Compute double factorial of i. """
    p = 1
    while i > 0:
        p *= i
        i -= 2
    return p


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


def get_monomial_terms(expr: Expr) -> List[Expr]:
    """
    Return a list of terms from a monomial. Example: x*y**2 -> [x, y**2]. Note that it
    is assumed that `is_monomial(expr)` would return True.
    """
    if is_constant(expr) or expr.func == Pow:
        return [expr]
    elif expr.func == Mul:
        return list(expr.args)
    else:
        # This should never happen.
        assert False


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
        last_pos = symbol.name.find("_", second_pos + 1)
        layer = int(symbol.name[1:first_pos])
        out_unit = int(symbol.name[first_pos + 1 : second_pos])
        in_unit = int(symbol.name[second_pos + 1 : last_pos])
        val = network[layer][0].weight[out_unit, in_unit]

    elif symbol.name.startswith("b"):
        first_pos = symbol.name.find("_")
        second_pos = symbol.name.find("_", first_pos + 1)
        layer = int(symbol.name[1:first_pos])
        out_unit = int(symbol.name[first_pos + 1 : second_pos])
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


def is_input_symbol(sym: Symbol) -> bool:
    """ Whether `sym` is an input symbol. """
    return isinstance(sym, Symbol) and sym.name.startswith("x")


def gaussian_integral(expr: Expr, d: int) -> Expr:
    """
    Symbolically computes the definite integral of `expr` as the input symbols xi vary
    according to the spherical unit Gaussian distribution over R^d. Note that `expr`
    must be a sum of monomials.
    """

    integral = 0

    # Check that `network_expr` is a sum of monomials.
    assert expr.func == Add
    for monomial in expr.args:
        assert is_monomial(monomial)

        # Compute integral iteratively over each term. Constants and weight/bias symbols
        # just get pulled out, input symbols with even power are integrated over. If
        # there are any input symbols with odd power, the integral of the entire
        # monomial is zero.
        current_integral = 1
        terms = get_monomial_terms(monomial)
        nonzero = True
        for term in terms:

            if term.func == Pow:
                base, power = term.args
                if is_input_symbol(base):
                    if power % 2 == 0:
                        j = power // 2
                        current_integral *= double_factorial(2*j - 1) / 2 ** j
                    else:
                        nonzero = False
                        break
                else:
                    current_integral *= term

            elif is_constant(term):
                if is_input_symbol(term):
                    nonzero = False
                    break
                else:
                    current_integral *= term

            else:
                # This should never happen.
                assert False

        if nonzero:
            current_integral *= pi ** (d / 2)
            integral += current_integral

    return integral


def evaluate_error(error_expr: Expr, networks: Dict[str, nn.Module]) -> float:
    """
    Evaluate `error_expr`. This expression should depend only on constants and symbols
    with names of the form `wi_j_k_<name>`, `bi_j_<name>`, which correspond to weights
    and biases from networks in `networks`. The name at the end of each symbol name
    should be a key in `networks`.
    """
    raise NotImplementedError


def get_network(
    input_size: int, output_size: int, hidden_size: int, num_layers: int
) -> nn.Module:
    """
    Construct an MLP with the given parameters and exponential activation units.
    """

    layers = []
    for i in range(num_layers):
        layer = []
        layer_in = input_size if i == 0 else hidden_size
        layer_out = output_size if i == num_layers - 1 else hidden_size
        layer.append(nn.Linear(layer_in, layer_out))
        if i < num_layers - 1:
            layer.append(ExpActivation())
        layers.append(nn.Sequential(*layer))
    return nn.Sequential(*layers)

def get_network_symbols(
    input_size: int, output_size: int, hidden_size: int, num_layers: int, name: str,
) -> Tuple[List[List[List[Symbol]]], List[List[Symbol]]]:
    """
    Construct SymPy symbols for network weights and biases. Returns nested lists of
    SymPy symbols representing weights and biases.
    """

    weight_symbols = []
    bias_symbols = []
    for i in range(num_layers):
        layer_in = input_size if i == 0 else hidden_size
        layer_out = output_size if i == num_layers - 1 else hidden_size
        weight_symbols.append([])
        bias_symbols.append([])
        for j in range(layer_out):
            weight_symbols[i].append([])
            for k in range(layer_in):
                weight_symbols[i][j].append(Symbol(f"w{i}_{j}_{k}_{name}"))
            bias_symbols[i].append(Symbol(f"b{i}_{j}_{name}"))

    return weight_symbols, bias_symbols

def main():
    """ Main function for ckd_sandbox.py. """

    # Set random seed.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # Construct teacher and student network.
    teacher = get_network(
        INPUT_SIZE, OUTPUT_SIZE, TEACHER_HIDDEN_SIZE, NUM_TEACHER_LAYERS
    )
    student = get_network(
        INPUT_SIZE, OUTPUT_SIZE, STUDENT_HIDDEN_SIZE, NUM_STUDENT_LAYERS
    )

    # Construct SymPy symbols for network inputs.
    input_symbols = [Symbol(f"x{i}") for i in range(INPUT_SIZE)]

    # Construct SymPy symbols for network inputs, weights, and biases.
    teacher_weight_symbols, teacher_bias_symbols = get_network_symbols(
        INPUT_SIZE, OUTPUT_SIZE, TEACHER_HIDDEN_SIZE, NUM_TEACHER_LAYERS, name="teacher"
    )
    student_weight_symbols, student_bias_symbols = get_network_symbols(
        INPUT_SIZE, OUTPUT_SIZE, STUDENT_HIDDEN_SIZE, NUM_STUDENT_LAYERS, name="student"
    )

    # Get power series representation of output for teacher and student, and their
    # difference.
    teacher_expr = network_to_expr(
        teacher, input_symbols, teacher_weight_symbols, teacher_bias_symbols
    )
    student_expr = network_to_expr(
        student, input_symbols, student_weight_symbols, student_bias_symbols
    )
    assert len(teacher_expr) == len(student_expr)
    error_expr = expand(sum((t - s) ** 2 for t, s in zip(teacher_expr, student_expr)))

    # Generate input data to test difference approximation.
    mean = torch.zeros((BATCH_SIZE, INPUT_SIZE))
    std = torch.ones((BATCH_SIZE, INPUT_SIZE))
    input_data = torch.normal(mean, std)

    # Get actual student error on batch.
    teacher_out = teacher(input_data)
    student_out = student(input_data)
    student_error = torch.sum((teacher_out - student_out) ** 2, dim=1)

    # Get estimated student error over entire data distribution.
    networks = {"teacher": teacher, "student": student}
    complete_error_expr = gaussian_integral(error_expr, INPUT_SIZE)
    print(complete_error_expr)
    complete_error = evaluate_error(complete_error_expr, networks)

    # Compare errors.
    avg_student_error = torch.mean(student_error).item()
    avg_error = (avg_student_error - complete_error) ** 2
    print(f"Actual student errors: {student_error}")
    print(f"Avg student error: {avg_student_error}")
    print(f"Complete error: {complete_error}")
    print(f"Diff: {avg_error}")


if __name__ == "__main__":
    main()
