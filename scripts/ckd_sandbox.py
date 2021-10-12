"""
Playing around with Complete Knowledge Distillation.

Current state of the script: Approximate the difference between the outputs of two
neural networks with a single power series.
"""

import os
import random
import time
import pickle
import argparse
from math import factorial, pi, ceil
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import sympy
from sympy import expand, Symbol, Expr, Number, Add, Mul, Pow


# Network parameters.
INPUT_SIZE = 3
OUTPUT_SIZE = 3
NUM_TEACHER_LAYERS = 2
TEACHER_HIDDEN_SIZE = 5
NUM_STUDENT_LAYERS = 2
STUDENT_HIDDEN_SIZE = 3

# Supervised KD parameters.
DATASET_SIZE = 1000
TRAIN_SPLIT = 0.9
NUM_UPDATES = 10000
BATCH_SIZE = 100

# Complete KD parameters.
MAX_PS_DEGREE = 4
LOSS_EXPR_PATH = "./data/ckd_loss_expr.pkl"

# General training parameters.
LR = 3e-6
MOMENTUM = 0.9
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


def evaluate_symbol(symbol: Symbol, networks: Dict[str, nn.Module]) -> torch.Tensor:
    """
    Evaluate a symbol representing a network parameter and return the corresponding
    torch Tensor. `symbol.name` should be of the form `wi_j_k_<name>` or `bi_j_<name>`,
    where <name> is a key in `networks`.
    """

    if symbol.name.startswith("w"):
        first_pos = symbol.name.find("_")
        second_pos = symbol.name.find("_", first_pos + 1)
        last_pos = symbol.name.find("_", second_pos + 1)
        layer = int(symbol.name[1:first_pos])
        out_unit = int(symbol.name[first_pos + 1 : second_pos])
        in_unit = int(symbol.name[second_pos + 1 : last_pos])
        name = symbol.name[last_pos + 1:]
        val = networks[name][layer][0].weight[out_unit, in_unit]

    elif symbol.name.startswith("b"):
        first_pos = symbol.name.find("_")
        second_pos = symbol.name.find("_", first_pos + 1)
        layer = int(symbol.name[1:first_pos])
        out_unit = int(symbol.name[first_pos + 1 : second_pos])
        name = symbol.name[second_pos + 1:]
        val = networks[name][layer][0].bias[out_unit]

    else:
        raise ValueError(
            "Can only handle symbols with names starting with 'w' or 'b'."
        )

    return val


def evaluate_monomial(monomial: Expr, networks: Dict[str, nn.Module]) -> torch.Tensor:
    """
    Numberically evaluate a monomial expression by substituting torch tensors in for the
    symbols, returning a torch Tensor.
    """

    if issubclass(monomial.func, Number):
        val = torch.Tensor([float(monomial)])

    elif monomial.func == Symbol:
        val = evaluate_symbol(monomial, networks)

    elif monomial.func == Pow:
        base, exp = monomial.args
        if issubclass(base.func, Number):
            base_val = torch.Tensor([float(base)])
        else:
            base_val = evaluate_symbol(base, networks)
        exp_val = torch.Tensor([float(exp)])
        val = torch.pow(base_val, exp_val)

    elif monomial.func == Mul:
        val = torch.Tensor([1])
        for subexpr in monomial.args:
            val *= evaluate_monomial(subexpr, networks)

    else:
        assert False

    return val


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
                        current_integral *= double_factorial(2*j - 1)
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
            # current_integral *= pi ** (d / 2)
            integral += current_integral

    return integral


def evaluate_error(error_expr: Expr, networks: Dict[str, nn.Module]) -> torch.Tensor:
    """
    Evaluate `error_expr` by substituting Symbols with the corresponding torch Tensors.
    This expression should depend only on constants and symbols with names of the form
    `wi_j_k_<name>`, `bi_j_<name>`, which correspond to weights and biases from networks
    in `networks`. The name at the end of each symbol name should be a key in
    `networks`.
    """

    # Check that `error_expr` is a sum of monomials, and sum evaluations of each
    # monomial.
    assert error_expr.func == Add
    error = 0
    for monomial in error_expr.args:
        assert is_monomial(monomial)
        error += evaluate_monomial(monomial, networks)

    return error


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


def run_kd(teacher: nn.Module, student: nn.Module, optimizer: torch.optim.Optimizer):
    """
    Runs knowledge distillation on a student and teacher network with supervised
    learning.
    """

    # Construct training data.
    train_size = round(DATASET_SIZE * TRAIN_SPLIT)
    test_size = DATASET_SIZE - train_size
    assert 0 < train_size < DATASET_SIZE
    train_data = torch.normal(
        torch.zeros(train_size, INPUT_SIZE),
        torch.ones(train_size, INPUT_SIZE),
    )
    test_data = torch.normal(
        torch.zeros(test_size, INPUT_SIZE),
        torch.ones(test_size, INPUT_SIZE),
    )

    # Construct data loaders.
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # Training loop.
    updates_per_epoch = ceil(train_size / BATCH_SIZE)
    num_epochs = ceil(NUM_UPDATES / updates_per_epoch)
    for epoch in range(num_epochs):

        # Run one training epoch.
        for step, batch in enumerate(iter(train_loader)):

            # Get student error on batch.
            teacher_out = teacher(batch)
            student_out = student(batch)
            student_error = torch.sum((teacher_out - student_out) ** 2, dim=1)
            loss = torch.mean(student_error)

            # Backwards pass to update student network.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Step {step} loss: {loss.item()}", end="\r")

        print("")

        # Evaluate on test set.
        batch_sizes = []
        test_losses = []
        for batch in iter(test_loader):

            # Get student error on batch.
            teacher_out = teacher(batch)
            student_out = student(batch)
            student_error = torch.sum((teacher_out - student_out) ** 2, dim=1)
            loss = torch.mean(student_error)
            test_losses.append(loss.item())
            batch_sizes.append(len(batch))

        test_loss = np.average(test_losses, weights=batch_sizes)
        print(f"Epoch {epoch} test loss: {test_loss}\n")


def run_complete_kd(teacher: nn.Module, student: nn.Module, optimizer: torch.optim.Optimizer):
    """ Runs complete knowledge distillation on a student and teacher network. """

    # Construct SymPy symbols for network inputs.
    input_symbols = [Symbol(f"x{i}") for i in range(INPUT_SIZE)]

    # Construct SymPy symbols for network inputs, weights, and biases.
    teacher_weight_symbols, teacher_bias_symbols = get_network_symbols(
        INPUT_SIZE, OUTPUT_SIZE, TEACHER_HIDDEN_SIZE, NUM_TEACHER_LAYERS, name="teacher"
    )
    student_weight_symbols, student_bias_symbols = get_network_symbols(
        INPUT_SIZE, OUTPUT_SIZE, STUDENT_HIDDEN_SIZE, NUM_STUDENT_LAYERS, name="student"
    )

    # Symbolically compute the student error over the entire input distribution as a
    # function of network parameters.
    networks = {"teacher": teacher, "student": student}
    if os.path.isfile(LOSS_EXPR_PATH):
        with open(LOSS_EXPR_PATH, "rb") as f:
            complete_error_expr = pickle.load(f)
        symbolic_time = None
    else:
        teacher_expr = network_to_expr(
            teacher, input_symbols, teacher_weight_symbols, teacher_bias_symbols
        )
        student_expr = network_to_expr(
            student, input_symbols, student_weight_symbols, student_bias_symbols
        )
        assert len(teacher_expr) == len(student_expr)
        error_expr = expand(sum((t - s) ** 2 for t, s in zip(teacher_expr, student_expr)))
        complete_error_expr = gaussian_integral(error_expr, INPUT_SIZE)

        # Cache expression.
        with open(LOSS_EXPR_PATH, "wb") as f:
            pickle.dump(complete_error_expr, f)

    # Training loop.
    for update in range(NUM_UPDATES):

        # Evaluate the complete student loss.
        loss = evaluate_error(complete_error_expr, networks)

        # Backwards pass to update student network.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Update {update} loss: {loss.item()}", end="\r")

    print("")


def main(complete=False) -> None:
    """
    Main function for ckd_sandbox.py.

    Parameters
    ----------
    complete : bool
        Whether to use Complete KD. Otherwise, student network is trained with
        supervised learning.
    """

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

    # Construct optimizer.
    optimizer = torch.optim.SGD(student.parameters(), lr=LR, momentum=MOMENTUM)

    # Call training function.
    if complete:
        run_complete_kd(teacher, student, optimizer)
    else:
        run_kd(teacher, student, optimizer)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--complete",
        default=False,
        action="store_true",
        help="Use Complete Knowledge Distillation instead of distillation with supervised learning."
    )
    args = parser.parse_args()

    main(complete=args.complete)
