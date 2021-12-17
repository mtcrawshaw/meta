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
INPUT_SIZE = 1
OUTPUT_SIZE = 1
NUM_TEACHER_LAYERS = 2
TEACHER_HIDDEN_SIZE = 2
NUM_STUDENT_LAYERS = 2
STUDENT_HIDDEN_SIZE = 1
ACTIVATION = "relu"

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
CUDA = True
DEVICE = torch.device("cuda:0" if CUDA else "cpu")

# Cached polynomial approximations.
RELU_POLY = {
    2: [0.375, 0.5, 0.1171875],
    3: [0.375, 0.5, 0.1171875, 0.0],
    4: [0.234375, 0.5, 0.205078, 0.0, -0.00640869],
    5: [0.234375, 0.5, 0.205078, 0.0, -0.00640869, 0.0],
    6: [0.170898, 0.5, 0.28839111, 0.0, -0.0220298767, 0.0, 0.000715970993],
}


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

            # Compute activation for each output unit, approximated with a power-series.
            for i in range(len(outputs)):
                if isinstance(activation, ExpActivation):
                    outputs[i] = sum(
                        outputs[i] ** m / factorial(m) for m in range(MAX_PS_DEGREE + 1)
                    )
                elif isinstance(activation, nn.ReLU):
                    outputs[i] = sum(
                        RELU_POLY[MAX_PS_DEGREE][m] * outputs[i] ** m
                        for m in range(MAX_PS_DEGREE + 1)
                    )
                else:
                    raise NotImplementedError

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


def symbol_to_position(symbol: Symbol) -> Tuple[int, int, int, str]:
    """
    Given a symbol which corresponds to a single parameter in a network, returns the
    indices of the layer, input unit, output unit, and network name corresponding for
    the corresponding parameter. When the parameter is a bias (so that there is no
    corresponding input unit), the input unit index is given as None.
    """

    if symbol.name.startswith("w"):
        first_pos = symbol.name.find("_")
        second_pos = symbol.name.find("_", first_pos + 1)
        last_pos = symbol.name.find("_", second_pos + 1)
        layer = int(symbol.name[1:first_pos])
        out_unit = int(symbol.name[first_pos + 1 : second_pos])
        in_unit = int(symbol.name[second_pos + 1 : last_pos])
        name = symbol.name[last_pos + 1 :]

    elif symbol.name.startswith("b"):
        first_pos = symbol.name.find("_")
        second_pos = symbol.name.find("_", first_pos + 1)
        layer = int(symbol.name[1:first_pos])
        out_unit = int(symbol.name[first_pos + 1 : second_pos])
        in_unit = None
        name = symbol.name[second_pos + 1 :]

    else:
        raise ValueError("Can only handle symbols with names starting with 'w' or 'b'.")

    return layer, in_unit, out_unit, name


def get_parameter_index(
    layer: int, in_unit: int, out_unit: int, network: nn.Module
) -> int:
    """
    Returns the index of a particular parameter in the flattened parameter vector of a
    network. This function assumes that the network is an instance of nn.Sequential
    whose elements are instances of nn.Sequential consisting of a Linear layer and a
    parameter-free activation function.
    """

    idx = 0
    for l in range(layer):
        idx += (network[l][0].in_features + 1) * network[l][0].out_features

    in_features = network[layer][0].in_features
    out_features = network[layer][0].out_features
    if in_unit is not None:
        # Weight.
        idx += in_features * out_unit
        idx += in_unit
    else:
        # Bias.
        idx += in_features * out_features
        idx += out_unit

    return idx


def get_variable_index(symbol: Symbol, networks: Dict[str, nn.Module]) -> int:
    """
    Returns the index of a symbol (which corresponds to a network weight) in the vector
    made from the flattened vector containing parameters of the teacher and student
    network. This function expects that the parameter vector contains the parameters of
    first the student network then the teacher network.
    """

    layer, in_unit, out_unit, name = symbol_to_position(symbol)

    network_idx = get_parameter_index(layer, in_unit, out_unit, networks[name])
    if name == "student":
        var_idx = network_idx
    elif name == "teacher":
        student_params = torch.cat(
            [p.view(-1) for p in networks["student"].parameters()]
        )
        var_idx = network_idx + len(student_params)
    else:
        raise NotImplementedError

    return var_idx


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
                        current_integral *= double_factorial(2 * j - 1)
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


def evaluate_error(
    powers: torch.Tensor, coeffs: torch.Tensor, networks: Dict[str, nn.Module]
) -> torch.Tensor:
    """
    Evaluate the complete KD loss using parallel matrix operations. `powers` is a matrix
    with shape `(num_monomials, num_vars)` (stored in sparse COO format) and `coeffs` is
    a vector with length `num_monomials`.
    """

    # Construct flattened parameter vector containing parameters from student and
    # teacher networks.
    student_params = torch.cat([p.view(-1) for p in networks["student"].parameters()])
    teacher_params = torch.cat([p.view(-1) for p in networks["teacher"].parameters()])
    params = torch.cat([student_params, teacher_params])

    # Compute complete KD loss. Note: This next line will have to change in order to
    # preserve sparsity of the result `terms`. Currently, even if `powers` is sparse,
    # `terms` will be mostly filled with 1s and therefore not sparse. When the code is
    # updated so that `powers` is sparse, we will have to change this line to preserve
    # sparsity, which we can probably do with some hacky matrix operations. As it
    # stands, the memory cost will be much too large.
    terms = params.unsqueeze(0) ** powers
    monomials = terms.prod(dim=1)
    weighted_monomials = monomials * coeffs
    loss = torch.sum(monomials)

    return loss


def monomial_to_matrices(
    monomial: Expr, networks: Dict[str, nn.Module]
) -> Tuple[float, List[int], List[int]]:
    """
    Collects and returns the coefficient, variable indices, and powers for a monomial
    expression.
    """

    coeff = 1.0
    var_idxs = []
    powers = []

    # Parse monomial based on its type to collect powers and coefficient.
    if issubclass(monomial.func, Number):
        coeff = torch.Tensor([float(monomial)])

    elif monomial.func == Symbol:
        var_idx = get_variable_index(monomial, networks)
        var_idxs.append(var_idx)
        powers.append(1.0)

    elif monomial.func == Pow:
        base, exp = monomial.args
        if issubclass(base.func, Number):
            coeff = float(base) ** float(exp)
        else:
            var_idx = get_variable_index(base, networks)
            var_idxs.append(var_idx)
            powers.append(float(exp))

    elif monomial.func == Mul:

        for subexpr in monomial.args:
            sub_coeff, sub_var_idxs, sub_powers = monomial_to_matrices(
                subexpr, networks
            )
            coeff *= sub_coeff
            var_idxs += sub_var_idxs
            powers += sub_powers

    else:
        raise NotImplementedError

    return coeff, var_idxs, powers


def expr_to_matrices(
    error_expr: Expr, networks: Dict[str, nn.Module], num_vars: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert the SymPy polynomial expression into matrix form. The polynomial is
    represented with one sparse matrix of size (num_monomials, num_vars) and one dense
    vector of size (num_monomials), both stored as Tensors. The matrix value at row i,
    column j holds the power of variable j in monomial i. If variable j doesn't appear
    in monomial i, the corresponding matrix value is natually 0. The vector value at
    position i holds the coefficient of monomial i. This distributed representation
    allows for a parallel evaluation of the huge polynomial with torch.
    """

    monomial_idxs = []
    var_idxs = []
    powers = []
    coeffs = []

    num_monomials = len(error_expr.args)

    # Loop over each monomial and collect values of powers and coefficients.
    assert error_expr.func == Add
    for m_idx, monomial in enumerate(error_expr.args):

        assert is_monomial(monomial)
        coeff, m_var_idxs, m_powers = monomial_to_matrices(monomial, networks)

        # Aggregate coefficients and powers from monomial into running list.
        for var_idx, power in zip(m_var_idxs, m_powers):
            monomial_idxs.append(m_idx)
            var_idxs.append(var_idx)
            powers.append(power)
        coeffs.append(coeff)

    # Construct matrices from power/coefficient values.
    powers = torch.sparse_coo_tensor(
        [monomial_idxs, var_idxs], powers, size=(num_monomials, num_vars)
    )
    coeffs = torch.Tensor(coeffs)
    powers = powers.to(DEVICE)
    coeffs = coeffs.to(DEVICE)

    return powers, coeffs


def get_network(
    input_size: int,
    output_size: int,
    hidden_size: int,
    num_layers: int,
    activation: str,
) -> nn.Module:
    """ Construct an MLP with the given parameters. """

    layers = []
    for i in range(num_layers):
        layer = []
        layer_in = input_size if i == 0 else hidden_size
        layer_out = output_size if i == num_layers - 1 else hidden_size
        layer.append(nn.Linear(layer_in, layer_out))
        if i < num_layers - 1:
            if activation == "exp":
                layer.append(ExpActivation())
            elif activation == "relu":
                layer.append(nn.ReLU())
            else:
                raise NotImplementedError
        layers.append(nn.Sequential(*layer))
    return nn.Sequential(*layers)


def get_network_symbols(
    input_size: int, output_size: int, hidden_size: int, num_layers: int, name: str,
) -> Tuple[List[List[List[Symbol]]], List[List[Symbol]], int]:
    """
    Construct SymPy symbols for network weights and biases. Returns nested lists of
    SymPy symbols representing weights and biases, as well as the total number of
    symbols in the network.
    """

    weight_symbols = []
    bias_symbols = []
    num_symbols = 0
    for i in range(num_layers):
        layer_in = input_size if i == 0 else hidden_size
        layer_out = output_size if i == num_layers - 1 else hidden_size
        weight_symbols.append([])
        bias_symbols.append([])
        num_symbols += (layer_in + 1) * layer_out
        for j in range(layer_out):
            weight_symbols[i].append([])
            for k in range(layer_in):
                weight_symbols[i][j].append(Symbol(f"w{i}_{j}_{k}_{name}"))
            bias_symbols[i].append(Symbol(f"b{i}_{j}_{name}"))

    return weight_symbols, bias_symbols, num_symbols


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
        torch.zeros(train_size, INPUT_SIZE), torch.ones(train_size, INPUT_SIZE),
    )
    test_data = torch.normal(
        torch.zeros(test_size, INPUT_SIZE), torch.ones(test_size, INPUT_SIZE),
    )
    train_data = train_data.to(DEVICE)
    test_data = test_data.to(DEVICE)

    # Construct data loaders.
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False,
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


def run_complete_kd(
    teacher: nn.Module,
    student: nn.Module,
    optimizer: torch.optim.Optimizer,
    use_cache: bool = False,
):
    """ Runs complete knowledge distillation on a student and teacher network. """

    # Construct SymPy symbols for network inputs.
    input_symbols = [Symbol(f"x{i}") for i in range(INPUT_SIZE)]

    # Construct SymPy symbols for network inputs, weights, and biases.
    teacher_weight_symbols, teacher_bias_symbols, teacher_symbols = get_network_symbols(
        INPUT_SIZE, OUTPUT_SIZE, TEACHER_HIDDEN_SIZE, NUM_TEACHER_LAYERS, name="teacher"
    )
    student_weight_symbols, student_bias_symbols, student_symbols = get_network_symbols(
        INPUT_SIZE, OUTPUT_SIZE, STUDENT_HIDDEN_SIZE, NUM_STUDENT_LAYERS, name="student"
    )
    num_symbols = teacher_symbols + student_symbols

    # Symbolically compute the student error over the entire input distribution as a
    # function of network parameters.
    networks = {"teacher": teacher, "student": student}
    if use_cache:
        raise ValueError("Cached expression doesn't exist, but `use_cache` is True.")
        with open(LOSS_EXPR_PATH, "rb") as f:
            complete_error_expr = pickle.load(f)
    else:
        teacher_expr = network_to_expr(
            teacher, input_symbols, teacher_weight_symbols, teacher_bias_symbols
        )
        student_expr = network_to_expr(
            student, input_symbols, student_weight_symbols, student_bias_symbols
        )
        assert len(teacher_expr) == len(student_expr)
        error_expr = expand(
            sum((t - s) ** 2 for t, s in zip(teacher_expr, student_expr))
        )
        complete_error_expr = gaussian_integral(error_expr, INPUT_SIZE)

        # Cache expression.
        with open(LOSS_EXPR_PATH, "wb") as f:
            pickle.dump(complete_error_expr, f)

    # Convert the SymPy polynomial into matrix form for fast evaluation during training.
    powers, coeffs = expr_to_matrices(complete_error_expr, networks, num_symbols)

    # TEMP: For now, we have to convert powers to a dense matrix in order to use it as
    # an operand in an exponentiation since PyTorch doesn't support exponentiation with
    # sparse Tensors. This is not scalable to networks of any reasonable size.
    powers = powers.to_dense()

    # Training loop.
    for update in range(NUM_UPDATES):

        # Evaluate the complete student loss.
        loss = evaluate_error(powers, coeffs, networks)

        # Backwards pass to update student network.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Update {update} loss: {loss.item()}", end="\r")

    print("")


def main(complete: bool = False, use_cache: bool = False) -> None:
    """
    Main function for ckd_sandbox.py.

    Parameters
    ----------
    complete : bool
        Whether to use Complete KD. Otherwise, student network is trained with
        supervised learning.
    use_cache : bool
        Whether to load the cached loss expression at `LOSS_EXPR_PATH`. Use this option
        with caution: you might accidentally load an expression that was generated with
        hyperparameters which don't match the current values.
    """

    # Set random seed.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # Construct teacher and student network.
    teacher = get_network(
        INPUT_SIZE, OUTPUT_SIZE, TEACHER_HIDDEN_SIZE, NUM_TEACHER_LAYERS, ACTIVATION
    )
    student = get_network(
        INPUT_SIZE, OUTPUT_SIZE, STUDENT_HIDDEN_SIZE, NUM_STUDENT_LAYERS, ACTIVATION
    )
    teacher = teacher.to(DEVICE)
    student = student.to(DEVICE)

    # Construct optimizer.
    optimizer = torch.optim.SGD(student.parameters(), lr=LR, momentum=MOMENTUM)

    # Call training function.
    if complete:
        run_complete_kd(teacher, student, optimizer, use_cache=use_cache)
    else:
        if use_cache:
            raise ValueError("The `use_cache` option is only valid for complete KD.")
        run_kd(teacher, student, optimizer)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--complete",
        default=False,
        action="store_true",
        help="Use Complete Knowledge Distillation instead of distillation with supervised learning.",
    )
    parser.add_argument(
        "--use_cache",
        default=False,
        action="store_true",
        help="Use the cached expression for the complete KD loss.",
    )
    args = parser.parse_args()

    main(complete=args.complete, use_cache=args.use_cache)
