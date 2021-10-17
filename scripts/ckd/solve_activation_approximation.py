"""
Script to solve for a polynomial of arbitrary degree that approximates a given
activation function. Currently, this script only approximates the ReLU activation
function assuming that the distribution of inputs is uniform over the interval [-BOUND,
BOUND].
"""

import argparse

import numpy as np
import torch
from torch import nn


BOUND = 4


class Polynomial(nn.Module):
    """ Module parameterizing a polynomial function. """

    def __init__(self, degree: int, coeffs: torch.Tensor = None) -> None:
        """ Init function for Polynomial. """
        super(Polynomial, self).__init__()
        self.degree = degree
        if coeffs is None:
            self.coeffs = nn.Parameter(2 * torch.rand(self.degree + 1) - 1)
        else:
            assert coeffs.shape == (self.degree + 1,)
            self.coeffs = nn.Parameter(coeffs)
        self.register_buffer("powers", torch.Tensor(list(range(self.degree + 1))))

    def __repr__(self) -> str:
        """ String representation of `self`. """
        return str(self.coeffs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward function for Polynomial. """
        return torch.sum(self.coeffs * x ** self.powers, dim=-1, keepdim=True)


def main(degree: int):
    """ Main function for solve_activation_approximation.py. """

    # Construct coefficient matrix to solve linear system of equations for error
    # minimization.
    n = degree + 1
    A = np.zeros((n, n))
    y = np.zeros((n,))
    for i in range(n):
        for j in range(n):
            if (i + j) % 2 == 0:
                A[i, j] = 4 * BOUND ** (i + j + 1) / (i + j + 1)
        y[i] = 2 * BOUND ** (i + 2) / (i + 2)

    # Solve for coefficients of polynomial approximation.
    coeffs = np.matmul(np.linalg.inv(A), y)

    # Output polynomial and error.
    error = BOUND ** 3 / 3
    for i in range(n):
        for j in range(n):
            if (i + j) % 2 == 0:
                error += 2 * BOUND ** (i + j + 1) / (i + j + 1) * coeffs[i] * coeffs[j]
        error -= 2 * BOUND ** (i + 2) / (i + 2) * coeffs[i]
    error /= 2 * BOUND

    print(f"Average squared error: {error}")
    print(f"Final polynomial: {coeffs}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--degree",
        default=4,
        type=int,
        help="Degree of polynomial approximation."
    )
    args = parser.parse_args()

    main(degree=args.degree)
