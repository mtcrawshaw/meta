"""
Script to optimize a polynomial of arbitrary degree to approximate a given activation
function. We should ideally just solve for this polynomial in closed-form, but this is a
temporary solution.
"""

import argparse
from typing import Callable

import torch
from torch import nn

DATA_DIST = "uniform"
UNIFORM_DATA_BOUND = 4
GAUSSIAN_DATA_STD = 1

NUM_UPDATES = 100000
BATCH_SIZE = 10000
LR = 3e-5
MOMENTUM = 0.9

CUDA = True
DEVICE = torch.device("cuda:0" if CUDA else "cpu")
SEED = 0


class Polynomial(nn.Module):
    """ Module parameterizing a polynomial function. """

    def __init__(self, degree: int) -> None:
        """ Init function for Polynomial. """
        super(Polynomial, self).__init__()
        self.degree = degree
        self.coeffs = nn.Parameter(2 * torch.rand(self.degree + 1) - 1)
        self.register_buffer("powers", torch.Tensor(list(range(self.degree + 1))))

    def __repr__(self) -> str:
        """ String representation of `self`. """
        return str(self.coeffs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward function for Polynomial. """
        return torch.sum(self.coeffs * x ** self.powers, dim=-1, keepdim=True)


def get_data_sampler(batch_size: int) -> Callable[[], torch.Tensor]:
    """ Return a function that samples a batch of data. """

    assert DATA_DIST in ["uniform", "gaussian"]

    if DATA_DIST == "uniform":
        def sampler() -> torch.Tensor:
            return UNIFORM_DATA_BOUND * (torch.rand(batch_size, 1, device=DEVICE) * 2 - 1)

    elif DATA_DIST == "gaussian": 
        mean = torch.zeros((BATCH_SIZE, 1), device=DEVICE)
        std = torch.ones((BATCH_SIZE, 1), device=DEVICE)
        def sampler() -> torch.Tensor:
            return torch.normal(mean, std) * GAUSSIAN_DATA_STD

    else:
        assert False

    return sampler

def main(degree: int):
    """ Main function for approximate_activations.py. """

    # Set random seed.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # Construct activation function and polynomial.
    activation = nn.ReLU()
    poly = Polynomial(degree)
    poly.to(DEVICE)

    # Construct optimizer.
    optimizer = torch.optim.SGD(poly.parameters(), lr=LR, momentum=MOMENTUM)

    # Construct data sampler.
    sampler = get_data_sampler(BATCH_SIZE)

    # Training loop.
    mean_loss = 0
    ema_alpha = 0.99
    for update in range(NUM_UPDATES):

        # Generate batch, get predictions, compute loss.
        batch_inputs = sampler()
        batch_outputs = activation(batch_inputs)
        batch_preds = poly(batch_inputs)
        loss = torch.mean((batch_preds - batch_outputs) ** 2)

        # Optimization step.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update progress.
        mean_loss = mean_loss * ema_alpha + loss.item() * (1 - ema_alpha)
        corrected_loss = mean_loss * 1.0 / (1 - ema_alpha ** (update + 1))
        print(f"Update {update} loss: {corrected_loss}")

    print(f"\nFinal polynomial: {poly}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--degree",
        type=int,
        default=4,
        help="Degree of polynomial approximation."
    )
    args = parser.parse_args()

    main(degree=args.degree)
