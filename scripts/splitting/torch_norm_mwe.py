"""
Minimum working example of torch.norm's dependency on the order of elements.
"""

import torch


NORM_TYPE = 2.0


def main():
    """ Main function. """

    # Create list and a rearranged version.
    values = [
        0.00656224275007843971,
        0.01008806377649307251,
        0.00686217984184622765,
        0.00748096127063035965,
        0.35544574260711669922,
        0.29662144184112548828,
        2.23350238800048828125,
        4.37898063659667968750,
        2.29397535324096679688,
        3.23525500297546386719,
        2.17198586463928222656,
        2.39126753807067871094,
    ]
    new_order = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 4, 5]
    rearrange_list = lambda l, order: [l[i] for i in order]
    rearranged_values = rearrange_list(values, new_order)
    assert set(values) == set(rearranged_values)

    # Convert to tensors.
    tensor = torch.Tensor(values)
    rearranged_tensor = torch.Tensor(rearranged_values)

    # Compute norms and compare.
    torch.set_printoptions(precision=20)
    norm = torch.norm(tensor, NORM_TYPE)
    rearranged_norm = torch.norm(rearranged_tensor, NORM_TYPE)
    print("norm: %s" % norm)
    print("rearranged_norm: %s" % rearranged_norm)
    assert norm == rearranged_norm


if __name__ == "__main__":
    main()
