"""
Script to test properties of Tensor.state_dict().
"""

import torch

from meta.networks import MLPNetwork


INPUT_SIZE = 2
NUM_LAYERS = 1
HIDDEN_SIZE = None
OUTPUT_SIZE = 3
BATCH_SIZE = 4
STEPS = 6
LR = 3e-4
CUDA = True


def main() -> None:
    """ Main function. """

    # Set device.
    device = torch.device("cuda:0") if CUDA else torch.device("cpu")

    # Construct network and optimizer.
    net = MLPNetwork(
        INPUT_SIZE,
        OUTPUT_SIZE,
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN_SIZE,
        device=device
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    # Save initial state dict.
    initial_state_dict = net.state_dict()
    copied_state_dict = {key: torch.clone(val) for key, val in net.state_dict().items()}

    # Run training.
    for step in range(STEPS):
        net.zero_grad()
        data = torch.rand((BATCH_SIZE, INPUT_SIZE), device=device)
        out = net(data)
        loss = torch.sum(out ** 2)
        loss.backward()
        optimizer.step()

    # Save final state dict.
    final_state_dict = net.state_dict()

    # Compare final and initial state dict.
    assert list(initial_state_dict.keys()) == list(final_state_dict.keys())
    assert list(initial_state_dict.keys()) == list(copied_state_dict.keys())
    initial_equal = False
    copied_equal = False
    for key in initial_state_dict.keys():
        initial_val = initial_state_dict[key]
        copied_val = copied_state_dict[key]
        final_val = final_state_dict[key]
        initial_different = torch.any(torch.abs(initial_val - final_val) != 0)
        copied_different = torch.any(torch.abs(copied_val - final_val) != 0)

        if not initial_different:
            print("\ninitial value: %s" % initial_val)
            print("final value: %s\n" % final_val)
            initial_equal = True

        if not copied_different:
            print("\ncopied value: %s" % copied_val)
            print("final value: %s\n" % final_val)
            copied_equal = True

    assert not copied_equal and initial_equal

if __name__ == "__main__":
    main()

