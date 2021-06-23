""" Unit tests for meta/networks/backbone.py. """

import torch

from meta.networks.backbone import BackboneNetwork


SETTINGS = {
    "input_size": (3, 32, 32),
    "output_size": [(10, 32, 32), (5, 32, 32), (1, 32, 32)],
    "arch_type": "conv",
    "num_backbone_layers": 4,
    "head_channels": 8,
    "initial_channels": 8,
    "device": torch.device("cpu"),
    "batch_size": 4,
}


def test_backward() -> None:
    """
    Test backward(). We just want to make sure that (in the multi-task case) the
    gradient with respect to the i-th task loss is zero for all parameters in output
    head j != i, and is nonzero for all parameters in output head i.
    """

    # Construct network.
    network = BackboneNetwork(
        input_size=SETTINGS["input_size"],
        output_size=SETTINGS["output_size"],
        arch_type=SETTINGS["arch_type"],
        num_backbone_layers=SETTINGS["num_backbone_layers"],
        head_channels=SETTINGS["head_channels"],
        initial_channels=SETTINGS["initial_channels"],
        device=SETTINGS["device"],
    )

    # Construct batch of inputs.
    inputs = 2 * torch.rand((SETTINGS["batch_size"], *SETTINGS["input_size"])) - 1

    # Get output of network.
    output = network(inputs)

    # Compute losses (we just compute the squared network output to keep it simple) and
    # test gradients.
    num_tasks = len(SETTINGS["output_size"])
    for i in range(num_tasks):

        # Zero out gradients.
        network.zero_grad()

        # Compute loss over outputs from the current task.
        task_start = sum([size[0] for size in SETTINGS["output_size"][:i]])
        task_end = sum([size[0] for size in SETTINGS["output_size"][: i + 1]])
        print(f"task_start: {task_start}")
        print(f"task_end: {task_end}")
        loss = torch.sum(output[:, task_start:task_end] ** 2)

        # Test gradients.
        loss.backward(retain_graph=True)
        check_gradients(network.backbone, nonzero=True)
        for j in range(num_tasks):
            nonzero = j == i
            check_gradients(network.head.p_modules[j], nonzero=nonzero)


def check_gradients(m: torch.nn.Module, nonzero: bool) -> None:
    """ Helper function to test whether gradients are nonzero. """

    for param in m.parameters():
        if nonzero:
            assert (param.grad != 0).any()
        else:
            assert param.grad is None or (param.grad == 0).all()
