"""
Dataset object for MNIST dataset. This is a wrapper around the PyTorch MNIST Dataset
object, which additionally contains information about metrics to compute, data
augmentation, loss function, etc.
"""

import torch.nn as nn
from torchvision.datasets import MNIST as torch_MNIST

from meta.train.loss import get_accuracy
from meta.datasets.utils import GRAY_TRANSFORM


class MNIST(torch_MNIST):
    """ MNIST dataset wrapper. """

    input_size = (1, 28, 28)
    output_size = 10
    loss_cls = nn.CrossEntropyLoss
    loss_kwargs = {}
    criterion_kwargs = {"train": {}, "eval": {}}
    extra_metrics = {
        "train_accuracy": {
            "fn": get_accuracy,
            "basename": "accuracy",
            "window": 50,
            "maximize": True,
            "train": True,
            "show": True,
        },
        "eval_accuracy": {
            "fn": get_accuracy,
            "basename": "accuracy",
            "window": 1,
            "maximize": True,
            "train": False,
            "show": True,
        },
    }
    dataset_kwargs = {
        "train": {"download": True, "transform": GRAY_TRANSFORM},
        "eval": {"download": True, "transform": GRAY_TRANSFORM},
    }

    def __init__(self, root: str, train: bool = True):
        """ Init function for MNIST. """

        split = "train" if train else "eval"
        kwargs = MNIST.dataset_kwargs[split]
        super(MNIST, self).__init__(root=root, train=train, **kwargs)
