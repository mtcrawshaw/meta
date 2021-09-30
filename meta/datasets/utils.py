""" Utilities for meta/datasets/. """

from typing import Callable, Any

import torchvision.transforms as transforms


GRAY_TRANSFORM = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
)
RGB_TRANSFORM = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
)

get_split = lambda train: "train" if train else "eval"


def slice_second_dim(idx: int, to_long: bool = False) -> Callable[[Any], Any]:
    """ Utility function to generate slice functions for multi-task losses. """

    def slice(x: Any) -> Any:
        if to_long:
            return x[:, idx].long()
        else:
            return x[:, idx]

    return slice
