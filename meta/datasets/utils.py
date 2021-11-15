""" Utilities for meta/datasets/. """

import os
from PIL import Image
from typing import Callable, Any

import numpy as np
import torch
import torchvision.transforms as transforms


GRAY_MEAN = [0.5]
GRAY_STD = [0.5]
RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]
GRAY_TRANSFORM = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=GRAY_MEAN, std=GRAY_STD)]
)
RGB_TRANSFORM = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=RGB_MEAN, std=RGB_STD)]
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


def visualize_img_tensor(imgs: torch.Tensor, path: str) -> None:
    """
    Save an image (or a batch of images) encoded as a normalized tensor into an image
    file(s). Note that the channel dimension of `imgs` should be 1 if `imgs` is a batch,
    or 0 othherwise.
    """

    # Enforce correct shape for `imgs`.
    if len(imgs.shape) == 3:
        imgs = imgs.unsqueeze(0)
    assert len(imgs.shape) == 4

    # Un-normalize `imgs` and store as numpy array.
    mean = torch.Tensor(RGB_MEAN, device=imgs.device).view(1, 3, 1, 1)
    std = torch.Tensor(RGB_STD, device=imgs.device).view(1, 3, 1, 1)
    imgs = (imgs * std) + mean
    imgs *= 255.0
    imgs_arr = imgs.detach().cpu().numpy()
    imgs_arr = imgs_arr.astype(np.uint8)
    imgs_arr = np.transpose(imgs_arr, (0, 2, 3, 1))

    # Save out images.
    dot = path.rfind(".")
    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    for i in range(imgs.shape[0]):
        img_arr = imgs_arr[i]
        img = Image.fromarray(img_arr)
        path_i = path[:dot] + f"_{i}" + path[dot:]
        img.save(path_i)
