""" Loss functions and utils for loss functions. """

from PIL import Image
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn


class CosineSimilarityLoss(nn.Module):
    """
    Returns negative mean of cosine similarity (scaled into [0, 1]) between two tensors
    computed along `dim`.
    """

    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        super(CosineSimilarityLoss, self).__init__()
        self.single_loss = nn.CosineSimilarity(dim=dim, eps=eps)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return (1 - torch.mean(self.single_loss(x1, x2))) / 2.0


class MultiTaskLoss(nn.Module):
    """ Computes the weighted sum of multiple loss functions. """

    def __init__(
        self,
        task_losses: List[Dict[str, Any]],
        loss_weighter_kwargs: Dict[str, Any] = {},
    ) -> None:
        """ Init function for MultiTaskLoss. """

        super(MultiTaskLoss, self).__init__()

        # Set task losses.
        self.task_losses = task_losses

        # Set task weighting strategy.
        loss_weighter = loss_weighter_kwargs["type"]
        del loss_weighter_kwargs["type"]
        if loss_weighter not in ["Constant", "DWA", "MLDW", "RLW"]:
            raise NotImplementedError
        loss_weighter_cls = eval(loss_weighter)
        self.loss_weighter = loss_weighter_cls(**loss_weighter_kwargs)

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """ Compute values of each task losses, then return the sum. """

        # Compute task losses.
        task_loss_vals = []
        for i, task_loss in enumerate(self.task_losses):
            task_output = task_loss["output_slice"](outputs)
            task_label = task_loss["label_slice"](labels)
            task_loss_val = task_loss["loss"](task_output, task_label)
            task_loss_vals.append(task_loss_val)

        # Compute total loss as weighted sum of task losses.
        task_loss_vals = torch.stack(task_loss_vals)
        total_loss = torch.sum(task_loss_vals * self.loss_weighter.loss_weights)

        # Update loss weighter with new task loss values.
        self.loss_weighter.update(task_loss_vals)

        return total_loss


class LossWeighter:
    """ Compute task loss weights for multi-task learning. """

    def __init__(self, loss_weights: List[float]) -> None:
        """ Init function for LossWeighter. """

        # Set state.
        self.loss_weights = torch.Tensor(loss_weights)
        self.initial_loss_weights = torch.clone(self.loss_weights)
        self.loss_history = []

    def update(self, loss_vals: torch.Tensor) -> None:
        """ Compute new loss weights using most recent values of task losses. """
        self.loss_history.append(loss_vals)
        self.loss_history = self.loss_history[-2:]
        self._update_weights()

    def _update_weights(self) -> None:
        """ Update loss weights. Should be implemented in subclasses. """
        raise NotImplementedError


class Constant(LossWeighter):
    """ Compute task loss weights with the constant method. """

    def _update_weights(self) -> None:
        """
        Compute new loss weights with the constant method. Since the weights are
        constant, we don't need to do anything.
        """
        pass


class DWA(LossWeighter):
    """
    Compute task loss weights with Dynamic Weight Averaging, detailed in
    https://arxiv.org/abs/1803.10704.
    """

    def __init__(self, loss_weights: torch.Tensor, temp: float) -> None:
        """ Init function for DWA. """
        super(DWA, self).__init__(loss_weights)
        self.temp = temp

    def _update_weights(self) -> None:
        """ Compute new loss weights with DWA. """
        raise NotImplementedError


class MLDW(LossWeighter):
    """
    Compute task loss weights with Multi-Loss Dynamic Training, detailed in
    https://arxiv.org/abs/1810.12193.
    """

    def __init__(
        self, loss_weights: torch.Tensor, ema_alpha: float, gamma: float
    ) -> None:
        """ Init function for MLDW. """
        super(MLDW, self).__init__(loss_weights)
        self.ema_alpha = ema_alpha
        self.gamma = gamma

    def _update_weights(self) -> None:
        """ Compute new loss weights with MLDW. """
        raise NotImplementedError


class NLW(LossWeighter):
    """
    Compute task loss weights with Noisy Loss Weighting. This method simply adds a
    sample from a Gaussian distribution to each initial loss weight. This means that the
    loss weights will have less variance than those from RWLW, where the Gaussian noise
    compounds.
    """

    def __init__(self, loss_weights: torch.Tensor, sigma: float) -> None:
        """
        Init function for NLW. `sigma` is the standard deviation of the distribution
        from which the Gaussian noise is sampled.
        """
        super(NLW, self).__init__(loss_weights)
        self.sigma = sigma

    def _update_weights(self) -> None:
        """ Compute new loss weights with NLW. """
        raise NotImplementedError


def RWLW(LossWeighter):
    """
    Compute task loss weights with Random Walk Loss Weighting. This method simply adds a
    sample from a Gaussian distribution to each previous loss weight, so that the noise
    from each step compounds.
    """

    def __init__(self, loss_weights: torch.Tensor, sigma: float) -> None:
        """
        Init function for RWLW. `sigma` is the standard deviation of the distribution
        from which the Gaussian noise is sampled.
        """
        super(RWLW, self).__init__(loss_weights)
        self.sigma = sigma

    def _update_weights(self) -> None:
        """ Compute new loss weights with RWLW. """
        raise NotImplementedError


def save_batch(
    task_losses: Dict[str, Any], outputs: torch.Tensor, labels: torch.Tensor
) -> None:
    """
    Debug function to save out batch of NYUv2 labels as images and exit. Should be
    called from `forward()` in `MultiTaskLoss`. Note that this should only be used when
    doing multi-task training on the NYUv2 dataset.
    """

    loss_names = ["seg", "sn", "depth"]
    batch_size = outputs.shape[0]
    colors = [
        (0, 0, 0),
        (0, 0, 127),
        (0, 0, 255),
        (0, 127, 0),
        (0, 127, 127),
        (0, 127, 255),
        (0, 255, 0),
        (0, 255, 127),
        (0, 255, 255),
        (127, 0, 0),
        (127, 0, 127),
        (127, 0, 255),
        (127, 127, 0),
        (127, 127, 127),
    ]

    for i, task_loss in enumerate(task_losses):
        task_label = task_loss["label_slice"](labels)

        name = loss_names[i]
        for j in range(batch_size):
            label = task_label[j]
            if name == "seg":
                label_arr = np.zeros((label.shape[0], label.shape[1], 3))
                for x in range(label_arr.shape[0]):
                    for y in range(label_arr.shape[1]):
                        label_arr[x, y] = colors[label[x, y]]
                label_arr = np.uint8(label_arr)
            elif name == "sn":
                label_arr = np.transpose(label.numpy(), (1, 2, 0))
                label_arr = np.uint8(label_arr * 255.0)
            elif name == "depth":
                label_arr = np.transpose(label.numpy(), (1, 2, 0))
                min_depth, max_depth = np.min(label_arr), np.max(label_arr)
                label_arr = (label_arr - min_depth) / (max_depth - min_depth)
                label_arr = np.concatenate([label_arr] * 3, axis=2)
                label_arr = np.uint8(label_arr * 255.0)
            else:
                raise NotImplementedError

            img = Image.fromarray(label_arr)
            img.save("test_%d_%s.png" % (j, name))

    exit()
