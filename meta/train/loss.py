""" Loss functions and related utils. """

import math
from PIL import Image
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        if loss_weighter not in ["Constant", "DWA", "MLDW", "LBTW", "NLW", "RWLW"]:
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
        self.num_tasks = len(loss_weights)
        self.loss_weights = torch.Tensor(loss_weights)
        self.initial_loss_weights = torch.clone(self.loss_weights)
        self.total_weight = torch.sum(self.loss_weights)
        self.loss_history = []
        self.MAX_HISTORY_LEN = 2

    def update(self, loss_vals: torch.Tensor) -> None:
        """ Compute new loss weights using most recent values of task losses. """
        self.loss_history.append(loss_vals.detach())
        self.loss_history = self.loss_history[-self.MAX_HISTORY_LEN :]
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

    def __init__(self, loss_weights: List[float], temp: float) -> None:
        """
        Init function for DWA. `temp` is the temperature used to smooth the weight
        distribution. The default value used in the paper is 2.
        """
        super(DWA, self).__init__(loss_weights)
        self.temp = temp

    def _update_weights(self) -> None:
        """ Compute new loss weights with DWA. """

        # Check that we have already performed a sufficient number of updates.
        if len(self.loss_history) < self.MAX_HISTORY_LEN:
            return
        assert len(self.loss_history) == self.MAX_HISTORY_LEN

        # Update weights.
        w = self.loss_history[-1] / self.loss_history[-2]
        w /= self.temp
        w = F.softmax(w, dim=0)
        w *= self.total_weight
        self.loss_weights = w


class MLDW(LossWeighter):
    """
    Compute task loss weights with Multi-Loss Dynamic Training, detailed in
    https://arxiv.org/abs/1810.12193.
    """

    def __init__(
        self, loss_weights: List[float], ema_alpha: float, gamma: float
    ) -> None:
        """
        Init function for MLDW. `ema_alpha` is the EMA coefficient used to track a
        moving average of the losses at each step, `gamma` is the Focal Loss parameter
        which controls the focusing intensity.
        """
        super(MLDW, self).__init__(loss_weights)
        self.ema_alpha = ema_alpha
        self.gamma = gamma
        self.loss_avg = None

    def _update_weights(self) -> None:
        """ Compute new loss weights with MLDW. """

        # Update exponential moving average of loss.
        if self.loss_avg is None:
            self.loss_avg = self.loss_history[-1]
            prev_avg = torch.clone(self.loss_avg)
        else:
            prev_avg = torch.clone(self.loss_avg)
            self.loss_avg = (
                self.ema_alpha * self.loss_avg
                + (1 - self.ema_alpha) * self.loss_history[-1]
            )

        # Update weights.
        p = torch.min(prev_avg, self.loss_avg) / prev_avg
        self.loss_weights = -torch.pow(1 - p, self.gamma) * torch.log(p)
        if torch.all(self.loss_weights == 0):
            self.loss_weights = torch.clone(self.initial_loss_weights)


class LBTW(LossWeighter):
    """
    Compute task loss weights with Loss Balanced Task Weighting, detailed in
    https://ojs.aaai.org//index.php/AAAI/article/view/5125.
    """

    def __init__(self, loss_weights: List[float], alpha: float, period: int) -> None:
        """
        Init function for LBTW. `alpha` is a parameter that controls the focusing
        intensity: the larger the value of `alpha` the more weight will be given to
        tasks with slower learning. `period` controls how often the baseline losses are
        saved.
        """
        super(LBTW, self).__init__(loss_weights)
        self.alpha = alpha
        self.period = period
        self.steps = 0
        self.baseline_losses = None

    def _update_weights(self) -> None:
        """ Compute new loss weights with MLDW. """

        # Update baseline losses.
        if self.baseline_losses is None or self.steps % self.period == 0:
            self.baseline_losses = self.loss_history[-1]

        # Update weights.
        self.loss_weights = torch.pow(
            self.loss_history[-1] / self.baseline_losses, self.alpha
        )

        # Update number of steps.
        self.steps += 1


class NLW(LossWeighter):
    """
    Compute task loss weights with Noisy Loss Weighting. This method simply adds a
    sample from a Gaussian distribution to each initial loss weight. This means that the
    loss weights will have less variance than those from RWLW, where the Gaussian noise
    compounds.
    """

    def __init__(self, loss_weights: List[float], sigma: float) -> None:
        """
        Init function for NLW. `sigma` is the standard deviation of the distribution
        from which the Gaussian noise is sampled.
        """
        super(NLW, self).__init__(loss_weights)
        self.sigma = sigma

    def _update_weights(self) -> None:
        """ Compute new loss weights with NLW. """

        mean = torch.zeros(self.num_tasks)
        std = torch.ones(self.num_tasks) * self.sigma
        self.loss_weights = self.initial_loss_weights + torch.normal(mean, std)


class RWLW(LossWeighter):
    """
    Compute task loss weights with Random Walk Loss Weighting. This method simply adds a
    sample from a Gaussian distribution to each previous loss weight, so that the noise
    from each step compounds.
    """

    def __init__(self, loss_weights: List[float], sigma: float) -> None:
        """
        Init function for RWLW. `sigma` is the standard deviation of the distribution
        from which the Gaussian noise is sampled.
        """
        super(RWLW, self).__init__(loss_weights)
        self.sigma = sigma

    def _update_weights(self) -> None:
        """ Compute new loss weights with RWLW. """

        mean = torch.zeros(self.num_tasks)
        std = torch.ones(self.num_tasks) * self.sigma
        self.loss_weights = self.loss_weights + torch.normal(mean, std)


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


def get_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute accuracy of classification prediction given outputs and labels.
    """

    accuracy = torch.sum(torch.argmax(outputs, dim=-1) == labels) / outputs.shape[0]
    return accuracy.item()


def NYUv2_seg_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute accuracy of semantic segmentation on the NYUv2 dataset. Here we assume that
    any pixels with label -1 are unlabeled, so we don't count these pixels in the
    accuracy computation. We also assume that the class dimension is directly after the
    batch dimension.
    """

    preds = torch.argmax(outputs, dim=1)
    correct = torch.sum(preds == labels)
    valid = torch.sum(labels != -1)
    accuracy = correct / valid
    return accuracy.item()


def NYUv2_sn_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute accuracy of surface normal estimation on the NYUv2 dataset. We define this
    as the number of pixels for which the angle between the true normal and the
    predicted normal is less than `DEGREE_THRESHOLD` degrees. Here we assume that the
    normal dimension is 1.
    """

    DEGREE_THRESHOLD = 10
    similarity_threshold = math.cos(DEGREE_THRESHOLD / 180 * math.pi)
    similarity = F.cosine_similarity(outputs, labels, dim=1)
    accuracy = torch.sum(similarity > similarity_threshold) / torch.numel(similarity)

    return accuracy.item()


def NYUv2_depth_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute accuracy of depth prediction on the NYUv2 dataset. We define this as the
    number of pixels for which the absolute value of the difference between the
    predicted depth and the true depth is less than `DEPTH_THRESHOLD`.
    """

    DEPTH_THRESHOLD = 0.25
    difference = torch.abs(outputs - labels)
    accuracy = torch.sum(difference < DEPTH_THRESHOLD) / torch.numel(difference)

    return accuracy.item()


def NYUv2_multi_seg_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute accuracy of semantic segmentation on the NYUv2 dataset when performing
    multi-task training. This function is essentially a wrapper around
    `NYUv2_seg_accuracy()`.
    """

    task_outputs = outputs[:, :13]
    task_labels = labels[:, 0].long()
    return NYUv2_seg_accuracy(task_outputs, task_labels)


def NYUv2_multi_sn_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute accuracy of surface normal estimation on the NYUv2 dataset when performing
    multi-task training. This function is essentially a wrapper around
    `NYUv2_sn_accuracy()`.
    """

    task_outputs = outputs[:, 13:16]
    task_labels = labels[:, 1:4]
    return NYUv2_sn_accuracy(task_outputs, task_labels)


def NYUv2_multi_depth_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute accuracy of depth prediction on the NYUv2 dataset when performing
    multi-task training. This function is essentially a wrapper around
    `NYUv2_depth_accuracy()`.
    """

    task_outputs = outputs[:, 16:17]
    task_labels = labels[:, 4:5]
    return NYUv2_depth_accuracy(task_outputs, task_labels)


def NYUv2_multi_avg_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute average accuracy of the three tasks on the NYUv2 dataset when performing
    multi-task training.
    """

    seg_accuracy = NYUv2_multi_seg_accuracy(outputs, labels)
    sn_accuracy = NYUv2_multi_sn_accuracy(outputs, labels)
    depth_accuracy = NYUv2_multi_depth_accuracy(outputs, labels)
    return np.mean([seg_accuracy, sn_accuracy, depth_accuracy])
