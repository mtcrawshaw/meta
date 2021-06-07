""" Loss functions and related utils. """

import math
from PIL import Image
from typing import List, Dict, Any, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from meta.utils.estimate import RunningStats


EPSILON = 1e-5


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
        device: torch.device = None,
    ) -> None:
        """ Init function for MultiTaskLoss. """

        super(MultiTaskLoss, self).__init__()

        # Set device.
        self.device = device if device is not None else torch.device("cpu")

        # Set task losses.
        self.task_losses = task_losses

        # Set task weighting strategy.
        loss_weighter = loss_weighter_kwargs["type"]
        del loss_weighter_kwargs["type"]
        if loss_weighter not in [
            "Constant",
            "DWA",
            "MLDW",
            "LBTW",
            "NLW",
            "RWLW",
            "CLW",
            "NCLW",
        ]:
            raise NotImplementedError
        loss_weighter_cls = eval(loss_weighter)
        loss_weighter_kwargs["device"] = self.device
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

    def __init__(self, loss_weights: List[float], device: torch.device = None) -> None:
        """ Init function for LossWeighter. """

        self.device = device if device is not None else torch.device("cpu")

        # Set state.
        self.num_tasks = len(loss_weights)
        self.loss_weights = torch.Tensor(loss_weights)
        self.loss_weights = self.loss_weights.to(self.device)
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

    def __init__(self, ema_alpha: float, temp: float, **kwargs: Dict[str, Any]) -> None:
        """
        Init function for DWA. `temp` is the temperature used to smooth the weight
        distribution. The default value used in the paper is 2.
        """
        super(DWA, self).__init__(**kwargs)
        self.ema_alpha = ema_alpha
        self.temp = temp
        self.loss_avg = None

    def _update_weights(self) -> None:
        """ Compute new loss weights with DWA. """

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
        w = self.loss_avg / prev_avg
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
        self, ema_alpha: float, gamma: float, **kwargs: Dict[str, Any]
    ) -> None:
        """
        Init function for MLDW. `ema_alpha` is the EMA coefficient used to track a
        moving average of the losses at each step, `gamma` is the Focal Loss parameter
        which controls the focusing intensity.
        """
        super(MLDW, self).__init__(**kwargs)
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

    def __init__(self, alpha: float, period: int, **kwargs: Dict[str, Any]) -> None:
        """
        Init function for LBTW. `alpha` is a parameter that controls the focusing
        intensity: the larger the value of `alpha` the more weight will be given to
        tasks with slower learning. `period` controls how often the baseline losses are
        saved.
        """
        super(LBTW, self).__init__(**kwargs)
        self.alpha = alpha
        self.period = period
        self.steps = 0
        self.baseline_losses = None

    def _update_weights(self) -> None:
        """ Compute new loss weights with LBTW. """

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

    def __init__(self, sigma: float, **kwargs: Dict[str, Any]) -> None:
        """
        Init function for NLW. `sigma` is the standard deviation of the distribution
        from which the Gaussian noise is sampled.
        """
        super(NLW, self).__init__(**kwargs)
        self.sigma = sigma

    def _update_weights(self) -> None:
        """ Compute new loss weights with NLW. """

        # Add noise to weights.
        mean = torch.zeros(self.num_tasks)
        std = torch.ones(self.num_tasks) * self.sigma
        noise = torch.normal(mean, std).to(self.device)
        self.loss_weights = self.initial_loss_weights + noise

        # Normalize weights to ensure positivity.
        self.loss_weights = F.softmax(self.loss_weights, dim=0)
        self.loss_weights *= self.total_weight


class RWLW(LossWeighter):
    """
    Compute task loss weights with Random Walk Loss Weighting. This method simply adds a
    sample from a Gaussian distribution to each previous loss weight, so that the noise
    from each step compounds.
    """

    def __init__(self, sigma: float, **kwargs: Dict[str, Any]) -> None:
        """
        Init function for RWLW. `sigma` is the standard deviation of the distribution
        from which the Gaussian noise is sampled.
        """
        super(RWLW, self).__init__(**kwargs)
        self.sigma = sigma

    def _update_weights(self) -> None:
        """ Compute new loss weights with RWLW. """

        # Add noise to weights.
        mean = torch.zeros(self.num_tasks)
        std = torch.ones(self.num_tasks) * self.sigma
        noise = torch.normal(mean, std).to(self.device)
        self.loss_weights = self.loss_weights + noise

        # Normalize weights to ensure positivity.
        self.loss_weights = F.softmax(self.loss_weights, dim=0)
        self.loss_weights *= self.total_weight


class CLW(LossWeighter):
    """
    Compute task loss weights with Centered Loss Weighting. Here we keep a running std
    of each task's loss, and set each task's loss weight equal to the inverse of the std
    of the task loss.
    """

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        """
        Init function for CLW. `sigma` is the standard deviation of the distribution
        from which the Gaussian noise is sampled.
        """
        super(CLW, self).__init__(**kwargs)

        self.loss_stats = RunningStats(
            compute_stdev=True,
            shape=(self.num_tasks,),
            ema_alpha=0.99,
            device=self.device,
        )

    def _update_weights(self) -> None:
        """ Compute new loss weights with NLW. """

        # Update stats.
        self.loss_stats.update(self.loss_history[-1])

        # Set loss weights equal to inverse of loss stdev, then normalize the weights so
        # they sum to the initial total weight. Note that we don't update the weights
        # until after the first step, since at that point each stdev is 0.
        if len(self.loss_history) > 1:
            threshold_stdev = torch.max(
                self.loss_stats.stdev, EPSILON * torch.ones_like(self.loss_stats.stdev)
            )
            self.loss_weights = 1.0 / threshold_stdev
            self.loss_weights /= torch.sum(self.loss_weights)
            self.loss_weights *= self.total_weight


class NCLW(LossWeighter):
    """
    Compute task loss weights with Noisy Centered Loss Weighting. Here we keep a running
    std of each task's loss, and set each task's loss weight equal to the inverse of the
    std of the task loss, plus a small amount of Gaussian noise as in NLW.
    """

    def __init__(self, sigma: float, **kwargs: Dict[str, Any]) -> None:
        """
        Init function for NCLW. `sigma` is the standard deviation of the distribution
        from which the Gaussian noise is sampled.
        """
        super(NCLW, self).__init__(**kwargs)

        self.sigma = sigma
        self.loss_stats = RunningStats(
            compute_stdev=True,
            shape=(self.num_tasks,),
            ema_alpha=0.99,
            device=self.device,
        )

    def _update_weights(self) -> None:
        """ Compute new loss weights with NLW. """

        # Update stats.
        self.loss_stats.update(self.loss_history[-1])

        # Set loss weights equal to inverse of loss stdev, then normalize the weights so
        # they sum to the initial total weight. Note that we don't update the weights
        # until after the first step, since at that point each stdev is 0.
        if len(self.loss_history) > 1:
            self.loss_weights = 1.0 / self.loss_stats.stdev
            self.loss_weights /= torch.sum(self.loss_weights)
            self.loss_weights *= self.total_weight

        # Add noise to loss weights.
        mean = torch.zeros(self.num_tasks)
        std = torch.ones(self.num_tasks) * self.sigma
        noise = torch.normal(mean, std).to(self.device)
        self.loss_weights = self.loss_weights + noise

        # Normalize weights to ensure positivity.
        self.loss_weights = F.softmax(self.loss_weights, dim=0)
        self.loss_weights *= self.total_weight


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


def get_MTRegression_normal_loss(
    num_tasks: int,
) -> Callable[[torch.Tensor, torch.Tensor], float]:
    """
    Constructs and returns a function which computes the MTRegression normalized
    multi-task loss from a set of labels and the corresponding predictions.
    """

    WEIGHTS = [1.0, 50.0, 30.0, 70.0, 20.0, 80.0, 10.0, 40.0, 60.0, 90.0]
    weights_t = torch.Tensor([WEIGHTS[:num_tasks]])

    def metric(outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Computes normalized multi-task loss for MTRegression task. Both `outputs` and
        `labels` should have shape `(batch_size, num_tasks, output_dim)`.
        """
        diffs = torch.sum((outputs - labels) ** 2, dim=2)
        weighted_diffs = torch.mean(diffs / (weights_t ** 2))
        return float(weighted_diffs)

    return metric
