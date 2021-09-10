""" Loss functions and related utils. """

import os
import random
import math
from PIL import Image
from typing import List, Dict, Any, Optional, Callable, Iterator, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F

from meta.networks import MultiTaskTrunkNetwork, BackboneNetwork
from meta.datasets.mt_regression import SCALES
from meta.utils.estimate import RunningStats


EPSILON = 1e-5


class CosineSimilarityLoss(nn.Module):
    """
    Returns negative mean of cosine similarity (scaled into [0, 1]) between two tensors
    computed along `dim`.
    """

    def __init__(
        self, dim: int = 1, eps: float = 1e-8, reduction: str = "mean"
    ) -> None:
        super(CosineSimilarityLoss, self).__init__()
        self.single_loss = nn.CosineSimilarity(dim=dim, eps=eps)
        self.reduction = reduction
        assert self.reduction in ["none", "mean", "sum"]

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:

        # Compute loss for each element in the batch.
        batch_loss = self.single_loss(x1, x2)
        batch_size = batch_loss.shape[0]
        batch_loss = batch_loss.view(batch_size, -1)
        batch_loss = (1 - torch.mean(batch_loss, dim=-1)) / 2.0

        # Reduce loss over the batch.
        if self.reduction == "none":
            loss = batch_loss
        elif self.reduction == "mean":
            loss = torch.mean(batch_loss)
        elif self.reduction == "sum":
            loss = torch.sum(batch_loss)
        else:
            raise NotImplementedError

        return loss


class ScaleInvariantDepthLoss(nn.Module):
    """
    Returns the scale invariant loss for depth prediction. This loss function is
    detailed in https://arxiv.org/abs/1406.2283. Here `alpha` is the coefficient
    `lambda` used in equation (4) of the above paper.
    """

    def __init__(self, alpha: float = 0.5, reduction: str = "mean") -> None:
        super(ScaleInvariantDepthLoss, self).__init__()
        self.alpha = alpha
        self.reduction = reduction
        assert self.reduction in ["none", "mean", "sum"]

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute scale invariant depth loss from predictions and ground truth. Note that
        we are comparing `outputs` with the log depth labels, so the network is being
        trained to predict the log depth.
        """

        # Compute loss for each element of batch.
        diffs = outputs - torch.log(labels)
        batch_size = diffs.shape[0]
        diffs = diffs.view(batch_size, -1)
        mse = torch.mean(diffs ** 2, dim=-1)
        relative = torch.sum(diffs, dim=-1) ** 2 / diffs.shape[1] ** 2
        batch_loss = mse - self.alpha * relative

        # Reduce loss over the batch.
        if self.reduction == "none":
            loss = batch_loss
        elif self.reduction == "mean":
            loss = torch.mean(batch_loss)
        elif self.reduction == "sum":
            loss = torch.sum(batch_loss)
        else:
            raise NotImplementedError

        return loss


class MultiTaskLoss(nn.Module):
    """ Computes the weighted sum of multiple loss functions. """

    def __init__(
        self,
        task_losses: List[Dict[str, Any]],
        loss_weighter_kwargs: Dict[str, Any],
        device: torch.device = None,
    ) -> None:
        """
        Init function for MultiTaskLoss.

        Parameters
        ----------
        task_losses : List[Dict[str, Any]]
            Loss functions for each task. Each element of the list is a dictionary which
            defines the loss function for a single task, with three keys:
            "output_slice", "label_slice", and "loss". The keys corresponding to
            "output_slice" and "label_slice" are functions that produce the task output
            and task label, respectively, given the multi-task output and the multi-task
            label. The key for "loss" is the task loss function.
        loss_weighter_kwargs : Dict[str, Any]
            Keyword arguments to be passed to LossWeighter instance.
        device : torch.device
            What device to store Tensors on. If None (default), defaults to CPU.
        """

        super(MultiTaskLoss, self).__init__()

        # Set device.
        self.device = device if device is not None else torch.device("cpu")

        # Set task losses.
        self.task_losses = task_losses
        self.num_tasks = len(self.task_losses)

        # Set task weighting strategy.
        loss_weighter = loss_weighter_kwargs["type"]
        del loss_weighter_kwargs["type"]
        if loss_weighter not in [
            "Constant",
            "Uncertainty",
            "GradNorm",
            "DWA",
            "MLDW",
            "LBTW",
            "NLW",
            "RWLW",
            "CLW",
            "CLAW",
            "CLAWTester",
        ]:
            raise NotImplementedError
        loss_weighter_cls = eval(loss_weighter)
        loss_weighter_kwargs["num_tasks"] = self.num_tasks
        loss_weighter_kwargs["device"] = self.device
        self.loss_weighter = loss_weighter_cls(**loss_weighter_kwargs)

        # Determine whether loss weights should be updated before or after task loss
        # computation.
        pre_loss_weighters = [Uncertainty, GradNorm, CLW, CLAW, CLAWTester]
        self.pre_loss_update = loss_weighter_cls in pre_loss_weighters

    def forward(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        train: bool = True,
        combine_losses: bool = True,
        **kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Compute values of each task losses, update the task-loss weights, then return
        the weighted sum of task losses. Extra arguments are passed to `update()` of
        `LossWeighter`.

        Parameters
        ----------
        outputs : torch.Tensor
            Multi-task network output.
        labels : torch.Tensor
            Multi-task labels.
        train : bool
            Whether or not this is a training step or testing step.
        combine_losses : bool
            Whether or not to combine task losses into a single scalar value. Defaults
            to `True`, which will almost always be used. When `False`, returns a tensor
            of shape `(len(task_losses),)` (the number of tasks), which holds the
            individual values of each task's loss function. In this case, no loss
            weighter is used.  Be careful using this option: it can easily break
            training. The task losses do have to be reduced into a single scalar
            somewhere in order to perform training.

        Returns
        -------
        loss : torch.Tensor
            Value of multi-task loss (weighted sum of task losses).
        """

        # Compute task losses.
        task_loss_vals = []
        for i, task_loss in enumerate(self.task_losses):
            task_output = task_loss["output_slice"](outputs)
            task_label = task_loss["label_slice"](labels)
            task_loss_val = task_loss["loss"](task_output, task_label)
            task_loss_vals.append(task_loss_val)
        task_loss_vals = torch.stack(task_loss_vals)

        # Return without combining task losses, if necessary.
        if not combine_losses:
            return task_loss_vals

        # Update loss weighter before loss computation, if necessary.
        if train and self.pre_loss_update:
            self.loss_weighter.update(task_loss_vals, **kwargs)

        # Compute total loss as weighted sum of task losses.
        total_loss = torch.sum(task_loss_vals * self.loss_weighter.loss_weights)

        # Add regularization term to loss when using "Weighting by Uncertainty".
        if isinstance(self.loss_weighter, Uncertainty):
            total_loss += self.loss_weighter.regularization()

        # Update loss weighter after loss computation, if necessary.
        if train and not self.pre_loss_update:
            self.loss_weighter.update(task_loss_vals, **kwargs)

        return total_loss


class LossWeighter(nn.Module):
    """ Compute task loss weights for multi-task learning. """

    def __init__(
        self, num_tasks: int, loss_weights: List[float], device: torch.device = None
    ) -> None:
        """ Init function for LossWeighter. """

        super(LossWeighter, self).__init__()

        self.device = device if device is not None else torch.device("cpu")

        # Set state.
        self.num_tasks = num_tasks
        if loss_weights is not None:
            assert len(loss_weights) == self.num_tasks
            self.loss_weights = torch.Tensor(loss_weights)
        else:
            self.loss_weights = torch.ones((self.num_tasks,))
        self.loss_weights = self.loss_weights.to(self.device)
        self.initial_loss_weights = torch.clone(self.loss_weights)
        self.total_weight = torch.sum(self.loss_weights)
        self.loss_history = []
        self.steps = 0
        self.MAX_HISTORY_LEN = 2

    def update(self, loss_vals: torch.Tensor, **kwargs: Dict[str, Any]) -> None:
        """
        Compute new loss weights using most recent values of task losses. Extra
        arguments are passed to `self._update_weights()`.
        """
        self.loss_history.append(loss_vals.detach())
        self.loss_history = self.loss_history[-self.MAX_HISTORY_LEN :]
        if (
            isinstance(self, GradNorm)
            or isinstance(self, CLW)
            or isinstance(self, CLAWTester)
        ):
            kwargs["loss_vals"] = loss_vals
        self._update_weights(**kwargs)
        self.steps += 1

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


class Uncertainty(LossWeighter):
    """
    Compute task loss weights with "Weighting by Uncertainty", detailed in
    https://arxiv.org/abs/1705.07115.
    """

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        """ Init function for Uncertainty. """
        super(Uncertainty, self).__init__(**kwargs)

        # Initialize weights. Unlike the other loss weighting methods, these are
        # actually learned parameters. Instead of learning the weights directly, we
        # learn s_i = -log(2w_i), so w_i = exp(-s_i) / 2, for numerical stability, as
        # suggested in the paper. We initialize the values of s_i so that each w_i is
        # equal to the given weight value.
        with torch.no_grad():
            log_variance_t = -torch.log(2.0 * self.loss_weights)
        self.log_variance = nn.Parameter(log_variance_t)

    def regularization(self) -> torch.Tensor:
        """ Compute regularization term on loss weights. """
        return torch.sum(self.log_variance / 2.0)

    def _update_weights(self) -> None:
        """ Compute new loss weights. """
        self.loss_weights = 0.5 * torch.exp(-self.log_variance)


class GradNorm(LossWeighter):
    """
    Compute task loss weights with GradNorm, detailed in
    https://arxiv.org/abs/1711.02257.
    """

    def __init__(
        self, asymmetry: float, weight_lr: float, **kwargs: Dict[str, Any],
    ) -> None:
        """ Init function for GradNorm. """

        super(GradNorm, self).__init__(**kwargs)

        # Save state.
        self.asymmetry = asymmetry
        self.weight_lr = weight_lr
        self.grad_len = None
        self.baseline_losses = None

        # Create parameter for loss weights and the corresponding optimizer.
        self.loss_weight_p = nn.Parameter(self.loss_weights.clone().detach())
        self.loss_weight_optim = torch.optim.Adam(self.parameters(), lr=self.weight_lr)

    def _update_weights(self, loss_vals: torch.Tensor, network: nn.Module) -> None:
        """ Compute new loss weights. """

        # Store losses from the first training step.
        if self.baseline_losses is None:
            self.baseline_losses = self.loss_history[-1]

        # Compute gradients of each task's loss. Note that `last_shared_params()` may
        # not be defined for any instance of `nn.Module`, so there will be compatibility
        # issues with other networks.
        if self.grad_len is None:
            task_grads = None
        else:
            task_grads = torch.zeros(
                (self.num_tasks, self.grad_len), device=self.device
            )
        for task in range(self.num_tasks):
            task_grad = torch.autograd.grad(
                loss_vals[task], network.last_shared_params(), retain_graph=True
            )
            task_grad = torch.cat([grad.view(-1) for grad in task_grad])

            if self.grad_len is None:
                self.grad_len = int(task_grad.shape[0])
                task_grads = torch.zeros(
                    (self.num_tasks, self.grad_len), device=self.device
                )

            task_grads[task] = task_grad.detach()

        # Compute weighted gradient norms.
        task_grad_norms = torch.sqrt(torch.sum(task_grads ** 2, dim=-1))
        task_grad_norms *= self.loss_weight_p

        # Compute gradient norm targets from the average gradient norm and relative
        # inverse training rates. These are treated as constants in the GradNorm loss
        # function.
        with torch.no_grad():
            avg_grad_norm = torch.mean(task_grad_norms)

            inv_rate = self.loss_history[-1] / self.baseline_losses
            avg_inv_rate = torch.mean(inv_rate)
            rel_inv_rate = inv_rate / avg_inv_rate

            grad_norm_target = avg_grad_norm * (rel_inv_rate ** self.asymmetry)

        # Compute GradNorm loss.
        gradnorm_loss = torch.sum(torch.abs(task_grad_norms - grad_norm_target))

        # Update loss weights.
        self.loss_weight_optim.zero_grad()
        gradnorm_loss.backward()
        self.loss_weight_optim.step()

        # Renormalize loss weights and assign values from internal parameter to
        # `self.loss_weights`.
        with torch.no_grad():
            self.loss_weight_p /= torch.sum(self.loss_weight_p)
            self.loss_weight_p *= self.total_weight
            self.loss_weights = self.loss_weight_p.data.clone().detach()


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
    Compute task loss weights with Centered Loss Weighting. At each step, we compute the
    gradient of each task's loss function, set each task's weight equal to the
    reciprocal of the norm of this gradient, then normalize weights.
    """

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        """ Init function for CLW. """

        super(CLW, self).__init__(**kwargs)

        # Save state.
        self.grad_len = None

    def _update_weights(self, loss_vals: torch.Tensor, network: nn.Module) -> None:
        """ Compute new loss weights with CLW. """

        # Compute gradients of each task's loss. Note that `last_shared_params()` may
        # not be defined for any instance of `nn.Module`, so there will be compatibility
        # issues with other networks.
        if self.grad_len is None:
            task_grads = None
        else:
            task_grads = torch.zeros(
                (self.num_tasks, self.grad_len), device=self.device
            )

        for task in range(self.num_tasks):
            task_grad = torch.autograd.grad(
                loss_vals[task], network.last_shared_params(), retain_graph=True
            )
            task_grad = torch.cat([grad.view(-1) for grad in task_grad])

            if self.grad_len is None:
                self.grad_len = int(task_grad.shape[0])
                task_grads = torch.zeros(
                    (self.num_tasks, self.grad_len), device=self.device
                )

            task_grads[task] = task_grad.detach()

        # Compute gradient norms.
        task_grad_norms = torch.sqrt(torch.sum(task_grads ** 2, dim=-1))

        # Set loss weights equal to inverse of gradient norms, then normalize the
        # weights so they sum to the initial total weight.
        threshold_norm = torch.max(
            task_grad_norms, EPSILON * torch.ones_like(task_grad_norms)
        )
        self.loss_weights = 1.0 / threshold_norm
        self.loss_weights /= torch.sum(self.loss_weights)
        self.loss_weights *= self.total_weight


class CLAW(LossWeighter):
    """
    Compute task loss weights with Centered Loss Approximated Weighting. Here we keep a
    running std of each task's loss, and set each task's loss weight equal to the
    inverse of the std of the task loss.
    """

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        """ Init function for CLAW. """
        super(CLAW, self).__init__(**kwargs)

        self.loss_stats = RunningStats(
            compute_stdev=True,
            shape=(self.num_tasks,),
            ema_alpha=0.99,
            device=self.device,
        )

    def _update_weights(self) -> None:
        """ Compute new loss weights with CLAW. """

        # Update stats.
        self.loss_stats.update(self.loss_history[-1])

        # Set loss weights equal to inverse of loss stdev, then normalize the weights so
        # they sum to the initial total weight. Note that we don't update the weights
        # until after the first step, since at that point each stdev is undefined.
        if self.steps > 0 and not any(torch.isnan(self.loss_stats.stdev)):
            threshold_stdev = torch.max(
                self.loss_stats.stdev, EPSILON * torch.ones_like(self.loss_stats.stdev)
            )
            self.loss_weights = 1.0 / threshold_stdev
            self.loss_weights /= torch.sum(self.loss_weights)
            self.loss_weights *= self.total_weight


class CLAWTester(LossWeighter):
    """
    Utility class to evaluate CLAW as an approximation to CLW. This class shouldn't ever
    be used for training because it will call `exit()` at a random step during the
    training process.
    """

    def __init__(
        self,
        step_bounds: Tuple[int, int],
        claw_data_path: str,
        claw_log_path: str,
        **kwargs: Dict[str, Any],
    ) -> None:
        """ Init function for CLAWTester. """

        super(CLAWTester, self).__init__(**kwargs)

        # Save state.
        self.step_bounds = step_bounds
        self.claw_data_path = claw_data_path
        self.claw_log_path = claw_log_path
        self.grad_len = None

        # Randomly generate the step index at which the CLAW vs. CLW comparison will be
        # made.
        self.last_step = random.randrange(self.step_bounds[0], self.step_bounds[1])

        self.loss_stats = RunningStats(
            compute_stdev=True,
            shape=(self.num_tasks,),
            ema_alpha=0.99,
            device=self.device,
        )

    def _update_weights(self, loss_vals: torch.Tensor, network: nn.Module) -> None:
        """ Compare CLW and CLAW weights. """

        # Update stats.
        self.loss_stats.update(self.loss_history[-1])

        # Determine whether or not to perform comparison on the current update step. If
        # so, then write out results to file and exit training.
        if self.steps >= self.last_step:

            # Compute gradients of each task's loss. Note that `last_shared_params()`
            # may not be defined for any instance of `nn.Module`, so there will be
            # compatibility issues with other networks.
            if self.grad_len is None:
                task_grads = None
            else:
                task_grads = torch.zeros(
                    (self.num_tasks, self.grad_len), device=self.device
                )

            for task in range(self.num_tasks):
                task_grad = torch.autograd.grad(
                    loss_vals[task], network.last_shared_params(), retain_graph=True
                )
                task_grad = torch.cat([grad.view(-1) for grad in task_grad])

                if self.grad_len is None:
                    self.grad_len = int(task_grad.shape[0])
                    task_grads = torch.zeros(
                        (self.num_tasks, self.grad_len), device=self.device
                    )

                task_grads[task] = task_grad.detach()

            # Compute CLW weights.
            task_grad_norms = torch.sqrt(torch.sum(task_grads ** 2, dim=-1))
            norm_recip = 1.0 / task_grad_norms
            clw_weights = self.num_tasks * norm_recip / torch.sum(norm_recip)

            # Compute CLAW weights.
            if self.steps > 0 and not any(torch.isnan(self.loss_stats.stdev)):
                approx_norms = torch.max(
                    self.loss_stats.stdev,
                    EPSILON * torch.ones_like(self.loss_stats.stdev),
                )
                approx_norm_recip = 1.0 / approx_norms
                claw_weights = (
                    self.num_tasks * approx_norm_recip / torch.sum(approx_norm_recip)
                )
            else:
                claw_weights = None

            # Save and exit if CLAW weights are valid.
            if claw_weights is not None:

                clw_weights = clw_weights.cpu().numpy()
                claw_weights = claw_weights.cpu().numpy()
                current_entry = np.stack([clw_weights, claw_weights])
                current_entry = np.expand_dims(current_entry, 0)

                if os.path.isfile(self.claw_data_path):
                    total_entries = np.load(self.claw_data_path)
                    total_entries = np.concatenate([total_entries, current_entry])
                else:
                    total_entries = current_entry
                np.save(self.claw_data_path, total_entries)

                with open(self.claw_log_path, "a+") as log_file:
                    log_file.write(f"Exiting at step {self.steps}\n")

                exit()


def save_batch(
    task_losses: Dict[str, Any], inputs: torch.Tensor, outputs: torch.Tensor, labels: torch.Tensor
) -> None:
    """
    Debug function to save out batch of NYUv2 labels as images and exit. Should be
    called from `forward()` in `MultiTaskLoss`. Note that this should only be used when
    doing multi-task training on the NYUv2 dataset.
    """

    inputs = inputs.detach().cpu()
    outputs = outputs.detach().cpu()
    labels = labels.detach().cpu()

    loss_names = ["seg", "sn", "depth"]
    assert outputs.shape[0] == inputs.shape[0]
    assert inputs.shape[0] == labels.shape[0]
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

    for j in range(batch_size):
        img_arr = np.transpose(inputs[j].numpy(), (1, 2, 0))
        img_arr = (img_arr + 1.0) / 2.0
        img_arr = np.uint8(img_arr * 255.0)
        img = Image.fromarray(img_arr)
        img.save("test_%d_input.png" % j)

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


def get_accuracy(
    outputs: torch.Tensor, labels: torch.Tensor, criterion: nn.Module = None
) -> float:
    """
    Compute accuracy of classification prediction given outputs and labels.
    """
    accuracy = torch.sum(torch.argmax(outputs, dim=-1) == labels) / outputs.shape[0]
    return accuracy.item()


def NYUv2_seg_pixel_accuracy(
    outputs: torch.Tensor, labels: torch.Tensor, criterion: nn.Module = None
) -> float:
    """
    Compute pixel accuracy of semantic segmentation on the NYUv2 dataset. Here we assume
    that any pixels with label -1 are unlabeled, so we don't count these pixels in the
    accuracy computation. We also assume that the class dimension is directly after the
    batch dimension.
    """
    preds = torch.argmax(outputs, dim=1)
    correct = torch.sum(preds == labels)
    valid = torch.sum(labels != -1)
    accuracy = correct / valid
    return accuracy.item()


def NYUv2_seg_class_accuracy(
    outputs: torch.Tensor, labels: torch.Tensor, criterion: nn.Module = None
) -> float:
    """
    Compute class accuracy of semantic segmentation on the NYUv2 dataset. Here we assume
    that any pixels with label -1 are unlabeled, so we don't count these pixels in the
    accuracy computation. We also assume that the class dimension is directly after the
    batch dimension.
    """

    # Get predictions.
    preds = torch.argmax(outputs, dim=1)

    # Get list of all labels in image.
    unlabel = -1
    all_labels = labels.unique().tolist()
    if unlabel in all_labels:
        all_labels.remove(unlabel)

    # Compute accuracy per-class.
    class_accuracies = torch.zeros(len(all_labels), device=outputs.device)
    for i, label in enumerate(all_labels):
        class_correct = torch.sum(torch.logical_and(preds == label, labels == label))
        class_valid = torch.sum(labels == label)
        class_accuracies[i] = class_correct / class_valid

    # Return average class accuracy.
    return class_accuracies.mean().item()


def NYUv2_seg_class_IOU(
    outputs: torch.Tensor, labels: torch.Tensor, criterion: nn.Module = None
) -> float:
    """
    Compute mean of IOU for each class of semantic segmentation on the NYUv2 dataset.
    Here we assume that any pixels with label -1 are unlabeled, so we don't count these
    pixels in the accuracy computation. We also assume that the class dimension is
    directly after the batch dimension.
    """

    # Get predictions.
    preds = torch.argmax(outputs, dim=1)

    # Get list of all labels in image.
    unlabel = -1
    all_labels = labels.unique().tolist()
    if unlabel in all_labels:
        all_labels.remove(unlabel)

    # Compute IOU per-class.
    class_IOUs = torch.zeros(len(all_labels), device=outputs.device)
    for i, label in enumerate(all_labels):
        class_intersection = torch.sum(
            torch.logical_and(preds == label, labels == label)
        )
        class_union = torch.sum(torch.logical_or(preds == label, labels == label))
        class_IOUs[i] = class_intersection / class_union

    # Return average class accuracy.
    return class_IOUs.mean().item()


def get_NYUv2_sn_accuracy(
    threshold: float,
) -> Callable[[torch.Tensor, torch.Tensor, nn.Module], float]:
    """
    Constructs and returns a function that computes the percentage of surface normal
    predictions which are within `threshold` degrees of the ground truth.
    """

    def NYUv2_sn_accuracy(
        outputs: torch.Tensor, labels: torch.Tensor, criterion: nn.Module = None
    ) -> float:
        """
        Compute accuracy of surface normal estimation on the NYUv2 dataset. We define
        this as the number of pixels for which the angle between the true normal and the
        predicted normal is less than `threshold` degrees. Here we assume that the
        normal dimension is 1.
        """
        similarity_threshold = math.cos(threshold / 180 * math.pi)
        similarity = F.cosine_similarity(outputs, labels, dim=1)
        accuracy = torch.sum(similarity > similarity_threshold) / torch.numel(
            similarity
        )

        return accuracy.item()

    return NYUv2_sn_accuracy


def NYUv2_sn_angle(
    outputs: torch.Tensor, labels: torch.Tensor, criterion: nn.Module = None
) -> float:
    """
    Compute the mean angle between ground truth and predicted normal for the NYUv2
    dataset.
    """
    cos = F.cosine_similarity(outputs, labels, dim=1)
    cos = torch.clamp(cos, -1, 1)
    angle = torch.acos(cos) * 180 / math.pi
    return torch.mean(angle).item()


def get_NYUv2_depth_accuracy(
    threshold: float,
) -> Callable[[torch.Tensor, torch.Tensor, nn.Module], float]:
    """
    Construct and return a function that computes the accuracy of depth predictions at
    threshold `threshold`. Note that the network is trained to compute the log-depth (in
    `ScaleInvariantDepthLoss`).
    """

    def NYUv2_depth_accuracy(
        outputs: torch.Tensor, labels: torch.Tensor, criterion: nn.Module = None
    ) -> float:
        """
        Compute accuracy of depth prediction on the NYUv2 dataset. We define this as the
        number of pixels for which the ratio between the predicted depth and the true depth
        is less than `threshold`.
        """
        preds = torch.exp(outputs)
        ratio = torch.max(preds / labels, labels / preds)
        accuracy = torch.sum(ratio < threshold) / torch.numel(ratio)
        return accuracy.item()

    return NYUv2_depth_accuracy


def NYUv2_depth_RMSE(
    outputs: torch.Tensor, labels: torch.Tensor, criterion: nn.Module = None
) -> float:
    """ Root mean-square error for NYUv2 depth prediction. """
    preds = torch.exp(outputs)
    return torch.sqrt(torch.mean((preds - labels) ** 2)).item()


def NYUv2_depth_log_RMSE(
    outputs: torch.Tensor, labels: torch.Tensor, criterion: nn.Module = None
) -> float:
    """ RMSE of log-prediction and log-ground truth for NYUv2 depth prediction. """
    return torch.sqrt(torch.mean((outputs - torch.log(labels)) ** 2)).item()


def NYUv2_depth_invariant_RMSE(
    outputs: torch.Tensor, labels: torch.Tensor, criterion: nn.Module = None
) -> float:
    """
    Scale-invariant RMSE of log-prediction and log-ground truth for NYUv2 depth
    prediction. Note that the network is trained to compute the log-depth (in
    `ScaleInvariantDepthLoss`).
    """
    diffs = outputs - torch.log(labels)
    mse = torch.mean(diffs ** 2)
    relative = torch.sum(diffs) ** 2 / torch.numel(diffs) ** 2
    return torch.sqrt(mse - relative).item()


def NYUv2_multi_seg_pixel_accuracy(
    outputs: torch.Tensor, labels: torch.Tensor, criterion: nn.Module = None
) -> float:
    """
    Compute pixel accuracy of semantic segmentation on the NYUv2 dataset when performing
    multi-task training. This function is essentially a wrapper around
    `NYUv2_seg_pixel_accuracy()`.
    """
    task_outputs = outputs[:, :13]
    task_labels = labels[:, 0].long()
    return NYUv2_seg_pixel_accuracy(task_outputs, task_labels, criterion)


def NYUv2_multi_seg_class_accuracy(
    outputs: torch.Tensor, labels: torch.Tensor, criterion: nn.Module = None
) -> float:
    """
    Compute accuracy of semantic segmentation on the NYUv2 dataset when performing
    multi-task training. This function is essentially a wrapper around
    `NYUv2_seg_class_accuracy()`.
    """
    task_outputs = outputs[:, :13]
    task_labels = labels[:, 0].long()
    return NYUv2_seg_class_accuracy(task_outputs, task_labels, criterion)


def NYUv2_multi_seg_class_IOU(
    outputs: torch.Tensor, labels: torch.Tensor, criterion: nn.Module = None
) -> float:
    """
    Compute accuracy of semantic segmentation on the NYUv2 dataset when performing
    multi-task training. This function is essentially a wrapper around
    `NYUv2_seg_class_IOU()`.
    """
    task_outputs = outputs[:, :13]
    task_labels = labels[:, 0].long()
    return NYUv2_seg_class_IOU(task_outputs, task_labels, criterion)


def get_NYUv2_multi_sn_accuracy(
    threshold: float,
) -> Callable[[torch.Tensor, torch.Tensor, nn.Module], float]:
    """
    Constructs and returns a function that computes the percentage of surface normal
    predictions which are within `threshold` degrees of the ground truth when multi-task
    training.
    """

    NYUv2_sn_accuracy = get_NYUv2_sn_accuracy(threshold)

    def NYUv2_multi_sn_accuracy(
        outputs: torch.Tensor, labels: torch.Tensor, criterion: nn.Module = None
    ) -> float:
        """
        Compute accuracy of surface normal estimation on the NYUv2 dataset when performing
        multi-task training. This function is essentially a wrapper around
        `get_NYUv2_sn_accuracy()`.
        """
        task_outputs = outputs[:, 13:16]
        task_labels = labels[:, 1:4]
        return NYUv2_sn_accuracy(task_outputs, task_labels)

    return NYUv2_multi_sn_accuracy


def NYUv2_multi_sn_angle(
    outputs: torch.Tensor, labels: torch.Tensor, criterion: nn.Module = None
) -> float:
    """
    Compute mean error angle of surface normal estimation on the NYUv2 dataset when
    performing multi-task training. This function is essentially a wrapper around
    `NYUv2_sn_angle()`.
    """
    task_outputs = outputs[:, 13:16]
    task_labels = labels[:, 1:4]
    return NYUv2_sn_angle(task_outputs, task_labels)


def get_NYUv2_multi_depth_accuracy(
    threshold: float,
) -> Callable[[torch.Tensor, torch.Tensor, nn.Module], float]:
    """
    Construct and return a function that computes the accuracy of depth predictions at
    threshold `threshold` while multi-task training.
    """

    NYUv2_depth_accuracy = get_NYUv2_depth_accuracy(threshold)

    def NYUv2_multi_depth_accuracy(
        outputs: torch.Tensor, labels: torch.Tensor, criterion: nn.Module = None
    ) -> float:
        """
        Compute accuracy of depth prediction on the NYUv2 dataset when performing
        multi-task training. This function is essentially a wrapper around
        `get_NYUv2_depth_accuracy()`.
        """
        task_outputs = outputs[:, 16:17]
        task_labels = labels[:, 4:5]
        return NYUv2_depth_accuracy(task_outputs, task_labels)

    return NYUv2_multi_depth_accuracy


def NYUv2_multi_depth_RMSE(
    outputs: torch.Tensor, labels: torch.Tensor, criterion: nn.Module = None
) -> float:
    """
    Compute RMSE of depth prediction on the NYUv2 dataset when performing multi-task
    training. This function is essentially a wrapper around `NYUv2_depth_RMSE()`.
    """
    task_outputs = outputs[:, 16:17]
    task_labels = labels[:, 4:5]
    return NYUv2_depth_RMSE(task_outputs, task_labels)


def NYUv2_multi_depth_log_RMSE(
    outputs: torch.Tensor, labels: torch.Tensor, criterion: nn.Module = None
) -> float:
    """
    Compute log-RMSE of depth prediction on the NYUv2 dataset when performing multi-task
    training. This function is essentially a wrapper around `NYUv2_depth_log_RMSE()`.
    """
    task_outputs = outputs[:, 16:17]
    task_labels = labels[:, 4:5]
    return NYUv2_depth_log_RMSE(task_outputs, task_labels)


def NYUv2_multi_depth_invariant_RMSE(
    outputs: torch.Tensor, labels: torch.Tensor, criterion: nn.Module = None
) -> float:
    """
    Compute scale-invariant RMSE of depth prediction on the NYUv2 dataset when
    performing multi-task training. This function is essentially a wrapper around
    `NYUv2_depth_invariant_RMSE()`.
    """
    task_outputs = outputs[:, 16:17]
    task_labels = labels[:, 4:5]
    return NYUv2_depth_invariant_RMSE(task_outputs, task_labels)


def NYUv2_multi_avg_accuracy(
    outputs: torch.Tensor, labels: torch.Tensor, criterion: nn.Module = None
) -> float:
    """
    Compute average accuracy of the three tasks on the NYUv2 dataset when performing
    multi-task training.
    """
    seg_accuracy = NYUv2_multi_seg_pixel_accuracy(outputs, labels)
    sn_accuracy = get_NYUv2_multi_sn_accuracy(11.25)(outputs, labels)
    depth_accuracy = get_NYUv2_multi_depth_accuracy(1.25)(outputs, labels)
    return np.mean([seg_accuracy, sn_accuracy, depth_accuracy])


def NYUv2_multi_var_accuracy(
    outputs: torch.Tensor, labels: torch.Tensor, criterion: nn.Module = None
) -> float:
    """
    Compute variance of accuracies of the three tasks on the NYUv2 dataset when
    performing multi-task training.
    """
    seg_accuracy = NYUv2_multi_seg_pixel_accuracy(outputs, labels)
    sn_accuracy = get_NYUv2_multi_sn_accuracy(11.25)(outputs, labels)
    depth_accuracy = get_NYUv2_multi_depth_accuracy(1.25)(outputs, labels)
    return np.var([seg_accuracy, sn_accuracy, depth_accuracy])


def get_MTRegression_normal_loss(
    num_tasks: int,
) -> Callable[[torch.Tensor, torch.Tensor, nn.Module], float]:
    """
    Constructs and returns a function which computes the MTRegression normalized
    multi-task loss from a set of labels and the corresponding predictions.
    """

    scales = np.array(SCALES[num_tasks])

    def normal_loss(
        outputs: torch.Tensor, labels: torch.Tensor, criterion: nn.Module = None
    ) -> float:
        """
        Computes normalized multi-task loss for MTRegression task. Both `outputs` and
        `labels` should have shape `(batch_size, num_tasks, output_dim)`.
        """
        diffs = torch.sum((outputs - labels) ** 2, dim=2).detach()
        diffs = torch.mean(diffs, dim=0)
        diffs = diffs if diffs.device == torch.device("cpu") else diffs.cpu()
        diffs = diffs.numpy()
        weighted_diffs = np.mean(diffs / (scales ** 2))
        return float(weighted_diffs)

    return normal_loss


def get_MTRegression_normal_loss_variance(
    num_tasks: int,
) -> Callable[[torch.Tensor, torch.Tensor, nn.Module], float]:
    """
    Constructs and returns a function which computes the variance of the MTRegression
    normalized loss across tasks from a set of labels and the corresponding predictions.
    """

    scales = np.array(SCALES[num_tasks])

    def normal_loss_variance(
        outputs: torch.Tensor, labels: torch.Tensor, criterion: nn.Module = None
    ) -> float:
        """
        Computes variance of normalized loss across tasks for MTRegression task. Both
        `outputs` and `labels` should have shape `(batch_size, num_tasks, output_dim)`.
        """
        diffs = torch.sum((outputs - labels) ** 2, dim=2).detach()
        diffs = torch.mean(diffs, dim=0)
        diffs = diffs if diffs.device == torch.device("cpu") else diffs.cpu()
        diffs = diffs.numpy()
        diffs = diffs / (scales ** 2)
        return float(np.var(diffs))

    return normal_loss_variance


def get_MTRegression_weight_error(
    num_tasks: int,
) -> Callable[[torch.Tensor, torch.Tensor, nn.Module], float]:
    """
    Constructs and returns a function which computes the MSE from the current
    (normalized) loss weights to the ideal loss weights for the MTRegression dataset.
    """

    scales = np.array(SCALES[num_tasks])
    ideal_weights = 1.0 / (scales ** 2)
    ideal_weights *= num_tasks / np.sum(ideal_weights)

    def weight_error(
        outputs: torch.Tensor, labels: torch.Tensor, criterion: nn.Module = None
    ) -> float:
        """
        Computes MSE from the current (normalized) loss weights to the ideal loss
        weights for the MTRegression dataset.
        """
        assert isinstance(criterion, MultiTaskLoss)
        current_weights = criterion.loss_weighter.loss_weights.detach().cpu().numpy()
        normalized_weights = num_tasks * current_weights / np.sum(current_weights)
        error = np.mean((normalized_weights - ideal_weights) ** 2)
        return error

    return weight_error


def PCBA_ROC_AUC(
    outputs: torch.Tensor, labels: torch.Tensor, criterion: nn.Module = None
) -> float:
    """
    Computes the area under the ROC curve for the PCBA binary classification task.
    """
    softmax_outputs = F.softmax(outputs, dim=2)
    flat_outputs = softmax_outputs[:, :, 1].reshape(-1)
    flat_labels = labels.reshape(-1)
    valid = torch.logical_or(flat_labels == 0, flat_labels == 1)
    valid_outputs = flat_outputs[valid].detach().cpu().numpy()
    valid_labels = flat_labels[valid].detach().cpu().long().numpy()
    score = roc_auc_score(valid_labels, valid_outputs, average="samples")
    return score


def PCBA_avg_precision(
    outputs: torch.Tensor, labels: torch.Tensor, criterion: nn.Module = None
) -> float:
    """
    Computes the average precision (AP) for the PCBA binary classification task.
    """
    softmax_outputs = F.softmax(outputs, dim=2)
    flat_outputs = softmax_outputs[:, :, 1].reshape(-1)
    flat_labels = labels.reshape(-1)
    valid = torch.logical_or(flat_labels == 0, flat_labels == 1)
    valid_outputs = flat_outputs[valid].detach().cpu().numpy()
    valid_labels = flat_labels[valid].detach().cpu().long().numpy()
    score = average_precision_score(valid_labels, valid_outputs, average="samples")
    return score


def get_multitask_loss_weight(
    task: int,
) -> Callable[[torch.Tensor, torch.Tensor, nn.Module], float]:
    """
    Constructs and returns a function which returns the multi-task loss weight for a
    given task from the multi-task loss function. Note that this should only be used
    when the given loss function is an instance of `MultiTaskLoss`.
    """

    def multitask_loss_weight(
        outputs: torch.Tensor, labels: torch.Tensor, criterion: nn.Module = None
    ) -> float:
        """
        Returns the multi-task loss weight for a given task from `criterion`, which
        should be an instance of `MultiTaskLoss`.
        """
        assert isinstance(criterion, MultiTaskLoss)
        return float(criterion.loss_weighter.loss_weights[task])

    return multitask_loss_weight
