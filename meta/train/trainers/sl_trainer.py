""" Definition of SLTrainer class for supervised learning. """

import os
import time
from copy import deepcopy
from itertools import chain
from math import ceil
from typing import Dict, Iterator, Iterable, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from meta.train.trainers.base_trainer import Trainer
from meta.datasets import *
from meta.train.loss import MultiTaskLoss, Uncertainty
from meta.networks import (
    ConvNetwork,
    ResNetwork,
    BackboneNetwork,
    MLPNetwork,
    MultiTaskTrunkNetwork,
    BaseMultiTaskSplittingNetwork,
    MultiTaskSplittingNetworkV1,
    MultiTaskSplittingNetworkV2,
)
from meta.utils.utils import aligned_train_configs, DATA_DIR


class SLTrainer(Trainer):
    """ Trainer class for supervised learning. """

    def init_model(self, config: Dict[str, Any]) -> None:
        """
        Initialize model and corresponding objects. The expected entries of `config` are
        listed below. `config` should contain all entries listed in the docstring of
        Trainer, as well as the settings specific to SLTrainer, which are listed below.

        Parameters
        ----------
        dataset : str
            Dataset to train on.
        num_updates : int
            Number of training epochs.
        batch_size : int
            Size of each minibatch on which to compute gradient updates.
        num_workers : int
            Number of worker processes to load data.
        """

        # Construct dataset.
        dataset_name = config["dataset"]
        if dataset_name not in DATASETS:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        dataset_cls = eval(dataset_name)
        root = os.path.join(DATA_DIR, dataset_name)
        dataset_kwargs = config["dataset_kwargs"]
        self.train_set = dataset_cls(root=root, train=True, **dataset_kwargs)
        self.test_set = dataset_cls(root=root, train=False, **dataset_kwargs)

        # Construct data loaders.
        self.batch_size = config["batch_size"]
        train_loader = torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=config["num_workers"],
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=config["num_workers"],
        )
        self.train_iter = iter(cycle(train_loader))

        # Set length of window for moving average of metrics for training and
        # evaluation. For both training and evaluation, we average over the last epoch.
        # Since each evaluation step iterates over the entire epoch, we only need to
        # look at the most recent metric values to get the metrics for the last epoch.
        self.train_window = ceil(len(self.train_set) / self.batch_size)
        self.eval_window = 1

        # Determine type of network to construct.
        input_size = self.train_set.input_size
        output_size = self.train_set.output_size
        network_kwargs = dict(config["architecture_config"])
        if config["architecture_config"]["type"] == "backbone":
            network_cls = BackboneNetwork
        elif config["architecture_config"]["type"] == "conv":
            assert isinstance(input_size, tuple) and len(input_size) == 3
            network_cls = ConvNetwork
        elif config["architecture_config"]["type"] == "resnet":
            assert isinstance(input_size, tuple) and len(input_size) == 3
            network_cls = ResNetwork
        elif config["architecture_config"]["type"] == "mlp":
            assert isinstance(input_size, int)
            network_cls = MLPNetwork
        elif config["architecture_config"]["type"] == "trunk":
            assert isinstance(input_size, int)
            network_cls = MultiTaskTrunkNetwork
        elif config["architecture_config"]["type"] == "splitting_v1":
            network_cls = MultiTaskSplittingNetworkV1
        elif config["architecture_config"]["type"] == "splitting_v2":
            network_cls = MultiTaskSplittingNetworkV2
        else:
            raise NotImplementedError

        # Construct network.
        del network_kwargs["type"]
        network_kwargs["input_size"] = input_size
        network_kwargs["output_size"] = output_size
        network_kwargs["device"] = self.device
        self.network = network_cls(**network_kwargs)

        # Construct loss function.
        loss_cls = self.train_set.loss_cls
        loss_kwargs = dict(self.train_set.loss_kwargs)
        if "loss_weighter" in config:
            loss_kwargs["loss_weighter_kwargs"] = dict(config["loss_weighter"])
        self.criterion = loss_cls(**loss_kwargs)
        self.criterion = self.criterion.to(self.device)

        # Construct arguments to `self.criterion`. These are passed as arguments to the
        # forward pass through `self.criterion`. Here we include the network itself as
        # an argument to the loss function, since computing the task-specific gradients
        # requires zero-ing out gradients between tasks, and this requires access to the
        # Module containing the relevant parameters. Note that this will need to change
        # in the case that GradNorm is operating over parameters outside of
        # `self.network`, or if the task-specific loss functions are dependent on
        # parameters outside of `self.network`.
        criterion_kwargs = deepcopy(self.train_set.criterion_kwargs)
        if "loss_weighter" in config:
            if config["loss_weighter"]["type"] in ["GradNorm", "SLW", "SLAWTester"]:
                criterion_kwargs["train"]["network"] = self.network

        # Determine whether or not to use PCGrad for training and check for appropriate
        # settings.
        if "PCGrad" in config:
            self.pcgrad = bool(config["PCGrad"])
            if self.pcgrad:
                mt_loss = loss_cls == MultiTaskLoss
                if not mt_loss:
                    raise ValueError(
                        "If using PCGrad, loss function must be MultiTaskLoss."
                    )
                criterion_kwargs["train"]["combine_losses"] = False
        else:
            self.pcgrad = False

        # Set loss function to return task-specific losses instead of the combined
        # weighted loss.
        if isinstance(self.network, BaseMultiTaskSplittingNetwork):
            criterion_kwargs["train"]["combine_losses"] = False

        self.criterion_kwargs = dict(criterion_kwargs)

    def _step(self) -> Dict[str, Any]:
        """ Perform one training step. """

        # Start timer for current step.
        start_time = time.time()

        # Sample a batch and move it to device.
        inputs, labels = next(self.train_iter)
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        # Zero gradients.
        self.optimizer.zero_grad()

        # Perform forward pass and compute loss.
        outputs = self.network(inputs)
        loss = self.criterion(outputs, labels, **self.criterion_kwargs["train"])

        # If we are training a splitting network, check for splits. After this, reduce
        # task losses into a single loss with a weighted sum.
        if isinstance(self.network, BaseMultiTaskSplittingNetwork):
            self.network.check_for_split(loss)
            self.init_optimizer()
            loss = torch.sum(loss * self.criterion.loss_weighter.loss_weights)

        # Perform backward pass, clip gradient, and take optimizer step.
        self._compute_grad(loss)
        self.clip_grads()
        self.optimizer.step()

        # Stop timer for current step.
        end_time = time.time()
        train_step_time = end_time - start_time

        # Compute metrics from training step.
        step_metrics = {
            "train_loss": [torch.sum(loss).item()],
            "train_step_time": [train_step_time],
        }
        with torch.no_grad():
            extra_metrics = self.train_set.compute_metrics(
                outputs, labels, self.criterion, train=True
            )
            step_metrics.update({key: [val] for key, val in extra_metrics.items()})

        return step_metrics

    def _compute_grad(self, loss: torch.Tensor) -> None:
        """
        Compute gradient of task parameters with respect to `loss`. This may be as
        simple as calling `backward()`, but for edge cases (such as PCGrad) we have to
        do some funky stuff.
        """

        # Use PCGrad. Compute the gradient of each task's loss individually, and use the
        # PCGrad projection rule to modify the gradients for each parameter which is
        # shared between all tasks. Note that this implementation assumes that each
        # parameter of the network is either shared between all tasks or specific to a
        # single task.
        if self.pcgrad:

            # Compute and store gradients of each task loss.
            num_tasks = self.criterion.num_tasks
            shared_grads = torch.zeros(
                num_tasks, self.network.num_shared_params, device=self.device
            )
            specific_grads = [
                torch.zeros(self.network.num_specific_params[task], device=self.device)
                for task in range(num_tasks)
            ]
            for task in range(num_tasks):
                loss[task].backward(retain_graph=True)
                shared_grads[task] = torch.cat(
                    [p.grad.view(-1) for p in self.network.shared_params()]
                )
                specific_grads[task] = torch.cat(
                    [p.grad.view(-1) for p in self.network.specific_params(task)]
                )
                self.network.zero_grad()

            # Project the gradients of shared parameters to avoid pairwise conflicts.
            with torch.no_grad():
                new_shared_grads = torch.clone(shared_grads)

                for i in range(num_tasks):
                    other_tasks = list(range(num_tasks))
                    other_tasks.remove(i)
                    other_tasks = np.array(other_tasks)
                    np.random.shuffle(other_tasks)

                    for j in other_tasks:

                        # Project a single gradient if the current pair is conflicting.
                        gi = new_shared_grads[i]
                        gj = shared_grads[j]
                        dot = torch.sum(gi * gj)
                        if dot < 0:
                            sq_j = torch.sum(gj ** 2)
                            new_shared_grads[i] = gi - gj * dot / sq_j

                combined_shared_grads = torch.sum(new_shared_grads, dim=0)

            # Set the gradients of shared and task-specific parameters.
            idx = 0
            for p in self.network.shared_params():
                grad_len = p.numel()
                flattened_grad = combined_shared_grads[idx : idx + grad_len]
                reshaped_grad = flattened_grad.reshape(p.shape)
                p.grad = reshaped_grad
                idx += grad_len

            for task in range(num_tasks):
                idx = 0
                for p in self.network.specific_params(task):
                    grad_len = p.numel()
                    flattened_grad = specific_grads[task][idx : idx + grad_len]
                    reshaped_grad = flattened_grad.reshape(p.shape)
                    p.grad = reshaped_grad
                    idx += grad_len

        # Regular case, just perform backwards pass.
        else:
            loss.backward()

    def evaluate(self) -> None:
        """ Evaluate current model. """

        # Initialize metrics.
        eval_step_metrics = {
            "eval_loss": [],
            **{
                f"eval_{metric_name}": []
                for metric_name, metric_info in self.train_set.extra_metrics.items()
                if metric_info["eval"]
            },
        }
        batch_sizes = []

        # Iterate over entire test set.
        for (inputs, labels) in self.test_loader:

            # Sample a batch and move it to device.
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            batch_sizes.append(len(inputs))

            # Perform forward pass and compute loss.
            outputs = self.network(inputs)
            loss = self.criterion(outputs, labels, **self.criterion_kwargs["eval"])

            # Compute metrics from evaluation step.
            eval_step_metrics["eval_loss"].append(loss.item())
            extra_metrics = self.train_set.compute_metrics(
                outputs, labels, self.criterion, train=False
            )
            for metric_name, metric_val in extra_metrics.items():
                eval_step_metrics[metric_name].append(metric_val)

        # Average value of metrics over all batches.
        for metric_name in eval_step_metrics:
            eval_step_metrics[metric_name] = [
                np.average(eval_step_metrics[metric_name], weights=batch_sizes)
            ]

        return eval_step_metrics

    def load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """ Load trainer state from checkpoint. """

        # Make sure current config and previous config line up, then load policy.
        assert aligned_train_configs(self.config, checkpoint["config"])
        self.network = checkpoint["network"]

    def get_checkpoint(self) -> None:
        """ Return trainer state as checkpoint. """

        checkpoint = {}
        checkpoint["network"] = self.network
        checkpoint["config"] = self.config
        return checkpoint

    def close(self, save_dir: str) -> None:
        """ Clean up the training process. """
        pass

    def parameters(self) -> Iterator[nn.parameter.Parameter]:
        """ Return parameters of model. """

        # Check whether we need to add extra parameters in the case that we are
        # multi-task training with "Weighting by Uncertainty".
        if isinstance(self.criterion, MultiTaskLoss) and isinstance(
            self.criterion.loss_weighter, Uncertainty
        ):
            param_iterator = chain(
                self.network.parameters(), self.criterion.loss_weighter.parameters()
            )
        else:
            param_iterator = self.network.parameters()
        return param_iterator

    @property
    def metric_set(self) -> List[Tuple]:
        """ Set of metrics for this trainer. """

        metric_set = [
            {
                "name": "train_loss",
                "basename": "loss",
                "window": self.train_window,
                "point_avg": False,
                "maximize": False,
                "show": True,
            },
            {
                "name": "eval_loss",
                "basename": "loss",
                "window": self.eval_window,
                "point_avg": False,
                "maximize": False,
                "show": True,
            },
            {
                "name": "train_step_time",
                "basename": "train_step_time",
                "window": None,
                "point_avg": False,
                "maximize": None,
                "show": False,
            },
        ]
        for metric_name, metric_info in self.train_set.extra_metrics.items():
            for split in ["train", "eval"]:
                if metric_info[split]:
                    window = self.train_window if split == "train" else self.eval_window
                    metric_set.append(
                        {
                            "name": f"{split}_{metric_name}",
                            "basename": metric_name,
                            "window": window,
                            "point_avg": False,
                            "maximize": metric_info["maximize"],
                            "show": metric_info["show"],
                        }
                    )
        return metric_set


def cycle(iterable: Iterable[Any]) -> Any:
    """
    Generator to repeatedly cycle through an iterable. This is a hacky way to get our
    batch sampling to work with the way our Trainer class is set up. In particular, the
    data loaders are stored as members of the SLTrainer class and each call to `_step()`
    requires one sample from these data loaders. This means we can't just loop over the
    data loaders, we have to sample the next batch one at a time.
    """
    while True:
        for x in iterable:
            yield x
