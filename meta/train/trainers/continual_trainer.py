""" Definition of ContinualTrainer class for continual learning. """

import os
import time
from copy import deepcopy
from itertools import chain
from math import ceil
from typing import Dict, Iterator, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image

from meta.train.trainers.base_trainer import Trainer
from meta.train.trainers.utils import cycle
from meta.datasets import *
from meta.networks import (
    ConvNetwork,
    BackboneNetwork,
    MLPNetwork,
    MultiTaskTrunkNetwork,
)
from meta.utils.utils import DATA_DIR


class ContinualTrainer(Trainer):
    """ Trainer class for continual learning. """

    def init_model(self, config: Dict[str, Any]) -> None:
        """
        Initialize model and corresponding objects. The expected entries of `config` are
        listed below. `config` should contain all entries listed in the docstring of
        Trainer, as well as the settings specific to ContinualTrainer, which are listed
        below.

        Parameters
        ----------
        dataset : str
            Name of dataset to train on. The corresponding dataset should be a subclass
            of ContinualDataset.
        dataset_kwargs : Dict[str, Any]
            Keyword arguments to be passed to Dataset upon construction.
        batch_size : int
            Size of each minibatch on which to compute gradient updates.
        num_workers : int
            Number of worker processes to load data.
        """

        # Make sure that no learning rate schedule is used. This is currently not
        # supported, since the current version of learning rate scheduling uses the
        # entire period of `num_updates` to compute the learning rate.
        assert config["lr_schedule_type"] is None

        # Check dataset for validity.
        dataset_name = config["dataset"]
        if dataset_name not in DATASETS:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        dataset_cls = eval(dataset_name)
        if not issubclass(dataset_cls, ContinualDataset):
            raise ValueError(
                f"Dataset {dataset_name} is not a subclass of ContinualDataset."
            )

        # Construct train and test datasets and set current task to 0.
        root = os.path.join(DATA_DIR, dataset_name)
        dataset_kwargs = config["dataset_kwargs"]
        self.train_set = dataset_cls(root=root, train=True, **dataset_kwargs)
        self.test_set = dataset_cls(root=root, train=False, **dataset_kwargs)

        # Construct data loaders.
        self.batch_size = config["batch_size"]
        self.train_loader = torch.utils.data.DataLoader(
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
        self.train_iter = iter(cycle(self.train_loader))

        # Set length of window for moving average of metrics for training and
        # evaluation. For both training and evaluation, we average over the last epoch.
        # Since each evaluation step iterates over the entire epoch, we only need to
        # look at the most recent metric values to get the metrics for the last epoch.
        # TODO: Make sure we aren't mixing metrics between training of different tasks.
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
        elif config["architecture_config"]["type"] == "mlp":
            assert isinstance(input_size, int)
            network_cls = MLPNetwork
        elif config["architecture_config"]["type"] == "trunk":
            assert isinstance(input_size, int)
            network_cls = MultiTaskTrunkNetwork
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
        self.criterion = loss_cls(**loss_kwargs)
        self.criterion = self.criterion.to(self.device)
        self.criterion_kwargs = dict(self.train_set.criterion_kwargs)

        # Increase self.num_updates by a factor of dataset.num_tasks, so that each task
        # is trained for `self.num_updates` steps.
        self.updates_per_task = int(self.num_updates)
        self.num_updates *= self.train_set.num_tasks

        # Storage for BatchNorm parameters after training on each task. This is
        # temporary, in order to test out the limitations of batch normalization in
        # continual learning.
        self.task_bn_moments = []

    def _step(self) -> Dict[str, Any]:
        """ Perform one training step. """

        self.network.train()

        # Check if task index needs to be switched.
        if (self.steps % self.updates_per_task) == 0:
            current_task = int(self.steps // self.updates_per_task)
            self.train_set.advance_task()
            self.test_set.advance_task()
            self.train_iter = iter(cycle(self.train_loader))

        # Sample a batch and move it to device.
        inputs, labels = next(self.train_iter)
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        # Zero gradients.
        self.optimizer.zero_grad()

        # Perform forward pass and compute loss.
        outputs = self.network(inputs)
        loss = self.criterion(outputs, labels, **self.criterion_kwargs["train"])

        # Perform backward pass, clip gradient, and take optimizer step.
        loss.backward()
        self.clip_grads()
        self.optimizer.step()

        # Compute metrics from training step.
        step_metrics = {
            "train_loss": [torch.sum(loss).item()],
        }
        with torch.no_grad():
            extra_metrics = self.train_set.compute_metrics(
                outputs, labels, self.criterion, train=True
            )
            step_metrics.update({key: [val] for key, val in extra_metrics.items()})

        # Check if BN moments should be saved (only on last step of the current task).
        if ((self.steps + 1) % self.updates_per_task) == 0:
            self.task_bn_moments.append({})
            for m_name, m in self.network.named_modules():
                if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                    buffer_names = ["running_mean", "running_var"]
                    for b_name, b in m.named_buffers():
                        if b_name not in buffer_names:
                            continue
                        full_name = f"{m_name}.{b_name}"
                        self.task_bn_moments[-1][full_name] = b.clone().detach()

        return step_metrics

    def evaluate(self) -> None:
        """ Evaluate current model. """

        self.network.eval()

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
        raise NotImplementedError

    def get_checkpoint(self) -> None:
        """ Return trainer state as checkpoint. """

        checkpoint = {}
        checkpoint["network"] = self.network
        checkpoint["config"] = self.config
        return checkpoint

    def close(self) -> None:
        """ Clean up the training process. """
        pass

    def parameters(self) -> Iterator[nn.parameter.Parameter]:
        """ Return parameters of model. """
        return self.network.parameters()

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

    def load_bn_params(self, task: int) -> None:
        """ Load stored BN params for task `task`. """
        self.network.load_state_dict(self.task_bn_moments[task], strict=False)

    def compute_global_bn_moments(self) -> None:
        """
        Reproduces experiments from https://openreview.net/forum?id=vwLLQ-HwqhZ. We
        compute the batch normalization moments across the data from all tasks, to
        minimize the so-called cross-task normalization effect.
        """

        # Re-initialize the running parameters of all BatchNorm layers and convert
        # running stats to cumulative average instead of exponential moving average.
        for m in self.network.modules():
            if isinstance(m, torch.nn.BatchNorm1d) or isinstance(m, torch.nn.BatchNorm2d):
                m.reset_running_stats()
                m.momentum = None
        self.network.train()

        # Forward pass all of the data from all tasks through the network, so that
        # all BN layers can cumulatively track the moments for all tasks.
        for task in range(self.train_set.num_tasks):
            self.train_set.set_current_task(task)
            for inputs, labels in iter(self.train_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.network(inputs)
