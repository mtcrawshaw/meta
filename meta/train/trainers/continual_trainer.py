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
from meta.train.loss import PartialCrossEntropyLoss
from meta.train.optimize import SGDG, AdamG, unit, get_PSI_optimizer
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
        continual_bn : str
            Specification of protocol for handling batch normalization over tasks. One
            of ["none", "global", "separate", "new", "frozen"].
        memory_size : int
            Number of data samples from each task to store in memory. Memories are used
            to alleviate catastrophic forgetting with A-GEM.
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

        # Add current task to arguments to loss function, if necessary.
        if isinstance(self.criterion, PartialCrossEntropyLoss):
            self.criterion_kwargs["train"]["current_task"] = 0
            self.criterion_kwargs["eval"]["current_task"] = 0

        # Increase self.num_updates by a factor of dataset.num_tasks, so that each task
        # is trained for `self.num_updates` steps.
        self.updates_per_task = int(self.num_updates)
        self.num_updates *= self.train_set.num_tasks

        # Storage for BatchNorm parameters after training on each task. This is
        # temporary, in order to test out the limitations of batch normalization in
        # continual learning.
        assert config["continual_bn"] in ["none", "global", "separate", "new", "frozen"]
        self.continual_bn = config["continual_bn"]
        self.task_bn_state = []

        # Initialize episodic memory for A-GEM.
        if (
            "memory_size" in config
            and config["memory_size"] is not None
            and config["memory_size"] > 0
        ):
            self.agem = True
            self.memory_size = config["memory_size"]
            self.input_memory = None
            self.label_memory = None
            self.current_input_memory = None
            self.current_label_memory = None

            # This is the number of memories that are sampled from each minibatch during
            # training until the memory for the current task is filled. We set this
            # value so that we sample a (nearly) equal number of samples from each
            # minibatch during the first epoch.
            self.memories_per_batch = ceil(
                self.memory_size / ceil(len(self.train_set) / self.batch_size)
            )
        else:
            self.agem = False

        # Check for compatibility of options.
        if self.continual_bn == "frozen":
            assert not self.agem

    def _step(self) -> Dict[str, Any]:
        """ Perform one training step. """

        self.network.train()

        # Check if task index needs to be switched.
        if self.steps > 0 and (self.steps % self.updates_per_task) == 0:
            current_task = int(self.steps // self.updates_per_task)
            self.set_current_task(current_task)

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
        self._compute_grad(loss)
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

        # Check if BN state should be saved (only on last step of the current task).
        if ((self.steps + 1) % self.updates_per_task) == 0:
            self.task_bn_state.append({})
            for m_name, m in self.network.named_modules():
                if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                    bn_state = deepcopy(m.state_dict())
                    for b_name, b in bn_state.items():
                        full_name = f"{m_name}.{b_name}"
                        self.task_bn_state[-1][full_name] = b

        # Add samples from batch into episodic memory for current task, if necessary.
        if self.agem and (
            self.current_input_memory is None
            or len(self.current_input_memory) < self.memory_size
        ):
            already_sampled = (
                0
                if self.current_input_memory is None
                else len(self.current_input_memory)
            )
            num_samples = min(
                self.memories_per_batch, self.memory_size - already_sampled
            )
            sample_idxs = torch.randint(len(inputs), (num_samples,))
            input_samples = inputs[sample_idxs]
            label_samples = labels[sample_idxs]

            if self.current_input_memory is None:
                self.current_input_memory = input_samples
                self.current_label_memory = label_samples
            else:
                self.current_input_memory = torch.cat(
                    [self.current_input_memory, input_samples], dim=0
                )
                self.current_label_memory = torch.cat(
                    [self.current_label_memory, label_samples], dim=0
                )

        # Add memory from current task to total episodic memory, if this is the last
        # training step for the current task.
        if self.agem and ((self.steps + 1) % self.updates_per_task) == 0:
            if self.input_memory is None:
                self.input_memory = self.current_input_memory
                self.label_memory = self.current_label_memory
            else:
                self.input_memory = torch.cat(
                    [self.input_memory, self.current_input_memory], dim=0
                )
                self.label_memory = torch.cat(
                    [self.label_memory, self.current_label_memory], dim=0
                )
            self.current_input_memory = None
            self.current_label_memory = None

        return step_metrics

    def _compute_grad(self, loss: torch.Tensor) -> None:
        """
        Compute gradient of task parameters with respect to `loss`. This may be as
        simple as calling `backward()`, more work is required for options like A-GEM.
        """

        # Use A-GEM. Compute the gradient of the current task's loss, and compute the
        # gradient of the loss for a minibatch sampled from episodic memory, consisting
        # of data from previous tasks. If these two gradients are conflicting, project
        # the current task gradient onto the normal plane of the episodic gradient
        # before applying it to update network parameters. Note that we just perform a
        # regular backwards pass when the memory hasn't yet been populated.
        if self.agem and self.input_memory is not None:

            # Compute gradient of current task loss.
            loss.backward()
            task_grad = torch.cat([p.grad.view(-1) for p in self.network.parameters()])
            self.network.zero_grad()

            # Compute gradient of previous task losses by sampling episodic memory.
            memory_len = len(self.input_memory)
            memory_batch_size = min(self.batch_size, memory_len)
            sample_idxs = torch.randint(memory_len, (memory_batch_size,))
            memory_inputs = self.input_memory[sample_idxs]
            memory_labels = self.label_memory[sample_idxs]
            memory_outputs = self.network(memory_inputs)
            memory_loss = self.criterion(
                memory_outputs, memory_labels, **self.criterion_kwargs["train"]
            )
            memory_loss.backward()
            memory_grad = torch.cat(
                [p.grad.view(-1) for p in self.network.parameters()]
            )

            # Project current gradient and apply to parameters, if necessary.
            dot = torch.dot(task_grad, memory_grad)
            if dot < 0:
                task_grad -= dot / torch.sum(memory_grad ** 2) * memory_grad

                idx = 0
                for p in self.network.parameters():
                    grad_len = p.numel()
                    flattened_grad = task_grad[idx : idx + grad_len]
                    reshaped_grad = flattened_grad.reshape(p.shape)
                    p.grad = reshaped_grad
                    idx += grad_len

        # Regular case, just perform backwards pass.
        else:
            loss.backward()

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

    def set_current_task(self, new_task: int) -> None:
        """ Set current task for training. """

        self.train_set.set_current_task(new_task)
        self.test_set.set_current_task(new_task)
        self.train_iter = iter(cycle(self.train_loader))

        # Freeze non-BN parameters, if necessary. The "frozen" continual BN protocol
        # freezes the non-BN parameters after training on the first task finishes,
        # and only trains the BN parameters for the remainder of the continual
        # learning training. In addition, each task gets its own copy of the BN
        # parameters. Note: For ease of implementation, this currently only freezes
        # parameters of fully connected layers and convolutional layers.
        if self.continual_bn == "frozen" and new_task == 1:
            for m in self.network.modules():
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                    m.requires_grad_(False)
            self.init_optimizer()

        # Re-initialize batchnorm parameters, if necessary.
        if self.continual_bn == "new":
            for m in self.network.modules():
                if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                    m.reset_parameters()

        # Update arguments to loss function, if necessary.
        if "current_task" in self.criterion_kwargs["train"]:
            self.criterion_kwargs["train"]["current_task"] = new_task
            self.criterion_kwargs["eval"]["current_task"] = new_task

    def init_optimizer(self) -> None:
        """
        Initialize optimizer. Overridding this function from the base Trainer class
        allows us to use the Riemannian optimizers for continual learning.
        """

        # Set optimizer to Adam if none was set in config.
        if "optimizer" not in self.config:
            self.config["optimizer"] = "adam"

        # Construct optimizer according to config.
        if self.config["optimizer"] == "adam":
            self.optimizer = torch.optim.Adam(
                self.parameters(), lr=self.initial_lr, eps=self.eps
            )

        elif self.config["optimizer"] in ["sgdg", "adamg"]:

            # Check that network uses batch normalization.
            if not hasattr(self.network, "batch_norm") or not self.network.batch_norm:
                raise ValueError(
                    "Riemannian optimizers should only be used for networks that "
                    " use batch normalization."
                )

            # The Riemannian optimizer is only used on parameters in layers which
            # precede a batch normalization layer. Collecting these parameters is only
            # supported for certain architectures.
            pre_bn_param_names = []
            pre_bn_params = []
            if isinstance(self.network, ConvNetwork):

                for i, layer in enumerate(self.network.conv):
                    assert isinstance(layer[0], nn.Conv2d)
                    assert isinstance(layer[1], nn.BatchNorm2d)
                    for name, p in layer[0].named_parameters():
                        full_name = f"conv.{i}.0.{name}"
                        pre_bn_param_names.append(full_name)
                        pre_bn_params.append(p)
                for i, layer in enumerate(self.network.fc[:-1]):
                    assert isinstance(layer[0], nn.Linear)
                    assert isinstance(layer[1], nn.BatchNorm1d)
                    for name, p in layer[0].named_parameters():
                        full_name = f"fc.{i}.0.{name}"
                        pre_bn_param_names.append(full_name)
                        pre_bn_params.append(p)

            elif isinstance(self.network, MLPNetwork):

                for i, layer in enumerate(self.network.layers):
                    if i == len(self.network.layers) - 1:
                        continue
                    assert isinstance(layer[0], nn.Linear)
                    assert isinstance(layer[1], nn.BatchNorm1d)
                    for name, p in layer[0].named_parameters():
                        full_name = f"layers.{i}.0.{name}"
                        pre_bn_param_names.append(full_name)
                        pre_bn_params.append(p)

            else:
                raise NotImplementedError
            pre_bn_params = [p for p in pre_bn_params if p.requires_grad]
            other_params = []
            for name, p in self.network.named_parameters():
                if name not in pre_bn_param_names and p.requires_grad:
                    other_params.append(p)

            # Scale all of the pre-BatchNorm parameters to unit norm.
            for i, p in enumerate(pre_bn_params):
                unitp, _ = unit(p.data.view(p.shape[0], -1))
                pre_bn_params[i].data.copy_(unitp.view(p.size()))

            # Initialize optimizer.
            if self.config["optimizer"] == "sgdg":
                optimizer_cls = SGDG
            elif self.config["optimizer"] == "adamg":
                optimizer_cls = AdamG
            else:
                raise NotImplementedError
            self.optimizer = optimizer_cls(
                [
                    {
                        "params": pre_bn_params,
                        "lr": self.initial_lr,
                        "momentum": 0.9,
                        "eps": self.eps,
                        "grassmann": True,
                        "grad_clip": 0.1,
                    },
                    {
                        "params": other_params,
                        "lr": self.initial_lr,
                        "momentum": 0.9,
                        "eps": self.eps,
                        "grassmann": False,
                    },
                ]
            )

        elif self.config["optimizer"] == "psi_sgd":
            momentum = config["momentum"] if "momentum" in "config" else 0.0
            self.optimizer = get_PSI_optimizer(self.network, self.initial_lr, momentum)
        else:
            raise NotImplementedError

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
        return [p for p in self.network.parameters() if p.requires_grad]

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
        self.network.load_state_dict(self.task_bn_state[task], strict=False)

    def compute_global_bn_moments(self) -> None:
        """
        Reproduces experiments from https://openreview.net/forum?id=vwLLQ-HwqhZ. We
        compute the batch normalization moments across the data from all tasks, to
        minimize the so-called cross-task normalization effect.
        """

        # Re-initialize the running parameters of all BatchNorm layers and convert
        # running stats to cumulative average instead of exponential moving average.
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
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
