""" Definition of SLTrainer class for supervised learning. """

import os
from typing import Dict, Iterator, Iterable, Any, List, Tuple

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from meta.train.trainers.base_trainer import Trainer
from meta.train.datasets import NYUv2
from meta.train.loss import CosineSimilarityLoss, MultiTaskLoss, get_accuracy
from meta.networks import ConvNetwork, BackboneNetwork, PRETRAINED_MODELS
from meta.utils.utils import aligned_train_configs, DATA_DIR


DEPTH_MEAN = 2.5
DEPTH_STD = 1
GRAY_TRANSFORM = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
)
RGB_TRANSFORM = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
)
DEPTH_TRANSFORM = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(DEPTH_MEAN, DEPTH_STD)]
)
SEG_TRANSFORM = transforms.ToTensor()
SN_TRANSFORM = transforms.ToTensor()

DATASETS = {
    "MNIST": {
        "input_size": (1, 28, 28),
        "output_size": 10,
        "builtin": True,
        "loss_cls": nn.CrossEntropyLoss,
        "loss_kwargs": {},
        "extra_metrics": {"accuracy": {"fn": get_accuracy, "maximize": True}},
        "base_name": "MNIST",
        "dataset_kwargs": {"transform": GRAY_TRANSFORM},
    },
    "CIFAR10": {
        "input_size": (3, 32, 32),
        "output_size": 10,
        "builtin": True,
        "loss_cls": nn.CrossEntropyLoss,
        "loss_kwargs": {},
        "extra_metrics": {"accuracy": {"fn": get_accuracy, "maximize": True}},
        "base_name": "CIFAR10",
        "dataset_kwargs": {"transform": RGB_TRANSFORM},
    },
    "CIFAR100": {
        "input_size": (3, 32, 32),
        "output_size": 100,
        "builtin": True,
        "loss_cls": nn.CrossEntropyLoss,
        "loss_kwargs": {},
        "extra_metrics": {"accuracy": {"fn": get_accuracy, "maximize": True}},
        "base_name": "CIFAR100",
        "dataset_kwargs": {"transform": RGB_TRANSFORM},
    },
    "NYUv2_seg": {
        "input_size": (3, 480, 64),
        "output_size": (13, 480, 64),
        "builtin": False,
        "loss_cls": nn.CrossEntropyLoss,
        "loss_kwargs": {"ignore_index": -1},
        "extra_metrics": {"accuracy": {"fn": None, "maximize": True}},
        "base_name": "NYUv2",
        "dataset_kwargs": {
            "rgb_transform": RGB_TRANSFORM,
            "seg_transform": SEG_TRANSFORM,
            "scale": 0.25,
        },
    },
    "NYUv2_sn": {
        "input_size": (3, 480, 64),
        "output_size": (3, 480, 64),
        "builtin": False,
        "loss_cls": CosineSimilarityLoss,
        "loss_kwargs": {},
        "extra_metrics": {"accuracy": {"fn": None, "maximize": True}},
        "base_name": "NYUv2",
        "dataset_kwargs": {
            "rgb_transform": RGB_TRANSFORM,
            "sn_transform": SN_TRANSFORM,
            "scale": 0.25,
        },
    },
    "NYUv2_depth": {
        "input_size": (3, 480, 64),
        "output_size": (1, 480, 64),
        "builtin": False,
        "loss_cls": nn.MSELoss,
        "loss_kwargs": {},
        "extra_metrics": {"accuracy": {"fn": None, "maximize": True}},
        "base_name": "NYUv2",
        "dataset_kwargs": {
            "rgb_transform": RGB_TRANSFORM,
            "depth_transform": DEPTH_TRANSFORM,
            "scale": 0.25,
        },
    },
    "NYUv2": {
        "input_size": (3, 480, 64),
        "output_size": [(13, 480, 64), (3, 480, 64), (1, 480, 64)],
        "builtin": False,
        "loss_cls": MultiTaskLoss,
        "loss_kwargs": {
            "task_losses": [
                {
                    "loss": nn.CrossEntropyLoss(ignore_index=-1),
                    "output_slice": lambda x: x[:, :13],
                    "label_slice": lambda x: x[:, 0].long(),
                },
                {
                    "loss": CosineSimilarityLoss(),
                    "output_slice": lambda x: x[:, 13:16],
                    "label_slice": lambda x: x[:, 1:4],
                },
                {
                    "loss": nn.MSELoss(),
                    "output_slice": lambda x: x[:, 16:17],
                    "label_slice": lambda x: x[:, 4:5],
                },
            ],
        },
        "extra_metrics": {
            "seg_accuracy": {"fn": None, "maximize": True},
            "sn_accuracy": {"fn": None, "maximize": True},
            "depth_accuracy": {"fn": None, "maximize": True},
            "avg_accuracy": {"fn": None, "maximize": True},
        },
        "base_name": "NYUv2",
        "dataset_kwargs": {
            "rgb_transform": RGB_TRANSFORM,
            "seg_transform": SEG_TRANSFORM,
            "sn_transform": SN_TRANSFORM,
            "depth_transform": DEPTH_TRANSFORM,
            "scale": 0.25,
        },
    },
}


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

        # Get dataset info.
        if config["dataset"] not in DATASETS:
            raise NotImplementedError
        self.dataset = config["dataset"]
        self.dataset_info = DATASETS[self.dataset]
        input_size = self.dataset_info["input_size"]
        output_size = self.dataset_info["output_size"]
        self.extra_metrics = self.dataset_info["extra_metrics"]

        # Construct data loaders.
        if self.dataset_info["builtin"]:
            dataset = eval("torchvision.datasets.%s" % self.dataset)
        else:
            dataset = eval(self.dataset_info["base_name"])
        root = os.path.join(DATA_DIR, self.dataset_info["base_name"])
        dataset_kwargs = self.dataset_info["dataset_kwargs"]
        train_set = dataset(root=root, train=True, download=True, **dataset_kwargs)
        test_set = dataset(root=root, train=False, download=True, **dataset_kwargs)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
        )
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
        )
        self.train_iter = iter(cycle(train_loader))
        self.test_iter = iter(cycle(test_loader))

        # Construct network, either from a pre-trained model or from scratch.
        network_kwargs = dict(config["architecture_config"])
        if config["architecture_config"]["type"] in PRETRAINED_MODELS:
            network_cls = BackboneNetwork
            network_kwargs["arch_type"] = network_kwargs["type"]
        else:
            network_cls = ConvNetwork
        del network_kwargs["type"]
        network_kwargs["input_size"] = input_size
        network_kwargs["output_size"] = output_size
        network_kwargs["device"] = self.device
        self.network = network_cls(**network_kwargs)

        # Construct loss function.
        loss_cls = self.dataset_info["loss_cls"]
        loss_kwargs = self.dataset_info["loss_kwargs"]
        if "loss_weighter" in config:
            loss_kwargs["loss_weighter_kwargs"] = dict(config["loss_weighter"])
        self.criterion = loss_cls(**loss_kwargs)

    def _step(self) -> Dict[str, Any]:
        """ Perform one training step. """

        # Sample a batch and move it to device.
        inputs, labels = next(self.train_iter)
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        # Perform forward pass and compute loss.
        outputs = self.network(inputs)
        loss = self.criterion(outputs, labels)

        # Perform backward pass, clip gradient, and take optimizer step.
        self.network.zero_grad()
        loss.backward()
        self.clip_grads()
        self.optimizer.step()

        # Compute metrics from training step.
        step_metrics = {
            "train_loss": [loss.item()],
        }
        for metric_name, metric_info in self.extra_metrics.items():
            full_name = "train_%s" % metric_name
            fn = metric_info["fn"]
            step_metrics[full_name] = [fn(outputs, labels)]


        return step_metrics

    def evaluate(self) -> None:
        """ Evaluate current model. """

        # Sample a batch and move it to device.
        inputs, labels = next(self.test_iter)
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        # Perform forward pass and copmute loss.
        outputs = self.network(inputs)
        loss = self.criterion(outputs, labels)

        # Compute metrics from evaluation step.
        eval_step_metrics = {
            "eval_loss": [loss.item()],
        }
        for metric_name, metric_info in self.extra_metrics.items():
            full_name = "eval_%s" % metric_name
            fn = metric_info["fn"]
            eval_step_metrics[full_name] = [fn(outputs, labels)]

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

    def close(self) -> None:
        """ Clean up the training process. """
        pass

    def parameters(self) -> Iterator[nn.parameter.Parameter]:
        """ Return parameters of model. """
        return self.network.parameters()

    @property
    def metric_set(self) -> List[Tuple]:
        """ Set of metrics for this trainer. """

        window = 100
        metric_set = [
            ("train_loss", window, False, False),
            ("eval_loss", window, False, False),
        ]
        for metric_name, metric_info in self.extra_metrics.items():
            metric_set.append(
                ("train_%s" % metric_name, window, False, metric_info["maximize"])
            )
            metric_set.append(
                ("eval_%s" % metric_name, window, False, metric_info["maximize"])
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
