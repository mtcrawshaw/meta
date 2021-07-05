""" Definition of SLTrainer class for supervised learning. """

import os
from itertools import chain
from typing import Dict, Iterator, Iterable, Any, List, Tuple, Callable

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from meta.train.trainers.base_trainer import Trainer
from meta.datasets import NYUv2, MTRegression
from meta.train.loss import (
    CosineSimilarityLoss,
    MultiTaskLoss,
    Uncertainty,
    get_accuracy,
    NYUv2_seg_accuracy,
    NYUv2_sn_accuracy,
    NYUv2_depth_accuracy,
    NYUv2_multi_seg_accuracy,
    NYUv2_multi_sn_accuracy,
    NYUv2_multi_depth_accuracy,
    NYUv2_multi_avg_accuracy,
    get_MTRegression_normal_loss,
    get_multitask_loss_weight,
)
from meta.networks import (
    ConvNetwork,
    BackboneNetwork,
    MLPNetwork,
    MultiTaskTrunkNetwork,
    PRETRAINED_MODELS,
)
from meta.networks import last_shared_params
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


def slice_second_dim(idx: int) -> Callable[[Any], Any]:
    """
    Utility function to generate slice functions for MTRegression task losses. Just for
    convenience so we don't have to hard-code 10 functions of the form generated below.
    """

    def slice(x: Any) -> Any:
        return x[:, idx]

    return slice


DATASETS = {
    "MNIST": {
        "input_size": (1, 28, 28),
        "output_size": 10,
        "builtin": True,
        "loss_cls": nn.CrossEntropyLoss,
        "loss_kwargs": {},
        "criterion_kwargs": {"train": {}, "eval": {}},
        "extra_metrics": {
            "train_accuracy": {
                "fn": get_accuracy,
                "basename": "accuracy",
                "maximize": True,
                "train": True,
                "show": True,
            },
            "eval_accuracy": {
                "fn": get_accuracy,
                "basename": "accuracy",
                "maximize": True,
                "train": False,
                "show": True,
            },
        },
        "base_name": "MNIST",
        "dataset_kwargs": {"download": True, "transform": GRAY_TRANSFORM},
    },
    "CIFAR10": {
        "input_size": (3, 32, 32),
        "output_size": 10,
        "builtin": True,
        "loss_cls": nn.CrossEntropyLoss,
        "loss_kwargs": {},
        "criterion_kwargs": {"train": {}, "eval": {}},
        "extra_metrics": {
            "train_accuracy": {
                "fn": get_accuracy,
                "basename": "accuracy",
                "maximize": True,
                "train": True,
                "show": True,
            },
            "eval_accuracy": {
                "fn": get_accuracy,
                "basename": "accuracy",
                "maximize": True,
                "train": False,
                "show": True,
            },
        },
        "base_name": "CIFAR10",
        "dataset_kwargs": {"download": True, "transform": RGB_TRANSFORM},
    },
    "CIFAR100": {
        "input_size": (3, 32, 32),
        "output_size": 100,
        "builtin": True,
        "loss_cls": nn.CrossEntropyLoss,
        "loss_kwargs": {},
        "criterion_kwargs": {"train": {}, "eval": {}},
        "extra_metrics": {
            "train_accuracy": {
                "fn": get_accuracy,
                "basename": "accuracy",
                "maximize": True,
                "train": True,
                "show": True,
            },
            "eval_accuracy": {
                "fn": get_accuracy,
                "basename": "accuracy",
                "maximize": True,
                "train": False,
                "show": True,
            },
        },
        "base_name": "CIFAR100",
        "dataset_kwargs": {"download": True, "transform": RGB_TRANSFORM},
    },
    "NYUv2_seg": {
        "input_size": (3, 480, 64),
        "output_size": (13, 480, 64),
        "builtin": False,
        "loss_cls": nn.CrossEntropyLoss,
        "loss_kwargs": {"ignore_index": -1},
        "criterion_kwargs": {"train": {}, "eval": {}},
        "extra_metrics": {
            "train_accuracy": {
                "fn": NYUv2_seg_accuracy,
                "basename": "accuracy",
                "maximize": True,
                "train": True,
                "show": True,
            },
            "eval_accuracy": {
                "fn": NYUv2_seg_accuracy,
                "basename": "accuracy",
                "maximize": True,
                "train": False,
                "show": True,
            },
        },
        "base_name": "NYUv2",
        "dataset_kwargs": {
            "download": True,
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
        "criterion_kwargs": {"train": {}, "eval": {}},
        "extra_metrics": {
            "train_accuracy": {
                "fn": NYUv2_sn_accuracy,
                "basename": "accuracy",
                "maximize": True,
                "train": True,
                "show": True,
            },
            "eval_accuracy": {
                "fn": NYUv2_sn_accuracy,
                "basename": "accuracy",
                "maximize": True,
                "train": False,
                "show": True,
            },
        },
        "base_name": "NYUv2",
        "dataset_kwargs": {
            "download": True,
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
        "criterion_kwargs": {"train": {}, "eval": {}},
        "extra_metrics": {
            "train_accuracy": {
                "fn": NYUv2_depth_accuracy,
                "basename": "accuracy",
                "maximize": True,
                "train": True,
                "show": True,
            },
            "eval_accuracy": {
                "fn": NYUv2_depth_accuracy,
                "basename": "accuracy",
                "maximize": True,
                "train": False,
                "show": True,
            },
        },
        "base_name": "NYUv2",
        "dataset_kwargs": {
            "download": True,
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
        "criterion_kwargs": {"train": {"train": True}, "eval": {"train": False}},
        "extra_metrics": {
            "train_seg_accuracy": {
                "fn": NYUv2_multi_seg_accuracy,
                "basename": "seg_accuracy",
                "maximize": True,
                "train": True,
                "show": False,
            },
            "eval_seg_accuracy": {
                "fn": NYUv2_multi_seg_accuracy,
                "basename": "seg_accuracy",
                "maximize": True,
                "train": False,
                "show": False,
            },
            "train_sn_accuracy": {
                "fn": NYUv2_multi_sn_accuracy,
                "basename": "sn_accuracy",
                "maximize": True,
                "train": True,
                "show": False,
            },
            "eval_sn_accuracy": {
                "fn": NYUv2_multi_sn_accuracy,
                "basename": "sn_accuracy",
                "maximize": True,
                "train": False,
                "show": False,
            },
            "train_depth_accuracy": {
                "fn": NYUv2_multi_depth_accuracy,
                "basename": "depth_accuracy",
                "maximize": True,
                "train": True,
                "show": False,
            },
            "eval_depth_accuracy": {
                "fn": NYUv2_multi_depth_accuracy,
                "basename": "depth_accuracy",
                "maximize": True,
                "train": False,
                "show": False,
            },
            "train_avg_accuracy": {
                "fn": NYUv2_multi_avg_accuracy,
                "basename": "avg_accuracy",
                "maximize": True,
                "train": True,
                "show": True,
            },
            "eval_avg_accuracy": {
                "fn": NYUv2_multi_avg_accuracy,
                "basename": "avg_accuracy",
                "maximize": True,
                "train": False,
                "show": True,
            },
            **{
                "loss_weight_%d"
                % i: {
                    "fn": get_multitask_loss_weight(i),
                    "basename": "loss_weight",
                    "maximize": None,
                    "train": True,
                    "show": False,
                }
                for i in range(3)
            },
        },
        "base_name": "NYUv2",
        "dataset_kwargs": {
            "download": True,
            "rgb_transform": RGB_TRANSFORM,
            "seg_transform": SEG_TRANSFORM,
            "sn_transform": SN_TRANSFORM,
            "depth_transform": DEPTH_TRANSFORM,
            "scale": 0.25,
        },
    },
    "MTRegression2": {
        "input_size": 250,
        "output_size": 100,
        "builtin": False,
        "loss_cls": MultiTaskLoss,
        "loss_kwargs": {
            "task_losses": [
                {
                    "loss": nn.MSELoss(),
                    "output_slice": slice_second_dim(i),
                    "label_slice": slice_second_dim(i),
                }
                for i in range(2)
            ],
        },
        "criterion_kwargs": {"train": {"train": True}, "eval": {"train": False}},
        "extra_metrics": {
            "train_normal_loss": {
                "fn": get_MTRegression_normal_loss(2),
                "basename": "normal_loss",
                "maximize": False,
                "train": True,
                "show": True,
            },
            "eval_normal_loss": {
                "fn": get_MTRegression_normal_loss(2),
                "basename": "normal_loss",
                "maximize": False,
                "train": False,
                "show": True,
            },
            **{
                "loss_weight_%d"
                % i: {
                    "fn": get_multitask_loss_weight(i),
                    "basename": "loss_weight",
                    "maximize": None,
                    "train": True,
                    "show": False,
                }
                for i in range(2)
            },
        },
        "base_name": "MTRegression",
        "dataset_kwargs": {"num_tasks": 2},
    },
    "MTRegression10": {
        "input_size": 250,
        "output_size": 100,
        "builtin": False,
        "loss_cls": MultiTaskLoss,
        "loss_kwargs": {
            "task_losses": [
                {
                    "loss": nn.MSELoss(),
                    "output_slice": slice_second_dim(i),
                    "label_slice": slice_second_dim(i),
                }
                for i in range(10)
            ],
        },
        "criterion_kwargs": {"train": {"train": True}, "eval": {"train": False}},
        "extra_metrics": {
            "train_normal_loss": {
                "fn": get_MTRegression_normal_loss(10),
                "basename": "normal_loss",
                "maximize": False,
                "train": True,
                "show": True,
            },
            "eval_normal_loss": {
                "fn": get_MTRegression_normal_loss(10),
                "basename": "normal_loss",
                "maximize": False,
                "train": False,
                "show": True,
            },
            **{
                "loss_weight_%d"
                % i: {
                    "fn": get_multitask_loss_weight(i),
                    "basename": "loss_weight",
                    "maximize": None,
                    "train": True,
                    "show": False,
                }
                for i in range(10)
            },
        },
        "base_name": "MTRegression",
        "dataset_kwargs": {"num_tasks": 10},
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
        train_set = dataset(root=root, train=True, **dataset_kwargs)
        test_set = dataset(root=root, train=False, **dataset_kwargs)
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

        # Determine type of network to construct.
        network_kwargs = dict(config["architecture_config"])
        input_size = self.dataset_info["input_size"]
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

        # Set up case for loss function.
        loss_cls = self.dataset_info["loss_cls"]
        loss_kwargs = self.dataset_info["loss_kwargs"]

        # Add arguments to `self.criterion` in case we are multi-task training. These
        # are passed as arguments to the constructor of `self.criterion`.
        if "loss_weighter" in config:
            loss_weighter_kwargs = dict(config["loss_weighter"])
            if config["loss_weighter"]["type"] == "GradNorm":
                loss_weighter_kwargs["shared_params"] = last_shared_params(self.network)
            loss_kwargs["loss_weighter_kwargs"] = loss_weighter_kwargs
        if loss_cls == MultiTaskLoss:
            loss_kwargs["device"] = self.device

        # Construct loss function.
        self.criterion = loss_cls(**loss_kwargs)

        # Construct arguments to `self.criterion`. These are passed as arguments to the
        # forward pass through `self.criterion`. Here we include the network itself as
        # an argument to the loss function, since computing the task-specific gradients
        # requires zero-ing out gradients between tasks, and this requires access to the
        # Module containing the relevant parameters. Note that this will need to change
        # in the case that GradNorm is operating over parameters outside of
        # `self.network`, or if the task-specific loss functions are dependent on
        # parameters outside of `self.network`.
        criterion_kwargs = self.dataset_info["criterion_kwargs"]
        if "loss_weighter" in config:
            if config["loss_weighter"]["type"] == "GradNorm":
                criterion_kwargs["train"]["network"] = self.network
        self.criterion_kwargs = dict(criterion_kwargs)

    def _step(self) -> Dict[str, Any]:
        """ Perform one training step. """

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
            "train_loss": [loss.item()],
        }
        for metric_name, metric_info in self.extra_metrics.items():
            if metric_info["train"]:
                fn = metric_info["fn"]
                step_metrics[metric_name] = [fn(outputs, labels, self.criterion)]

        return step_metrics

    def evaluate(self) -> None:
        """ Evaluate current model. """

        # Sample a batch and move it to device.
        inputs, labels = next(self.test_iter)
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        # Perform forward pass and copmute loss.
        outputs = self.network(inputs)
        loss = self.criterion(outputs, labels, **self.criterion_kwargs["eval"])

        # Compute metrics from evaluation step.
        eval_step_metrics = {
            "eval_loss": [loss.item()],
        }
        for metric_name, metric_info in self.extra_metrics.items():
            if not metric_info["train"]:
                fn = metric_info["fn"]
                eval_step_metrics[metric_name] = [fn(outputs, labels, self.criterion)]

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

        window = 100
        metric_set = [
            {
                "name": "train_loss",
                "basename": "loss",
                "window": window,
                "point_avg": False,
                "maximize": False,
                "show": True,
            },
            {
                "name": "eval_loss",
                "basename": "loss",
                "window": window,
                "point_avg": False,
                "maximize": False,
                "show": True,
            },
        ]
        metric_set += [
            {
                "name": metric_name,
                "basename": metric_info["basename"],
                "window": window,
                "point_avg": False,
                "maximize": metric_info["maximize"],
                "show": metric_info["show"],
            }
            for metric_name, metric_info in self.extra_metrics.items()
        ]
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
