""" Definition of SLTrainer class for supervised learning. """

import os
from typing import Dict, Iterator, Iterable, Any

import torch
import torchvision
import torchvision.transforms as transforms

from meta.train.trainers.base_trainer import Trainer
from meta.train.datasets import NYUv2
from meta.networks import ConvNetwork, BackboneNetwork, PRETRAINED_MODELS
from meta.networks.utils import get_fc_layer, init_base, IntermediateLayerGetter
from meta.utils.utils import aligned_train_configs, DATA_DIR


DATASETS = {
    "MNIST": {
        "input_size": (1, 28, 28),
        "output_size": 10,
        "builtin": True,
        "loss": torch.nn.CrossEntropyLoss,
        "compute_accuracy": True,
        "base_name": "MNIST",
    },
    "CIFAR10": {
        "input_size": (3, 32, 32),
        "output_size": 10,
        "builtin": True,
        "loss": torch.nn.CrossEntropyLoss,
        "compute_accuracy": True,
        "base_name": "CIFAR10",
    },
    "CIFAR100": {
        "input_size": (3, 32, 32),
        "output_size": 100,
        "builtin": True,
        "loss": torch.nn.CrossEntropyLoss,
        "compute_accuracy": True,
        "base_name": "CIFAR100",
    },
    "NYUv2_depth": {
        "input_size": (3, 480, 64),
        "output_size": (1, 480, 64),
        "builtin": False,
        "loss": torch.nn.MSELoss,
        "compute_accuracy": False,
        "base_name": "NYUv2",
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
        self.compute_accuracy = self.dataset_info["compute_accuracy"]

        # Data transformation.
        mean = [0.5] * input_size[0]
        std = [0.5] * input_size[0]
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
        if self.dataset == "NYUv2_depth":
            depth_mean = 2.5
            depth_std = 1
            depth_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(depth_mean, depth_std)]
                # [transforms.ToTensor()]
            )
            kwargs = {"rgb_transform": transform, "depth_transform": depth_transform}
        else:
            kwargs = {"transform": transform}

        # Construct data loaders.
        if self.dataset_info["builtin"]:
            dataset = eval("torchvision.datasets.%s" % self.dataset)
        else:
            dataset = eval(self.dataset_info["base_name"])
        root = os.path.join(DATA_DIR, self.dataset_info["base_name"])
        train_set = dataset(root=root, train=True, download=True, **kwargs,)
        test_set = dataset(root=root, train=False, download=True, **kwargs,)
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
        self.criterion = self.dataset_info["loss"]()

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
        if self.compute_accuracy:
            accuracy = (
                torch.sum(torch.argmax(outputs, dim=-1) == labels)
                / self.config["batch_size"]
            )
            step_metrics["train_accuracy"] = [accuracy.item()]

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
            "test_loss": [loss.item()],
        }
        if self.compute_accuracy:
            accuracy = (
                torch.sum(torch.argmax(outputs, dim=-1) == labels)
                / self.config["batch_size"]
            )
            eval_step_metrics["test_accuracy"] = [accuracy.item()]

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

    def parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
        """ Return parameters of model. """
        return self.network.parameters()


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
