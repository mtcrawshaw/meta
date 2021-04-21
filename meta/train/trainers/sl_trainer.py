""" Definition of SLTrainer class for supervised learning. """

import os
from typing import Dict, Iterator, Iterable, Any

import torch
import torchvision
import torchvision.transforms as transforms

from meta.train.trainers.base_trainer import Trainer
from meta.networks import ConvNetwork
from meta.networks.utils import get_fc_layer, init_base
from meta.utils.utils import aligned_train_configs, DATA_DIR


DATASET_SIZES = {
    "MNIST": {"input_size": (28, 28, 1), "output_size": 10},
    "CIFAR10": {"input_size": (32, 32, 3), "output_size": 10},
    "CIFAR100": {"input_size": (32, 32, 3), "output_size": 100},
}
SUPPORTED_DATASETS = list(DATASET_SIZES.keys())
PRETRAINED_MODELS = ["resnet18", "resnet34", "resnet50", "resnet101"]


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

        # Get input/output size of dataset.
        if config["dataset"] not in SUPPORTED_DATASETS:
            raise NotImplementedError
        input_size = DATASET_SIZES[config["dataset"]]["input_size"]
        output_size = DATASET_SIZES[config["dataset"]]["output_size"]

        # Construct data loaders.
        mean = [0.5] * input_size[2]
        std = [0.5] * input_size[2]
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std),]
        )
        dataset = eval("torchvision.datasets.%s" % config["dataset"])
        train_set = dataset(
            root=os.path.join(DATA_DIR, config["dataset"]),
            train=True,
            download=True,
            transform=transform,
        )
        test_set = dataset(
            root=os.path.join(DATA_DIR, config["dataset"]),
            train=False,
            download=True,
            transform=transform,
        )
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
        )
        self.train_iter = iter(cycle(train_loader))
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
        )
        self.test_iter = iter(cycle(test_loader))

        # Construct network, either from a pre-trained model or from scratch.
        if config["architecture_config"]["type"] in PRETRAINED_MODELS:

            # Get pre-trained model and re-initialize the final fully-connected layer.
            pretrained = config["architecture_config"]["pretrained"]
            model = eval(
                "torchvision.models.%s" % config["architecture_config"]["type"]
            )
            self.network = model(pretrained=pretrained)
            num_features = self.network.fc.in_features
            self.network.fc = get_fc_layer(
                in_size=num_features,
                out_size=output_size,
                activation=None,
                layer_init=init_base,
            )
            self.network.to(self.device)

        else:

            # Construct network from scratch.
            network_kwargs = dict(config["architecture_config"])
            del network_kwargs["type"]
            network_kwargs["input_size"] = input_size
            network_kwargs["output_size"] = output_size
            network_kwargs["device"] = self.device
            self.network = ConvNetwork(**network_kwargs)

        # Construct loss function.
        self.criterion = torch.nn.CrossEntropyLoss()

    def _step(self) -> Dict[str, Any]:
        """ Perform one training step. """

        # Sample a batch and move it to device.
        inputs, labels = next(self.train_iter)
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        # Perform forward pass and compute loss.
        outputs = self.network(inputs)
        loss = self.criterion(outputs, labels)
        accuracy = (
            torch.sum(torch.argmax(outputs, dim=-1) == labels)
            / self.config["batch_size"]
        )

        # Perform backward pass, clip gradient, and take optimizer step.
        self.network.zero_grad()
        loss.backward()
        self.clip_grads()
        self.optimizer.step()

        # Return metrics from training step.
        step_metrics = {
            "train_loss": [loss.item()],
            "train_accuracy": [accuracy.item()],
        }
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
        accuracy = (
            torch.sum(torch.argmax(outputs, dim=-1) == labels)
            / self.config["batch_size"]
        )

        # Return metrics from training step.
        eval_step_metrics = {
            "test_loss": [loss.item()],
            "test_accuracy": [accuracy.item()],
        }
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
