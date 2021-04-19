""" Definition of SLTrainer class for supervised learning. """

from typing import Dict, Any

import torch
import torchvision
import torchvision.transforms as transforms

from meta.train.trainers.base_trainer import Trainer
from meta.networks import ConvNetwork
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
        num_epochs : int
            Number of training epochs.
        batch_size : int
            Size of each minibatch on which to compute gradient updates.
        num_workers : int
            Number of worker processes to load data.
        """

        # Get input/output size of dataset.
        if config["dataset"] not in DATASET_SIZES:
            raise NotImplementedError
        input_size = DATASET_SIZES["dataset"]["input_size"]
        output_size = DATASET_SIZES["dataset"]["output_size"]

        # Construct data loaders.
        transform = transforms.Compose(
            [
                transforms.toTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = eval("torchvision.datasets.%s" % config["dataset"])
        train_set = dataset(
            root=os.path.join(DATA_DIR, config["dataset"]),
            train=True,
            download=True,
            transform=True,
        )
        test_set = dataset(
            root=os.path.join(DATA_DIR, config["dataset"]),
            train=False,
            download=True,
            transform=True,
        )
        self.train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
        )

        # Construct network.
        network_kwargs = dict(config["architecture_config"])
        del network_kwargs["type"]
        network_kwargs["input_size"] = input_size
        network_kwargs["output_size"] = output_size
        network_kwargs["device"] = self.device
        self.network = ConvNetwork(**network_kwargs)

    def _step(self) -> Dict[str, Any]:
        """ Perform one training step. """

        # Sample a batch.
        pass

        # Perform forward pass.
        pass

        # Compute loss.
        loss = None

        # Perform backward pass, clip gradient, and take optimizer step.
        self.network.zero_grad()
        loss.backward()
        self.clip_grads()
        self.optimizer.step()

        # Return metrics from training step.
        step_metrics = {"train_loss": loss.item()}
        return step_metrics

    def evaluate(self) -> None:
        """ Evaluate current model. """

        # Sample a batch.
        pass

        # Perform forward pass.
        pass

        # Compute loss.
        loss = None

        # Return metrics from training step.
        eval_step_metrics = {
            "eval_loss": loss.item(),
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


DATASET_SIZES = {
    "MNIST": {"input_size": (28, 28, 1), "output_size": 10},
    "CIFAR": {"input_size": (32, 32, 3), "output_size": 10},
}
