""" Definition of SLTrainer class. """

from typing import Dict, Any

import torch

from meta.train.trainers.base_trainer import Trainer
from meta.utils.utils import aligned_train_configs


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
        minibatch_size : int
            Size of each minibatch on which to compute gradient updates.
        """

        # Construct network and data loader.
        self.network = None

    def _step(self) -> Dict[str, Any]:
        """ Perform one training step. """

        # Sample a batch.

        # Perform forward pass.

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

        # Perform forward pass.

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
