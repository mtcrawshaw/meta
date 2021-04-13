""" Definition of Trainer base class. """

from typing import Dict, Any

import numpy as np
import torch


class Trainer:
    """ Abstract base class for trainers. """

    def __init__(self, config: Dict[str, Any]) -> None:
        """ Init function for Trainer. """

        self.config = config

        # Set random seed, number of threads, and device.
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.set_num_threads(1)
        if config["cuda"]:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device("cpu")
                print(
                    'Warning: config["cuda"] = True but torch.cuda.is_available() = '
                    "False. Using CPU for training."
                )
        else:
            self.device = torch.device("cpu")

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """ Perform one training step. """
        raise NotImplementedError

    def evaluate(self) -> None:
        """ Evaluate current model. """
        raise NotImplementedError

    def load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """ Load trainer state from checkpoint. """
        raise NotImplementedError

    def get_checkpoint(self) -> None:
        """ Return trainer state as checkpoint. """
        raise NotImplementedError

    def close(self) -> None:
        """ Clean up the training process. """
        raise NotImplementedError

    def parameters(self) -> None:
        """ Return parameters of model. """
        raise NotImplementedError

    def clip_grads(self) -> None:
        """ Clip gradients of model parameters, if necessary. """
        if self.config["max_grad_norm"] is not None:
            torch.nn.utils.clip_grad_norm_(
                self.parameters(), self.config["max_grad_norm"],
            )
