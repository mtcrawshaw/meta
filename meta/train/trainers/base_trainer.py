""" Definition of Trainer base class. """

import math
from typing import Dict, Any

import numpy as np
import torch


class Trainer:
    """ Abstract base class for trainers. """

    def __init__(self, config: Dict[str, Any], **kwargs: Dict[str, Any]) -> None:
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

        # Initialize model.
        self.init_model(config, **kwargs)

        # Initialize optimizer.
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=config["initial_lr"], eps=config["eps"]
        )

        # Initialize learning rate schedule.
        if config["lr_schedule_type"] == "exponential":
            total_lr_decay = config["final_lr"] / config["initial_lr"]
            decay_per_epoch = math.pow(total_lr_decay, 1.0 / config["num_updates"])
            self.lr_schedule = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=self.optimizer, gamma=decay_per_epoch,
            )

        elif config["lr_schedule_type"] == "cosine":
            self.lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=self.optimizer,
                T_max=config["num_updates"],
                eta_min=config["final_lr"],
            )

        elif config["lr_schedule_type"] == "linear":

            def factor(step: int) -> float:
                lr_shift = config["final_lr"] - config["initial_lr"]
                desired_lr = config["initial_lr"] + lr_shift * float(step) / (
                    config["num_updates"] - 1
                )
                return desired_lr / config["initial_lr"]

            self.lr_schedule = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.optimizer, lr_lambda=factor,
            )

        elif config["lr_schedule_type"] is None:
            self.lr_schedule = None

        else:
            raise ValueError(
                "Unrecognized lr scheduler type: %s" % config["lr_schedule_type"]
            )

    def init_model(self) -> None:
        """ Initialize model and corresponding objects. """
        raise NotImplementedError

    def step(self) -> Dict[str, Any]:
        """ Perform a training step. """

        # Perform training step.
        step_metrics = self._step()

        # Step learning rate schedule, if necessary.
        if self.lr_schedule is not None:
            self.lr_schedule.step()

        return step_metrics

    def _step(self) -> Dict[str, Any]:
        """ Perform a training step specific to the trainer type. """
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
