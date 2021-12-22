""" Definition of Trainer base class. """

import math
from typing import Dict, Any

import numpy as np
import torch

from meta.train.optimize import SGDG, AdamG, get_grassmann_optimizer, get_PSI_optimizer


class Trainer:
    """ Abstract base class for trainers. """

    def __init__(self, config: Dict[str, Any], **kwargs: Dict[str, Any]) -> None:
        """
        Init function for Trainer. The expected entries of `config` are listed below,
        though the concrete extensions of this class require more entries which are
        listed in their docstrings.

        Parameters
        ----------
        num_updates : int
            Number of training steps.
        lr_schedule_type : str
            Either None, "exponential", "cosine", or "linear". If None is given, the
            learning rate will stay at initial_lr for the duration of training.
        initial_lr : float
            Initial policy learning rate.
        final_lr : float
            Final policy learning rate.
        max_grad_norm : float
            Max norm of gradients
        eps : float
            Epsilon value for numerical stability.
        architecture_config: Dict[str, Any]
            Config dictionary for the architecture. Should contain an entry for "type",
            which is either "vanilla", "conv", "trunk", "splitting_v1" or
            "splitting_v2", and all other entries should correspond to the keyword
            arguments for the corresponding network class, which is either
            VanillaNetwork, ConvNetwork, MultiTaskTrunkNetwork,
            MultiTaskSplittingNetworkV1, or MultiTaskSplittingNetworkV2. This can also
            be None in the case that `policy` is not None.
        cuda : bool
            Whether or not to train on GPU.
        seed : int
            Random seed.
        """

        self.config = config

        # Set random seed, number of threads, and device.
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
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
        self.num_updates = config["num_updates"]
        self.init_model(config, **kwargs)

        # Initialize optimizer.
        self.initial_lr = config["initial_lr"]
        self.momentum = config["momentum"] if "momentum" in config else None
        self.max_grad_norm = config["max_grad_norm"] if "max_grad_norm" in config else None
        self.eps = config["eps"] if "eps" in config else None
        self.init_optimizer()

        # Initialize learning rate schedule.
        if config["lr_schedule_type"] == "exponential":
            total_lr_decay = config["final_lr"] / config["initial_lr"]
            decay_per_epoch = math.pow(total_lr_decay, 1.0 / self.num_updates)
            self.lr_schedule = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=self.optimizer, gamma=decay_per_epoch,
            )

        elif config["lr_schedule_type"] == "cosine":
            self.lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=self.optimizer,
                T_max=self.num_updates,
                eta_min=config["final_lr"],
            )

        elif config["lr_schedule_type"] == "linear":

            def factor(step: int) -> float:
                lr_shift = config["final_lr"] - config["initial_lr"]
                desired_lr = config["initial_lr"] + lr_shift * float(step) / (
                    self.num_updates - 1
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

        self.steps = 0

    def init_model(self) -> None:
        """ Initialize model and corresponding objects. """
        raise NotImplementedError

    def init_optimizer(self) -> None:
        """ Initialize optimizer. """

        # Set optimizer to Adam if none was set in config.
        if "optimizer" not in self.config:
            self.config["optimizer"] = "adam"

        # Construct optimizer according to config.
        if self.config["optimizer"] == "adam":
            self.optimizer = torch.optim.Adam(
                self.parameters(), lr=self.initial_lr, eps=self.eps,
            )

        elif self.config["optimizer"] == "sgd":
            self.optimizer = torch.optim.SGD(
                self.parameters(), lr=self.initial_lr, momentum=self.momentum,
            )

        elif self.config["optimizer"] in ["sgdg", "adamg"]:
            self.optimizer = get_grassmann_optimizer(
                self.network, self.config["optimizer"], self.initial_lr, self.momentum, self.max_grad_norm
            )

        elif self.config["optimizer"] == "psi_sgd":
            assert self.max_grad_norm is None
            self.optimizer = get_PSI_optimizer(self.network, self.initial_lr, self.momentum)

        else:
            raise NotImplementedError

    def step(self) -> Dict[str, Any]:
        """ Perform a training step. """

        # Perform training step.
        step_metrics = self._step()

        # Step learning rate schedule, if necessary.
        if self.lr_schedule is not None:
            self.lr_schedule.step()

        # Increment index of training step.
        self.steps += 1

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
        """
        Clip gradients of model parameters, if necessary. If a Grassmann optimizer is
        used, gradient clipping is not performed here (it is performed in the optimizer
        instead).
        """
        grassmann = isinstance(self.optimizer, SGDG) or isinstance(self.optimizer, AdamG)
        if self.max_grad_norm is not None and not grassmann:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)

    @property
    def metric_set(self) -> None:
        """ Set of metrics for this trainer. """
        raise NotImplementedError
