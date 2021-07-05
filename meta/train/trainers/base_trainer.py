""" Definition of Trainer base class. """

import math
from typing import Dict, Any

import numpy as np
import torch


class Trainer:
    """ Abstract base class for trainers. """

    def __init__(self, config: Dict[str, Any], **kwargs: Dict[str, Any]) -> None:
        """
        Init function for Trainer. The expected entries of `config` are listed below,
        though the concrete extensions of this class require more entries which are
        listed in their docstrings.

        Parameters
        ----------
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

    @property
    def metric_set(self) -> None:
        """ Set of metrics for this trainer. """
        raise NotImplementedError
