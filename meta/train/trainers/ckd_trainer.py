""" Definition of CKDTrainer class. """

from typing import List, Tuple, Dict, Any, Iterator

import torch
import torch.nn as nn

from meta.networks.mlp import MLPNetwork
from meta.train.trainers.base_trainer import Trainer


class CKDTrainer(Trainer):
    """ Trainer class for Complete Knowledge Distillation. """

    def init_model(self, config: Dict[str, Any]) -> None:
        """
        Initialize model. `config` should contain all entries listed in the docstring of
        Trainer, as well as the settings specific to CKDTrainer, which are listed below.
        Note that CKDTrainer currently only supports MLPNetwork for both student and
        teacher.

        Parameters
        ----------
        teacher_path : str
            Path to file containing a saved torch Module (saved with `torch.save()`) to
            be used as the teacher network for knowledge distillation.
        num_updates : int
            Number of update steps.
        """

        self.teacher_path = config["teacher_path"]
        self.num_updates = config["num_updates"]

        # Load teacher model and disable gradient computation for its parameters.
        self.teacher = torch.load(self.teacher_path)
        self.teacher.requires_grad_(False)
        assert isinstance(self.teacher, MLPNetwork)

        # Construct student model.
        network_kwargs = dict(config["architecture_config"])
        assert config["architecture_config"]["type"] == "mlp"
        network_kwargs["input_size"] = self.teacher.input_size
        network_kwargs["output_size"] = self.teacher.output_size
        network_kwargs["device"] = self.device
        self.student = MLPNetwork(**network_kwargs)

        # Construct loss function.
        pass

    def _step(self) -> Dict[str, Any]:
        """ Perform one training step.  """
        assert NotImplementedError
        return {}

    def evaluate(self) -> None:
        """ Evaluate current model. """
        assert NotImplementedError
        return {}

    def load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """ Load trainer state from checkpoint. """
        assert NotImplementedError

    def get_checkpoint(self) -> None:
        """ Return trainer state as checkpoint. """
        assert NotImplementedError

    def close(self) -> None:
        """ Clean up the training process. """
        pass

    def parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
        """ Return parameters of model. """
        return self.student.parameters()

    @property
    def metric_set(self) -> List[Tuple]:
        """ Set of metrics for this trainer. """
        return []
